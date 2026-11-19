# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[theseus p2p_nixl] Point-to-point MoE all2all with the rows carried by NIXL.

``P2pNixlAll2AllEngine`` keeps the plan of ``P2pAll2AllEngine`` (which ranks
every token visits, in destination order), the fixed-size launch bookkeeping
and the fp32 accumulation of the returned partials, and replaces the two data
movements, ``dispatch`` and ``combine_send``, and the count exchange: instead of
``all_to_all_single`` (NCCL kernels that hold the GPU until the peer's kernel
has moved its share) the rows travel as one-sided NIXL WRITEs into the
peer's pre-registered receive regions, followed by a notification. The GPU
runs no kernel while the rows are in flight; the wait is a host-side poll.

Layout: every rank registers, per received tensor (hidden, ids, weights,
...), one region of ``num_slots * world_size * R`` rows (``R =
max_tokens_per_rank``, one slot per micro-batch) for dispatch and one for
the partial sums coming back (combine), plus a send buffer of the same size
(NIXL moves data only between registered memory, so the packed rows are
staged there before the WRITE). Both are pinned host memory (host staging,
below). Rank ``s`` writes its share of slot ``u``
for rank ``d`` into rows ``[(u * world_size + s) * R, +R)`` of ``d``'s
regions; a token visits a rank at most once, so a share never exceeds ``R``
rows. The receive is then gathered (source-major, like the NCCL engine)
into the slot's gather buffer, so the prepare / finalize code and the
fixed-size launch are unchanged.

Posting and waiting are separate, so a micro-batch can hand the CPU away
while its rows are in flight (``deferred_receive``): ``dispatch`` and
``combine_send`` stage, WRITE and return at once; ``dispatch_wait`` and
``combine_accumulate`` poll for the notifications and gather. Every host
wait also drives the posted WRITEs to completion, because NIXL sends the
notification of a WRITE only when its state is checked after it finished.

The regions are registered at construction, before the agent metadata is
exchanged: NIXL's agent metadata carries the registered memory, and a peer
can only WRITE into memory it learnt about that way (registering later would
leave the peers with no backend able to reach the region). The tensor shapes
are therefore given to the constructor (hidden size, top-k, dtypes) and an
unexpected shape at dispatch is an error.

No count exchange: every dispatch notification carries its row count,
``d<seq>:<rows>``, and a rank sends one to every peer on every dispatch (a
bare notification, without a WRITE, when it has no rows for that peer). The
receiver learns what each peer sent when its rows have arrived, so no rank
waits for the others before sending. The plan sequence number advances
identically on every rank (one plan per layer and micro-batch). A combine
notification is ``c<seq>``: each rank knows how many partial sums come back
(the rows it sent).

A slot's region is safe to rewrite without a barrier: a rank sends the next
rows of a slot to a peer only after the partial sums of the previous rows it
sent that peer have come back, which the peer posts after its gather of those
rows has run (``_post`` waits for the stream first). Where a rank sent a peer
no rows, that peer had nothing to gather from its region.

Host staging: the regions and send buffers are pinned host memory, and the
engine itself moves the rows between them and the GPU with asynchronous
copies on the current stream (packed rows and partial sums GPU -> host
before the WRITE, received rows host -> GPU in the gather), so UCX only
sends host memory over TCP and makes no CUDA calls. With GPU memory
registered instead, UCX copies between the GPU and its TCP buffers itself,
with a synchronous CUDA copy inside ``transfer()``; in the AFD connector,
which has the same layout, one of those copies was seen never to return.

The NIXL agent runs NIXL's progress thread (NIXL's default).

[theseus p2p_delta] Delta dispatch (``delta_row_bytes`` > 0, driven by
``prepare_finalize/p2p_delta.py``): the hidden region is replaced by a byte
region of ``delta_row_bytes`` per row (the widest row format of
``p2p_delta.py``) and a word region (one int32 per row). A rank packs its
rows for a peer back to back at their own sizes, WRITEs them to the start of
its share and notifies ``d<seq>:<rows>:<bytes>``. The rows a rank sends
itself never leave the GPU: the prepare/finalize puts them after the
received rows, and their partial sums are copied into their dispatch place
before the peers' ones are gathered around them. With combine compression
(``combine_row_bytes`` > 0) the peers' partial sums travel as byte rows (a
``cbytes`` region) that the prepare/finalize rebuilds.

Same principle as afd-plugin's ``NixlRoutedAFDConnector`` (the AFD one-hop
NIXL connector), written here because the EP all2all lives in vLLM and
cannot import the plugin. Limitation: unquantized rows only (no fp8 wire).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .p2p_nccl_all2all import (
    P2pAll2AllEngine,
    P2pAll2AllPendingPlan,
    P2pAll2AllPlan,
    _current_stream,
)

logger = logging.getLogger(__name__)

NIXL_BACKEND = "UCX"
# A stalled peer surfaces as an error instead of a hang (same as the AFD
# connector's NCCL groups).
TRANSFER_TIMEOUT_S = 120.0
# NIXL memory type of the regions and send buffers: pinned host memory, device
# id 0 (see "Host staging" in the module docstring).
NIXL_MEM_TYPE = "DRAM"


def _agent_name(rank: int) -> str:
    return f"p2p-nixl-{rank}"


def _profiling() -> bool:
    """Whether a profiler is recording. A ``record_function`` range costs
    several microseconds of host time even when none is, so the ranges of
    this module are opened only while one records."""
    return torch.autograd.profiler._is_profiler_enabled


def _profile_range(name: str) -> contextlib.AbstractContextManager:
    """``record_function(name)`` while a profiler records, else nothing."""
    if _profiling():
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _mark(name: str) -> None:
    """A point in time in profiler traces: a range of zero length."""
    if _profiling():
        with torch.profiler.record_function(name):
            pass


@dataclass(frozen=True)
class _PendingReceive:
    """Rows WRITTEN to this rank, not yet waited for, and the region keys to
    gather them from. Dispatch: ``notif`` is the prefix ``d<seq>:`` of the
    peers' notifications (they carry the counts) and ``plan`` gets the counts
    once they are in. Combine: the exact ``c<seq>`` and the counts are
    known (``recv_counts``, the rows this rank sent)."""

    notif: bytes
    keys: list[tuple]
    recv_counts: list[int] | None = None
    plan: P2pAll2AllPlan | None = None
    # [theseus p2p_delta] bytes of a compressed combine row (0: plain rows)
    row_bytes: int = 0


class P2pNixlAll2AllEngine(P2pAll2AllEngine):
    """``P2pAll2AllEngine`` whose dispatch/combine rows travel as NIXL WRITEs."""

    deferred_receive = True

    def __init__(
        self,
        *args: Any,
        hidden_dim: int,
        top_k: int,
        act_dtype: torch.dtype,
        ids_dtype: torch.dtype = torch.int32,
        weights_dtype: torch.dtype = torch.float32,
        num_slots: int = 1,
        delta_row_bytes: int = 0,  # [theseus p2p_delta]
        combine_row_bytes: int = 0,  # [theseus p2p_delta]
        settings: str = "",  # [theseus p2p_delta]
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from nixl import nixl_agent, nixl_agent_config  # noqa: PLC0415

        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self.agent = nixl_agent(
            _agent_name(self.rank),
            nixl_agent_config(backends=[NIXL_BACKEND], enable_prog_thread=True),
        )
        # (kind, tail, dtype) -> local region of num_slots * world_size * R
        # rows, and the gather buffer of the same size that dispatch returns
        # (fixed-size launch wants one contiguous buffer per slot; the
        # region's shares are R apart).
        self._regions: dict[tuple, torch.Tensor] = {}
        self._gathers: dict[tuple, torch.Tensor] = {}
        # Registered send buffers (same size): the WRITE source must be
        # registered memory as well.
        self._sends: dict[tuple, torch.Tensor] = {}
        # Per dispatch key: the GPU buffer the rows are packed into before
        # they are copied into the pinned send buffer.
        self._packs: dict[tuple, torch.Tensor] = {}
        # (slot, "d" | "c") -> "the last gather out of that slot's regions
        # has run", so the host may overwrite this rank's own share in them.
        self._gathered: dict[tuple[int, str], Callable[[], bool]] = {}
        # Per region key: rank -> (base address, row bytes) of the peers'
        # regions, exchanged over the NCCL group.
        self._peer_regions: dict[tuple, list[tuple[int, int]]] = {}
        self._peer_agents: dict[int, str] = {}
        # slot -> WRITE handles posted and not yet seen complete.
        self._writes: dict[int, list[Any]] = {}
        self._notifs: dict[str, list[bytes]] = {}
        # (slot, "d" | "c") -> rows in flight towards this rank.
        self._pending: dict[tuple[int, str], _PendingReceive] = {}
        # dispatch: hidden, ids, weights (the prepare sends them in this
        # order, unquantized); combine: the partial sums, hidden-sized.
        # [theseus p2p_delta] delta dispatch: byte rows of at most
        # delta_row_bytes and one int32 word per row instead of the hidden
        # rows; compressed combine rows of at most combine_row_bytes;
        # settings: the delta settings every rank must share.
        self.delta_row_bytes = delta_row_bytes
        self.combine_row_bytes = combine_row_bytes
        self.settings = settings
        hidden_keys = (
            [("dbytes", (delta_row_bytes,), torch.uint8), ("dwords", (1,), torch.int32)]
            if delta_row_bytes
            else [("recv0", (hidden_dim,), act_dtype)]
        )
        routing_keys = [
            ("recv1", (top_k,), ids_dtype),
            ("recv2", (top_k,), weights_dtype),
        ]
        combine_keys = [("back", (hidden_dim,), act_dtype)]
        if combine_row_bytes:
            combine_keys.append(("cbytes", (combine_row_bytes,), torch.uint8))
        for key in (*hidden_keys, *routing_keys, *combine_keys):
            self._alloc(key)
        # Delta dispatch keys in packing order: bytes, words, ids, weights.
        self._delta_keys = [*hidden_keys, *routing_keys] if delta_row_bytes else []
        # slot -> its views of the delta pack buffers (delta_packs).
        self._delta_pack_views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._register_agents()

    # ---------------------------------------------------------------- setup
    def _alloc(self, key: tuple) -> None:
        """Allocate and register the receive region and the send buffer of
        ``key`` (``(kind, tail, dtype)``), plus the gather buffer."""
        kind, tail, dtype = key
        rows = self.num_slots * self.world_size * self.max_tokens_per_rank
        # Pinned host memory (plain host memory when the engine runs on CPU
        # tensors in tests).
        pin = self.device.type == "cuda"
        region = torch.empty((rows, *tail), dtype=dtype, device="cpu", pin_memory=pin)
        send = torch.empty((rows, *tail), dtype=dtype, device="cpu", pin_memory=pin)
        self.agent.register_memory([region, send])
        self._regions[key] = region
        self._sends[key] = send
        self._gathers[key] = torch.empty((rows, *tail), dtype=dtype, device=self.device)
        if kind not in ("back", "cbytes"):
            # combine_send copies the partial sums straight from the GPU
            # ([theseus p2p_delta] compressed ones too).
            self._packs[key] = torch.empty(
                (rows, *tail), dtype=dtype, device=self.device
            )

    def _register_agents(self) -> None:
        """Exchange agent metadata (which carries the registered regions) and
        the region addresses; collective on the NCCL group."""
        mine = (
            self.agent.get_agent_metadata(),
            {
                key: (region.data_ptr(), region[0].numel() * region.element_size())
                for key, region in self._regions.items()
            },
            self.num_slots,
            self.settings,  # [theseus p2p_delta]
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        if {info[2] for info in everyone} != {self.num_slots}:
            raise RuntimeError(
                "p2p_nixl: every rank needs the same micro-batch count, got "
                f"{[info[2] for info in everyone]}"
            )
        # [theseus p2p_delta] the byte rows and words only line up when every
        # rank runs the same delta dispatch settings.
        if {info[3] for info in everyone} != {self.settings}:
            raise RuntimeError(
                "p2p_nixl: ranks differ in their delta dispatch settings: "
                f"{[info[3] for info in everyone]}"
            )
        for key in self._regions:
            self._peer_regions[key] = [info[1][key] for info in everyone]
        for peer, (metadata, _, _, _) in enumerate(everyone):
            if peer == self.rank:
                continue
            # add_remote_agent returns the name as bytes while notifications
            # are keyed by str, so peers are addressed by the known name.
            self.agent.add_remote_agent(metadata)
            self._peer_agents[peer] = _agent_name(peer)

    def _region(self, kind: str, tail: tuple[int, ...], dtype: torch.dtype) -> tuple:
        """Key of the registered region for one tensor shape (registered at
        construction; an unexpected shape cannot be added later, see the
        module docstring)."""
        key = (kind, tuple(tail), dtype)
        if key not in self._regions:
            raise RuntimeError(
                f"p2p_nixl: no registered region for {key}; registered: "
                f"{sorted(self._regions)}"
            )
        return key

    def _slot_rows(self, slot: int) -> slice:
        """Rows of ``slot`` in a region / send / gather buffer."""
        size = self.world_size * self.max_tokens_per_rank
        return slice(slot * size, (slot + 1) * size)

    # ------------------------------------------------------------- transfer
    def _stream_done(self) -> Callable[[], bool]:
        """Predicate for "the work queued so far on the current stream has
        run" (always true on CPU, where the ops are synchronous)."""
        if self.device.type != "cuda":
            return lambda: True
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event.query

    def exchange_counts(self, pending: P2pAll2AllPendingPlan) -> None:
        """Nothing to exchange: the notifications carry the counts (see the
        module docstring)."""
        return None

    def finish_plan(self, pending: P2pAll2AllPendingPlan) -> P2pAll2AllPlan:
        # The host copy of this rank's send counts would block in the stream
        # sync; poll for the plan kernels instead so the posted WRITEs are
        # serviced (their notifications only go out when their state is
        # checked) meanwhile. The receive counts arrive with the rows.
        with _profile_range("p2p_nixl.plan_wait"):
            self._wait_until(self._stream_done(), "plan")
        return self.finish_plan_local(pending)

    def _write(
        self,
        slot: int,
        peer: int,
        local: list[tuple[tuple, int, int]],
        count: int,
        notif: bytes,
    ) -> None:
        """One WRITE of ``count`` rows from every ``(key, address, row bytes)``
        of ``local`` (rows of this rank's registered send buffers) into this
        rank's share of ``slot`` in ``peer``'s regions, followed by one
        ``notif``. The descriptors are built from addresses, which costs less
        host time than slicing the tensors for NIXL to read back."""
        first_row = (slot * self.world_size + self.rank) * self.max_tokens_per_rank
        src: list[tuple[int, int, int]] = []
        dst: list[tuple[int, int, int]] = []
        for key, addr, row_bytes in local:
            peer_addr, peer_row_bytes = self._peer_regions[key][peer]
            src.append((addr, count * row_bytes, 0))
            dst.append(
                (peer_addr + first_row * peer_row_bytes, count * peer_row_bytes, 0)
            )
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(src, NIXL_MEM_TYPE),
            self.agent.get_xfer_descs(dst, NIXL_MEM_TYPE),
            self._peer_agents[peer],
            notif,
        )
        state = self.agent.transfer(handle)
        _mark(f"p2p_nixl.write_post[{slot}:{notif.decode()}:{peer}]")
        if state == "ERR":
            raise RuntimeError(f"p2p_nixl: WRITE to rank {peer} failed to post")
        self._writes.setdefault(slot, []).append(handle)

    def _service_writes(self) -> None:
        """Release the WRITEs that have completed, without blocking.

        NIXL sends the notification of a WRITE that was still in flight when
        it was posted only once its state is checked, so every host wait
        calls this in its loop and the peers see the rows promptly.
        """
        for slot, handles in list(self._writes.items()):
            remaining: list[Any] = []
            for handle in handles:
                state = self.agent.check_xfer_state(handle)
                if state == "PROC":
                    remaining.append(handle)
                elif state == "DONE":
                    self.agent.release_xfer_handle(handle)
                    _mark(f"p2p_nixl.write_done[{slot}]")
                else:
                    raise RuntimeError(f"p2p_nixl: WRITE ended in state {state}")
            if remaining:
                self._writes[slot] = remaining
            else:
                del self._writes[slot]

    def _wait_until(self, done: Callable[[], bool], what: str) -> None:
        """Poll ``done`` (servicing the WRITEs meanwhile) until it holds."""
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        while True:
            self._service_writes()
            if done():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"p2p_nixl: {what} did not finish in time")

    def _finish_writes(self, slot: int) -> None:
        """Wait until the WRITEs posted for ``slot`` have completed."""
        self._wait_until(lambda: slot not in self._writes, "WRITE")

    def _collect_notifs(self) -> None:
        for name, notifs in self.agent.get_new_notifs().items():
            for notif in notifs:
                _mark(f"p2p_nixl.notif_in[{notif.decode()}:{name}]")
            self._notifs.setdefault(name, []).extend(notifs)

    def _wait_notifs(self, expected: dict[int, bytes]) -> None:
        """Wait for ``expected[peer]`` from every listed peer."""
        missing = {self._peer_agents[p]: n for p, n in expected.items()}

        def done() -> bool:
            self._collect_notifs()
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            return not missing

        self._wait_until(done, f"notification from {sorted(missing)}")

    def _wait_counts(self, prefix: bytes, peers: list[int]) -> dict[int, int]:
        """Wait for the notification ``<prefix><rows>`` from every peer of
        ``peers`` and return the row count each one announced."""
        missing = {self._peer_agents[p]: p for p in peers}
        counts: dict[int, int] = {}

        def done() -> bool:
            self._collect_notifs()
            for name in list(missing):
                notifs = self._notifs.get(name, [])
                for notif in notifs:
                    if notif.startswith(prefix):
                        notifs.remove(notif)
                        counts[missing.pop(name)] = int(notif[len(prefix) :])
                        break
            return not missing

        self._wait_until(done, f"notification {prefix!r} from {sorted(missing)}")
        return counts

    def _post(
        self,
        slot: int,
        kind: str,
        packed: list[tuple[tuple, torch.Tensor]],
        send_counts: list[int],
        notif: bytes,
    ) -> None:
        """Move every ``(key, rows)`` of ``packed`` (destination-major,
        ``send_counts``, views of the slot's registered send buffers) to its
        owners once the current stream has produced the rows: the local share
        is copied into this rank's own region of ``kind`` (by the host, after
        the region's previous gather has run), the others get one WRITE per
        peer (all keys in it, one notification). A dispatch notifies every
        peer with its row count (``<notif>:<rows>``, a bare notification for
        zero rows); a combine notifies only the peers it WRITES to."""
        R = self.max_tokens_per_rank
        self._wait_until(self._stream_done(), "packing")
        # (key, address of the first row, row bytes) of every packed tensor.
        sources = [
            (key, rows.data_ptr(), rows.stride(0) * rows.element_size())
            for key, rows in packed
        ]
        offset = 0
        for peer, count in enumerate(send_counts):
            if peer == self.rank:
                if count:
                    gathered = self._gathered.get((slot, kind))
                    if gathered is not None:
                        self._wait_until(gathered, "previous gather")
                    for key, rows in packed:
                        region = self._regions[key][self._slot_rows(slot)]
                        region[self.rank * R : self.rank * R + count].copy_(
                            rows[offset : offset + count]
                        )
            elif count:
                local = [(key, addr + offset * rb, rb) for key, addr, rb in sources]
                tagged = notif + f":{count}".encode() if kind == "d" else notif
                self._write(slot, peer, local, count, tagged)
            elif kind == "d":
                tagged = notif + b":0"
                self.agent.send_notif(self._peer_agents[peer], tagged)
                _mark(f"p2p_nixl.write_post[{slot}:{tagged.decode()}:{peer}]")
            offset += count

    def _receive(self, slot: int, kind: str) -> None:
        """Wait for the rows of (``slot``, ``kind``) and gather each key's
        region source-major into the head of the slot's gather buffer on the
        current stream. The shares are at most R rows each so the head
        always fits."""
        pending = self._pending.pop((slot, kind))
        self._finish_writes(slot)
        if pending.plan is not None:
            # Dispatch: every peer announces its rows (zero included).
            plan = pending.plan
            counts = self._wait_counts(
                pending.notif, [p for p in range(self.world_size) if p != self.rank]
            )
            counts[self.rank] = plan.send_counts[self.rank]
            recv_counts = [counts[p] for p in range(self.world_size)]
            self.set_recv_counts(plan, recv_counts)
        else:
            recv_counts = pending.recv_counts
            self._wait_notifs(
                {
                    p: pending.notif
                    for p, c in enumerate(recv_counts)
                    if c and p != self.rank
                }
            )
        R = self.max_tokens_per_rank
        num_in = sum(recv_counts)
        for key in pending.keys:
            region = self._regions[key][self._slot_rows(slot)]
            gather = self._gathers[key][self._slot_rows(slot)]
            if pending.plan is not None:
                # The rows past num_in may still hold ids of an earlier
                # dispatch (see mask_rows_past), now that num_in is known.
                dirty = (slot, gather.data_ptr())
                if dirty in self._dirty_rows:
                    self._dirty_rows[dirty] = max(self._dirty_rows[dirty], num_in)
            parts = [region[p * R : p * R + c] for p, c in enumerate(recv_counts)]
            if not num_in:
                continue
            # Pinned host -> GPU, asynchronous like a kernel.
            row = 0
            for part in parts:
                gather[row : row + part.shape[0]].copy_(part, non_blocking=True)
                row += part.shape[0]
        self._gathered[(slot, kind)] = self._stream_done()

    def dispatch(
        self,
        plan: P2pAll2AllPlan,
        tensors: tuple[torch.Tensor, ...],
        min_rows: int = 0,
    ) -> list[torch.Tensor]:
        """Same contract as the NCCL engine's ``dispatch`` (see there); the
        rows travel as NIXL WRITEs and are only valid after ``dispatch_wait``,
        which also fills in the plan's receive counts. The views cover
        ``min_rows`` rows (fixed-size launch) or the slot's whole share of the
        gather buffer (the caller trims them to ``num_recv`` after the wait);
        the rows past ``num_recv`` are handled by ``mask_rows_past`` exactly as
        for the NCCL receive buffers."""
        slot = plan.slot
        if slot >= self.num_slots:
            raise RuntimeError(
                f"p2p_nixl: slot {slot} but the engine has {self.num_slots} "
                "micro-batch slots"
            )
        if (slot, "d") in self._pending:
            raise RuntimeError(
                f"p2p_nixl: slot {slot} already has a dispatch in flight"
            )
        outputs: list[torch.Tensor] = []
        stream = _current_stream(self.device)
        num_send = plan.num_send
        if stream is not None and num_send > 0:
            plan.perm.record_stream(stream)
        notif = f"d{plan.seq}".encode()
        keys: list[tuple] = []
        packed: list[tuple[tuple, torch.Tensor]] = []
        # The receive counts are only known once the rows arrive.
        rows = min_rows or self.world_size * self.max_tokens_per_rank
        for i, x in enumerate(tensors):
            if x.shape[0] != plan.num_tokens:
                raise ValueError(
                    f"tensor {i} has {x.shape[0]} rows, plan has {plan.num_tokens}"
                )
            if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise RuntimeError("p2p_nixl carries unquantized rows only")
            key = self._region(f"recv{i}", tuple(x.shape[1:]), x.dtype)
            rows_out = self._sends[key][self._slot_rows(slot)][:num_send]
            if num_send > 0:
                # Pack on the GPU, then into the pinned send buffer on the
                # same stream (_post waits for the stream before the WRITE).
                packed_rows = self._packs[key][self._slot_rows(slot)][:num_send]
                torch.index_select(x, 0, plan.perm, out=packed_rows)
                rows_out.copy_(packed_rows, non_blocking=True)
            packed.append((key, rows_out))
            keys.append(key)
            outputs.append(self._gathers[key][self._slot_rows(slot)][:rows])
        # One WRITE per peer carrying hidden, ids and weights, so one
        # notification per peer announces the whole share.
        with _profile_range(f"p2p_nixl.dispatch_post[{slot}]"):
            self._post(slot, "d", packed, plan.send_counts, notif)
        self._pending[(slot, "d")] = _PendingReceive(
            notif=notif + b":", keys=keys, plan=plan
        )
        return outputs

    def dispatch_wait(self, plan: P2pAll2AllPlan) -> None:
        """Wait for the rows of ``plan``'s dispatch; the views ``dispatch``
        returned hold them afterwards (gathered on the current stream)."""
        with _profile_range(f"p2p_nixl.dispatch_wait[{plan.slot}]"):
            self._receive(plan.slot, "d")

    def combine_send(self, plan: P2pAll2AllPlan, partial: torch.Tensor) -> torch.Tensor:
        """Return ``partial`` (``[num_recv, ...]``) to the owners as NIXL WRITEs;
        same contract as the NCCL engine's ``combine_send``, the returned rows
        being valid once ``combine_accumulate`` has waited for them."""
        if plan.num_recv is None:
            raise RuntimeError("p2p_nixl: combine_send before dispatch_wait")
        if partial.shape[0] != plan.num_recv:
            raise ValueError(
                f"partial has {partial.shape[0]} rows, plan expects {plan.num_recv}"
            )
        slot = plan.slot
        if (slot, "c") in self._pending:
            raise RuntimeError(f"p2p_nixl: slot {slot} already has a combine in flight")
        # Stage into the slot's registered send buffer. Going back: this rank
        # sends recv_counts (source-major rows it holds) and receives
        # send_counts (its own tokens' partials, perm order).
        key = self._region("back", tuple(partial.shape[1:]), partial.dtype)
        staged = self._sends[key][self._slot_rows(slot)][: plan.num_recv]
        # GPU -> pinned host, asynchronous; _post waits for the stream before
        # the WRITE.
        staged.copy_(partial, non_blocking=True)
        notif = f"c{plan.seq}".encode()
        with _profile_range(f"p2p_nixl.combine_post[{slot}]"):
            self._post(slot, "c", [(key, staged)], plan.recv_counts, notif)
        self._pending[(slot, "c")] = _PendingReceive(
            notif=notif, keys=[key], recv_counts=plan.send_counts
        )
        return self._gathers[key][self._slot_rows(slot)][: plan.num_send]

    def combine_accumulate(
        self, plan: P2pAll2AllPlan, back: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Wait for the partial sums of ``plan``'s combine, then sum them into
        ``out`` as the NCCL engine does."""
        with _profile_range(f"p2p_nixl.combine_wait[{plan.slot}]"):
            self._receive(plan.slot, "c")
        return super().combine_accumulate(plan, back, out)

    # ------------------------------------------- [theseus p2p_delta] delta
    def wait_stream(self, what: str) -> None:
        """Wait (servicing the WRITEs) until the current stream has run the
        work queued so far, e.g. a copy into host memory about to be read."""
        self._wait_until(self._stream_done(), what)

    def delta_packs(self, slot: int) -> tuple[torch.Tensor, ...]:
        """The GPU buffers a delta dispatch of ``slot`` is packed into, peer
        major: bytes and words (flat), expert ids and weights (rows)."""
        packs = self._delta_pack_views.get(slot)
        if packs is None:
            if not self.delta_row_bytes:
                raise RuntimeError("p2p_nixl: delta dispatch is off")
            rows = self._slot_rows(slot)
            by_key = [self._packs[key][rows] for key in self._delta_keys]
            packs = (by_key[0].view(-1), by_key[1].view(-1), by_key[2], by_key[3])
            self._delta_pack_views[slot] = packs
        return packs

    def _write_spans(
        self,
        slot: int,
        peer: int,
        spans: list[tuple[tuple, int, int]],
        notif: bytes,
    ) -> None:
        """``_write`` for spans that are not whole rows: one WRITE of every
        ``(key, address, bytes)`` of ``spans`` (this rank's registered send
        memory) to the start of this rank's share of ``slot`` in ``peer``'s
        region of ``key``, followed by ``notif``."""
        R = self.max_tokens_per_rank
        first_row = (slot * self.world_size + self.rank) * R
        src: list[tuple[int, int, int]] = []
        dst: list[tuple[int, int, int]] = []
        for key, addr, size in spans:
            peer_addr, peer_row_bytes = self._peer_regions[key][peer]
            if size > R * peer_row_bytes:
                raise RuntimeError(
                    f"p2p_nixl: {size} bytes exceed the {key[0]} share of "
                    f"{R * peer_row_bytes}"
                )
            src.append((addr, size, 0))
            dst.append((peer_addr + first_row * peer_row_bytes, size, 0))
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(src, NIXL_MEM_TYPE),
            self.agent.get_xfer_descs(dst, NIXL_MEM_TYPE),
            self._peer_agents[peer],
            notif,
        )
        state = self.agent.transfer(handle)
        _mark(f"p2p_nixl.write_post[{slot}:{notif.decode()}:{peer}]")
        if state == "ERR":
            raise RuntimeError(f"p2p_nixl: WRITE to rank {peer} failed to post")
        self._writes.setdefault(slot, []).append(handle)

    def _wait_rows_bytes(
        self, prefix: bytes, peers: list[int]
    ) -> dict[int, tuple[int, int]]:
        """``_wait_counts`` for the delta notifications
        ``<prefix><rows>:<bytes>``: (rows, bytes) per peer."""
        missing = {self._peer_agents[p]: p for p in peers}
        got: dict[int, tuple[int, int]] = {}

        def done() -> bool:
            self._collect_notifs()
            for name in list(missing):
                notifs = self._notifs.get(name, [])
                for notif in notifs:
                    if notif.startswith(prefix):
                        notifs.remove(notif)
                        rows, _, size = notif[len(prefix) :].partition(b":")
                        got[missing.pop(name)] = (int(rows), int(size))
                        break
            return not missing

        self._wait_until(done, f"notification {prefix!r} from {sorted(missing)}")
        return got

    def dispatch_delta(
        self,
        plan: P2pAll2AllPlan,
        counts: list[int],
        nbytes: list[int],
        min_rows: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """Send the rows packed in ``delta_packs(plan.slot)``: ``counts[p]``
        rows of ``nbytes[p]`` bytes to every peer ``p``, in peer order (zero
        for this rank, whose own rows stay on the GPU). Every peer gets one
        WRITE and ``d<seq>:<rows>:<bytes>`` (a bare notification for no
        rows). Returns the views ``delta_dispatch_wait`` fills: bytes and
        words (flat), expert ids and weights (``min_rows`` rows, or the
        slot's whole share)."""
        slot = plan.slot
        if not self.delta_row_bytes:
            raise RuntimeError("p2p_nixl: delta dispatch is off")
        if slot >= self.num_slots:
            raise RuntimeError(
                f"p2p_nixl: slot {slot} but the engine has {self.num_slots} "
                "micro-batch slots"
            )
        if (slot, "d") in self._pending:
            raise RuntimeError(
                f"p2p_nixl: slot {slot} already has a dispatch in flight"
            )
        if counts[self.rank] or nbytes[self.rank]:
            raise ValueError("p2p_nixl: a rank's own rows stay out of a delta dispatch")
        keys = self._delta_keys
        rows = self._slot_rows(slot)
        packs = self.delta_packs(slot)
        sends = [self._sends[key][rows] for key in keys]
        num, total = sum(counts), sum(nbytes)
        if num:
            # GPU -> pinned host on the current stream; the WRITEs wait for it.
            sends[0].view(-1)[:total].copy_(packs[0][:total], non_blocking=True)
            sends[1].view(-1)[:num].copy_(packs[1][:num], non_blocking=True)
            for send, pack in zip(sends[2:], packs[2:]):
                send[:num].copy_(pack[:num], non_blocking=True)
        with _profile_range(f"p2p_nixl.dispatch_post[{slot}]"):
            self._wait_until(self._stream_done(), "packing")
            row = offset = 0
            for peer, (count, size) in enumerate(zip(counts, nbytes)):
                if peer == self.rank:
                    continue
                notif = f"d{plan.seq}:{count}:{size}".encode()
                if count:
                    spans = [(keys[0], sends[0].data_ptr() + offset, size)]
                    for key, send in zip(keys[1:], sends[1:]):
                        row_bytes = send.stride(0) * send.element_size()
                        spans.append(
                            (key, send.data_ptr() + row * row_bytes, count * row_bytes)
                        )
                    self._write_spans(slot, peer, spans, notif)
                else:
                    self.agent.send_notif(self._peer_agents[peer], notif)
                    _mark(f"p2p_nixl.write_post[{slot}:{notif.decode()}:{peer}]")
                row += count
                offset += size
        self._pending[(slot, "d")] = _PendingReceive(
            notif=f"d{plan.seq}:".encode(), keys=list(keys), plan=plan
        )
        gathers = [self._gathers[key][rows] for key in keys]
        out_rows = min_rows or self.world_size * self.max_tokens_per_rank
        return (
            gathers[0].view(-1),
            gathers[1].view(-1),
            gathers[2][:out_rows],
            gathers[3][:out_rows],
        )

    def delta_dispatch_wait(
        self, plan: P2pAll2AllPlan, own: int
    ) -> list[tuple[int, int]]:
        """Wait for the rows of ``plan``'s delta dispatch and gather the
        peers' bytes, words, ids and weights source-major into the heads of
        the views ``dispatch_delta`` returned (the caller puts this rank's
        ``own`` rows after them). Fills in the plan's receive counts and
        returns (rows, bytes) per rank, ``(own, 0)`` for this one."""
        slot = plan.slot
        with _profile_range(f"p2p_nixl.dispatch_wait[{slot}]"):
            pending = self._pending.pop((slot, "d"))
            self._finish_writes(slot)
            got = self._wait_rows_bytes(
                pending.notif, [p for p in range(self.world_size) if p != self.rank]
            )
            got[self.rank] = (own, 0)
            info = [got[p] for p in range(self.world_size)]
            self.set_recv_counts(plan, [count for count, _ in info])
            assert plan.num_recv is not None
            R = self.max_tokens_per_rank
            rows = self._slot_rows(slot)
            peers = [
                (p, count, size)
                for p, (count, size) in enumerate(info)
                if p != self.rank and count
            ]
            # Bytes: each peer's rows at the head of its share, at their sizes.
            region = self._regions[pending.keys[0]][rows]
            gather = self._gathers[pending.keys[0]][rows].view(-1)
            offset = 0
            for p, _, size in peers:
                share = region[p * R : (p + 1) * R].view(-1)[:size]
                gather[offset : offset + size].copy_(share, non_blocking=True)
                offset += size
            for key in pending.keys[1:]:
                region = self._regions[key][rows]
                gather = self._gathers[key][rows]
                # The rows past num_recv may still hold ids of an earlier
                # dispatch (see mask_rows_past).
                dirty = (slot, gather.data_ptr())
                if dirty in self._dirty_rows:
                    self._dirty_rows[dirty] = max(
                        self._dirty_rows[dirty], plan.num_recv
                    )
                row = 0
                for p, count, _ in peers:
                    gather[row : row + count].copy_(
                        region[p * R : p * R + count], non_blocking=True
                    )
                    row += count
        return info

    def delta_combine_send(
        self,
        plan: P2pAll2AllPlan,
        partial: torch.Tensor,
        own: int,
        compressed: torch.Tensor | None = None,
        row_bytes: int = 0,
    ) -> torch.Tensor:
        """Return the partial sums of a delta dispatch: ``partial``
        (``[num_recv, ...]``) holds the peers' rows source-major, then this
        rank's ``own`` rows. The peers' rows go back as WRITEs (``c<seq>``),
        as they are or ``compressed`` (uint8, flat: the same rows at
        ``row_bytes`` each); the own rows are copied on the GPU into their
        dispatch place of the returned ``[num_send, ...]`` view, which
        ``delta_combine_wait`` completes (plain rows) or leaves to the
        caller (compressed rows)."""
        if plan.num_recv is None or plan.recv_counts is None:
            raise RuntimeError("p2p_nixl: combine before delta_dispatch_wait")
        if partial.shape[0] != plan.num_recv:
            raise ValueError(
                f"partial has {partial.shape[0]} rows, plan expects {plan.num_recv}"
            )
        if own != plan.send_counts[self.rank] or own != plan.recv_counts[self.rank]:
            raise ValueError(f"p2p_nixl: {own} own rows do not match the plan")
        slot = plan.slot
        if (slot, "c") in self._pending:
            raise RuntimeError(f"p2p_nixl: slot {slot} already has a combine in flight")
        key = self._region("back", tuple(partial.shape[1:]), partial.dtype)
        rows = self._slot_rows(slot)
        num_in = plan.num_recv - own
        # GPU -> pinned host, asynchronous; the WRITEs wait for the stream.
        if compressed is None:
            wire, size = key, partial.shape[1] * partial.element_size()
            staged = self._sends[key][rows][:num_in]
            staged.copy_(partial[:num_in], non_blocking=True)
        else:
            wire = self._region("cbytes", (self.combine_row_bytes,), torch.uint8)
            size = row_bytes
            staged = self._sends[wire][rows].view(-1)[: num_in * size]
            staged.copy_(compressed[: num_in * size], non_blocking=True)
        back = self._gathers[key][rows][: plan.num_send]
        if own:
            start = sum(plan.send_counts[: self.rank])
            back[start : start + own].copy_(partial[num_in:])
        notif = f"c{plan.seq}".encode()
        with _profile_range(f"p2p_nixl.combine_post[{slot}]"):
            self._wait_until(self._stream_done(), "packing")
            base, row = staged.data_ptr(), 0
            for peer, count in enumerate(plan.recv_counts):
                if peer == self.rank:
                    continue
                if count:
                    span = (wire, base + row * size, count * size)
                    self._write_spans(slot, peer, [span], notif)
                row += count
        self._pending[(slot, "c")] = _PendingReceive(
            notif=notif,
            keys=[wire],
            recv_counts=plan.send_counts,
            row_bytes=0 if compressed is None else size,
        )
        return back

    def delta_combine_wait(self, plan: P2pAll2AllPlan) -> torch.Tensor | None:
        """Wait for the peers' partial sums of ``plan``'s delta combine.
        Plain rows go around this rank's own rows in the view
        ``delta_combine_send`` returned (returns None); compressed rows are
        gathered into one GPU byte buffer in dispatch order, the own block
        left out, which is returned for the caller to rebuild."""
        slot = plan.slot
        with _profile_range(f"p2p_nixl.combine_wait[{slot}]"):
            pending = self._pending.pop((slot, "c"))
            self._finish_writes(slot)
            counts = pending.recv_counts
            assert counts is not None
            self._wait_notifs(
                {p: pending.notif for p, c in enumerate(counts) if c and p != self.rank}
            )
            R = self.max_tokens_per_rank
            rows = self._slot_rows(slot)
            region = self._regions[pending.keys[0]][rows]
            gather = self._gathers[pending.keys[0]][rows]
            size = pending.row_bytes
            if size:
                flat, offset = gather.view(-1), 0
                for p, count in enumerate(counts):
                    if count and p != self.rank:
                        share = region[p * R : (p + 1) * R].view(-1)[: count * size]
                        flat[offset : offset + count * size].copy_(
                            share, non_blocking=True
                        )
                        offset += count * size
                return flat[:offset]
            row = 0
            for p, count in enumerate(counts):
                if count and p != self.rank:
                    gather[row : row + count].copy_(
                        region[p * R : p * R + count], non_blocking=True
                    )
                row += count
        return None

    def delta_accumulate(
        self, plan: P2pAll2AllPlan, back: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """``combine_accumulate`` once ``delta_combine_wait`` has run (and
        the caller has put the rebuilt compressed rows into ``back``)."""
        return super().combine_accumulate(plan, back, out)


__all__ = ["P2pNixlAll2AllEngine"]
