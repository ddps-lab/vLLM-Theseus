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

Host staging (``VLLM_P2P_NIXL_HOST_STAGING``, on by default): the engine
itself moves the rows between the GPU and the pinned regions / send buffers,
with asynchronous copies on the current stream (packed rows and partial sums
GPU -> host before the WRITE, received rows host -> GPU in the gather), so
UCX only sends host memory over TCP and makes no CUDA calls. With
``VLLM_P2P_NIXL_HOST_STAGING=0`` the regions are GPU memory and UCX copies
between the GPU and its TCP buffers itself, with a synchronous CUDA copy
inside ``transfer()``; in the AFD connector, which has the same layout, one
of those copies was seen never to return.

Progress thread (``VLLM_P2P_NIXL_PROGRESS_THREAD``, off by default): NIXL's
Python agent polls the UCX worker from its progress thread with no pause, so
that thread keeps a CPU core busy even when idle. Every host wait of this
engine polls NIXL itself, so the transfers progress without it. Without it
the agent uses ``NIXL_THREAD_SYNC_RW``, which keeps the UCX worker
thread-safe for the micro-batch threads that call it in turn.

Same principle as afd-plugin's ``NixlRoutedAFDConnector`` (the AFD one-hop
NIXL connector), written here because the EP all2all lives in vLLM and
cannot import the plugin. Limitation: unquantized rows only (no fp8 wire).
"""

from __future__ import annotations

import contextlib
import logging
import os
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
# VLLM_P2P_NIXL_HOST_STAGING=0: register GPU memory with NIXL instead of
# pinned host buffers (see "Host staging" in the module docstring).
_HOST_STAGING = os.environ.get("VLLM_P2P_NIXL_HOST_STAGING", "1") != "0"
# VLLM_P2P_NIXL_PROGRESS_THREAD=1: create the NIXL agent with its progress
# thread (see "Progress thread" in the module docstring).
_PROGRESS_THREAD = os.environ.get("VLLM_P2P_NIXL_PROGRESS_THREAD", "0") != "0"


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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from nixl import nixl_agent, nixl_agent_config  # noqa: PLC0415

        # Not ``nixl.nixl_thread_sync_t``: that name is the raw binding, and
        # the config only takes this enum.
        from nixl._api import nixl_thread_sync_t  # noqa: PLC0415

        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        # NIXL memory type of the regions: pinned host memory under host
        # staging, GPU memory otherwise; host memory when the engine runs on
        # CPU tensors (tests).
        self._host_staging = _HOST_STAGING
        on_gpu = self.device.type == "cuda" and not self._host_staging
        self._mem_type = "VRAM" if on_gpu else "DRAM"
        self._device_id = (self.device.index or 0) if on_gpu else 0
        self._progress_thread = _PROGRESS_THREAD
        self.agent = nixl_agent(
            _agent_name(self.rank),
            nixl_agent_config(
                backends=[NIXL_BACKEND],
                enable_prog_thread=self._progress_thread,
                sync_mode=None
                if self._progress_thread
                else nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
            ),
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
        # Under host staging, per dispatch key: the GPU buffer the rows are
        # packed into before they are copied into the pinned send buffer.
        self._packs: dict[tuple, torch.Tensor] = {}
        # (slot, "d" | "c") -> "the last gather out of that slot's regions
        # has run", so the host may overwrite this rank's own share in them.
        self._gathered: dict[tuple[int, str], Callable[[], bool]] = {}
        # Per region key: rank -> (base address, row bytes) of the peers'
        # regions and the peers' device ids, exchanged over the NCCL group.
        self._peer_regions: dict[tuple, list[tuple[int, int]]] = {}
        self._peer_devices: list[int] = []
        self._peer_agents: dict[int, str] = {}
        # slot -> WRITE handles posted and not yet seen complete.
        self._writes: dict[int, list[Any]] = {}
        self._notifs: dict[str, list[bytes]] = {}
        # (slot, "d" | "c") -> rows in flight towards this rank.
        self._pending: dict[tuple[int, str], _PendingReceive] = {}
        # dispatch: hidden, ids, weights (the prepare sends them in this
        # order, unquantized); combine: the partial sums, hidden-sized.
        for key in (
            ("recv0", (hidden_dim,), act_dtype),
            ("recv1", (top_k,), ids_dtype),
            ("recv2", (top_k,), weights_dtype),
            ("back", (hidden_dim,), act_dtype),
        ):
            self._alloc(key)
        self._register_agents()

    # ---------------------------------------------------------------- setup
    def _alloc(self, key: tuple) -> None:
        """Allocate and register the receive region and the send buffer of
        ``key`` (``(kind, tail, dtype)``), plus the gather buffer."""
        kind, tail, dtype = key
        rows = self.num_slots * self.world_size * self.max_tokens_per_rank
        where = torch.device("cpu") if self._host_staging else self.device
        pin = self._host_staging and self.device.type == "cuda"
        region = torch.empty((rows, *tail), dtype=dtype, device=where, pin_memory=pin)
        send = torch.empty((rows, *tail), dtype=dtype, device=where, pin_memory=pin)
        self.agent.register_memory([region, send])
        self._regions[key] = region
        self._sends[key] = send
        self._gathers[key] = torch.empty((rows, *tail), dtype=dtype, device=self.device)
        if self._host_staging and kind != "back":
            # combine_send copies the partial sums straight from the GPU.
            self._packs[key] = torch.empty(
                (rows, *tail), dtype=dtype, device=self.device
            )

    def _register_agents(self) -> None:
        """Exchange agent metadata (which carries the registered regions) and
        the region addresses; collective on the NCCL group."""
        mine = (
            self.agent.get_agent_metadata(),
            self._device_id,
            {
                key: (region.data_ptr(), region[0].numel() * region.element_size())
                for key, region in self._regions.items()
            },
            (self.num_slots, self._mem_type),
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        if {info[3] for info in everyone} != {(self.num_slots, self._mem_type)}:
            raise RuntimeError(
                "p2p_nixl: every rank needs the same micro-batch count and "
                "VLLM_P2P_NIXL_HOST_STAGING, got "
                f"{[info[3] for info in everyone]}"
            )
        self._peer_devices = [info[1] for info in everyone]
        for key in self._regions:
            self._peer_regions[key] = [info[2][key] for info in everyone]
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

    @property
    def host_staging(self) -> bool:
        """Whether the NIXL regions are pinned host memory (see the module
        docstring)."""
        return self._host_staging

    @property
    def progress_thread(self) -> bool:
        """Whether the NIXL agent runs its progress thread (see the module
        docstring)."""
        return self._progress_thread

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
            src.append((addr, count * row_bytes, self._device_id))
            dst.append(
                (
                    peer_addr + first_row * peer_row_bytes,
                    count * peer_row_bytes,
                    self._peer_devices[peer],
                )
            )
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(src, self._mem_type),
            self.agent.get_xfer_descs(dst, self._mem_type),
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
        is copied into this rank's own region of ``kind`` (on the current
        stream; by the host under host staging, after the region's previous
        gather has run), the others get one WRITE per peer (all keys in it,
        one notification). A dispatch notifies every peer with its row count
        (``<notif>:<rows>``, a bare notification for zero rows); a combine
        notifies only the peers it WRITES to."""
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
                    if self._host_staging and gathered is not None:
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
            if not self._host_staging:
                torch.cat(parts, out=gather[:num_in])
                continue
            # Pinned host -> GPU, asynchronous like a kernel.
            row = 0
            for part in parts:
                gather[row : row + part.shape[0]].copy_(part, non_blocking=True)
                row += part.shape[0]
        if self._host_staging:
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
            pack = self._packs.get(key)
            if num_send > 0:
                if pack is None:
                    torch.index_select(x, 0, plan.perm, out=rows_out)
                else:
                    # Pack on the GPU, then into the pinned send buffer on the
                    # same stream (_post waits for the stream before the WRITE).
                    packed_rows = pack[self._slot_rows(slot)][:num_send]
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
        # Under host staging GPU -> pinned host, asynchronous; _post waits for
        # the stream before the WRITE.
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


__all__ = ["P2pNixlAll2AllEngine"]
