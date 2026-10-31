# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[theseus p2p_nixl] Point-to-point MoE all2all with the rows carried by NIXL.

``P2pNixlAll2AllEngine`` keeps everything of ``P2pAll2AllEngine`` (the
per-layer count exchange over NCCL, the plan, the fixed-size launch
bookkeeping, the fp32 accumulation of the returned partials) and replaces
only the two data movements, ``dispatch`` and ``combine_send``: instead of
``all_to_all_single`` (NCCL kernels that hold the GPU until the peer's kernel
has moved its share) the rows travel as one-sided NIXL WRITEs into the
peer's pre-registered receive regions, followed by a notification. The GPU
runs no kernel while the rows are in flight; the wait is a host-side poll.

Layout: every rank registers, per received tensor (hidden, ids, weights,
...), one region of ``num_slots * world_size * R`` rows (``R =
max_tokens_per_rank``, one slot per micro-batch) for dispatch and one for
the partial sums coming back (combine), plus a send buffer of the same size
(NIXL moves data only between registered memory, so the packed rows are
staged there before the WRITE). Rank ``s`` writes its share of slot ``u``
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

A slot's region is safe to rewrite once the next layer's count exchange of
that slot (NCCL, every rank takes part, in slot order) has happened: by then
the previous rows were consumed. The notification carries the plan sequence
number, which advances identically on every rank.

Same principle as afd-plugin's ``NixlRoutedAFDConnector`` (the AFD one-hop
NIXL connector), written here because the EP all2all lives in vLLM and
cannot import the plugin. Limitation: unquantized rows only (no fp8 wire).
"""

from __future__ import annotations

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


def _agent_name(rank: int) -> str:
    return f"p2p-nixl-{rank}"


@dataclass(frozen=True)
class _PendingReceive:
    """Rows WRITTEN to this rank, not yet waited for: per region key, the
    source-major counts to gather and the rows of the result view."""

    notif: bytes
    recv_counts: list[int]
    keys: list[tuple]
    out_rows: list[int]


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

        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        # NIXL memory type of the regions: GPU memory in serving, host memory
        # when the engine runs on CPU tensors (tests).
        self._mem_type = "VRAM" if self.device.type == "cuda" else "DRAM"
        self._device_id = self.device.index or 0
        self.agent = nixl_agent(
            _agent_name(self.rank), nixl_agent_config(backends=[NIXL_BACKEND])
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
        _, tail, dtype = key
        rows = self.num_slots * self.world_size * self.max_tokens_per_rank
        region = torch.empty((rows, *tail), dtype=dtype, device=self.device)
        send = torch.empty_like(region)
        self.agent.register_memory([region, send])
        self._regions[key] = region
        self._sends[key] = send
        self._gathers[key] = torch.empty_like(region)

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
            self.num_slots,
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        if {info[3] for info in everyone} != {self.num_slots}:
            raise RuntimeError(
                "p2p_nixl: every rank needs the same micro-batch count, got "
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

    def finish_plan(self, pending: P2pAll2AllPendingPlan) -> P2pAll2AllPlan:
        # The host copy of the counts would block in the stream sync; poll
        # for the exchange instead so the posted WRITEs are serviced (their
        # notifications only go out when their state is checked) meanwhile.
        with torch.profiler.record_function("p2p_nixl.count_wait"):
            self._wait_until(self._stream_done(), "count exchange")
        return super().finish_plan(pending)

    def _write(
        self,
        slot: int,
        peer: int,
        shares: list[tuple[tuple, torch.Tensor]],
        notif: bytes,
    ) -> None:
        """One WRITE of every ``(key, rows)`` of ``shares`` into this rank's
        share of ``slot`` in ``peer``'s regions, followed by one ``notif``."""
        first_row = (slot * self.world_size + self.rank) * self.max_tokens_per_rank
        local: list[torch.Tensor] = []
        remote: list[tuple[int, int, int]] = []
        for key, rows in shares:
            addr, row_bytes = self._peer_regions[key][peer]
            local.append(rows)
            remote.append(
                (
                    addr + first_row * row_bytes,
                    rows.shape[0] * row_bytes,
                    self._peer_devices[peer],
                )
            )
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(local),
            self.agent.get_xfer_descs(remote, self._mem_type),
            self._peer_agents[peer],
            notif,
        )
        if self.agent.transfer(handle) == "ERR":
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

    def _wait_notifs(self, expected: dict[int, bytes]) -> None:
        """Wait for ``expected[peer]`` from every listed peer."""
        missing = {self._peer_agents[p]: n for p, n in expected.items()}

        def done() -> bool:
            for name, notifs in self.agent.get_new_notifs().items():
                self._notifs.setdefault(name, []).extend(notifs)
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            return not missing

        self._wait_until(done, f"notification from {sorted(missing)}")

    def _post(
        self,
        slot: int,
        packed: list[tuple[tuple, torch.Tensor]],
        send_counts: list[int],
        notif: bytes,
    ) -> None:
        """Move every ``(key, rows)`` of ``packed`` (destination-major,
        ``send_counts``, views of the slot's registered send buffers) to its
        owners: the local share is copied on the current stream, the others
        get one WRITE per peer (all keys in it, one notification) once the
        current stream has produced the rows."""
        R = self.max_tokens_per_rank
        self._wait_until(self._stream_done(), "packing")
        offset = 0
        for peer, count in enumerate(send_counts):
            if count:
                shares = [(key, rows[offset : offset + count]) for key, rows in packed]
                if peer == self.rank:
                    for key, share in shares:
                        region = self._regions[key][self._slot_rows(slot)]
                        region[self.rank * R : self.rank * R + count].copy_(share)
                else:
                    self._write(slot, peer, shares, notif)
            offset += count

    def _receive(self, slot: int, kind: str) -> None:
        """Wait for the rows of (``slot``, ``kind``) and gather each key's
        region source-major into the head of the slot's gather buffer on the
        current stream. The shares are at most R rows each so the head
        always fits."""
        pending = self._pending.pop((slot, kind))
        self._finish_writes(slot)
        self._wait_notifs(
            {
                p: pending.notif
                for p, c in enumerate(pending.recv_counts)
                if c and p != self.rank
            }
        )
        R = self.max_tokens_per_rank
        num_in = sum(pending.recv_counts)
        for key in pending.keys:
            region = self._regions[key][self._slot_rows(slot)]
            gather = self._gathers[key][self._slot_rows(slot)]
            parts = [
                region[p * R : p * R + c] for p, c in enumerate(pending.recv_counts)
            ]
            if num_in:
                torch.cat(parts, out=gather[:num_in])

    def dispatch(
        self,
        plan: P2pAll2AllPlan,
        tensors: tuple[torch.Tensor, ...],
        min_rows: int = 0,
    ) -> list[torch.Tensor]:
        """Same contract as the NCCL engine's ``dispatch`` (see there); the
        rows travel as NIXL WRITEs and are only valid after ``dispatch_wait``.
        ``min_rows`` views come from the slot's gather buffer, whose rows past
        ``num_recv`` are handled by ``mask_rows_past`` exactly as for the NCCL
        receive buffers."""
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
        out_rows: list[int] = []
        packed: list[tuple[tuple, torch.Tensor]] = []
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
                torch.index_select(x, 0, plan.perm, out=rows_out)
            packed.append((key, rows_out))
            rows = max(plan.num_recv, min_rows)
            recv = self._gathers[key][self._slot_rows(slot)][:rows]
            dirty = (slot, recv.data_ptr())
            if dirty in self._dirty_rows:
                self._dirty_rows[dirty] = max(self._dirty_rows[dirty], plan.num_recv)
            keys.append(key)
            out_rows.append(rows)
            outputs.append(recv)
        # One WRITE per peer carrying hidden, ids and weights, so one
        # notification per peer announces the whole share.
        with torch.profiler.record_function(f"p2p_nixl.dispatch_post[{slot}]"):
            self._post(slot, packed, plan.send_counts, notif)
        self._pending[(slot, "d")] = _PendingReceive(
            notif=notif, recv_counts=plan.recv_counts, keys=keys, out_rows=out_rows
        )
        return outputs

    def dispatch_wait(self, plan: P2pAll2AllPlan) -> None:
        """Wait for the rows of ``plan``'s dispatch; the views ``dispatch``
        returned hold them afterwards (gathered on the current stream)."""
        with torch.profiler.record_function(f"p2p_nixl.dispatch_wait[{plan.slot}]"):
            self._receive(plan.slot, "d")

    def combine_send(self, plan: P2pAll2AllPlan, partial: torch.Tensor) -> torch.Tensor:
        """Return ``partial`` (``[num_recv, ...]``) to the owners as NIXL WRITEs;
        same contract as the NCCL engine's ``combine_send``, the returned rows
        being valid once ``combine_accumulate`` has waited for them."""
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
        staged.copy_(partial)
        notif = f"c{plan.seq}".encode()
        with torch.profiler.record_function(f"p2p_nixl.combine_post[{slot}]"):
            self._post(slot, [(key, staged)], plan.recv_counts, notif)
        self._pending[(slot, "c")] = _PendingReceive(
            notif=notif,
            recv_counts=plan.send_counts,
            keys=[key],
            out_rows=[plan.num_send],
        )
        return self._gathers[key][self._slot_rows(slot)][: plan.num_send]

    def combine_accumulate(
        self, plan: P2pAll2AllPlan, back: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Wait for the partial sums of ``plan``'s combine, then sum them into
        ``out`` as the NCCL engine does."""
        with torch.profiler.record_function(f"p2p_nixl.combine_wait[{plan.slot}]"):
            self._receive(plan.slot, "c")
        return super().combine_accumulate(plan, back, out)


__all__ = ["P2pNixlAll2AllEngine"]
