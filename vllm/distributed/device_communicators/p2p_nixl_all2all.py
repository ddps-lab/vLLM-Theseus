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
...), one region of ``world_size * R`` rows (``R = max_tokens_per_rank``) for
dispatch and one for the partial sums coming back (combine), plus a send
buffer of the same size (NIXL moves data only between registered memory,
so the packed rows are staged there before the WRITE). Rank ``s`` writes
its share for rank ``d`` into rows ``[s*R, (s+1)*R)`` of ``d``'s regions; a
token visits a rank at most once, so a share never exceeds ``R`` rows.
``dispatch`` then gathers the shares (source-major, like the NCCL engine)
into the receive buffer the caller expects, so the prepare / finalize code
and the fixed-size launch are unchanged.

A region is safe to rewrite once the next layer's count exchange (NCCL,
every rank takes part) has happened: by then the previous rows were
consumed. The notification carries the plan sequence number, which advances
identically on every rank.

Same principle as afd-plugin's ``NixlRoutedAFDConnector`` (the AFD one-hop
NIXL connector), written here because the EP all2all lives in vLLM and
cannot import the plugin. Limitations of this first version: unquantized
rows only (no fp8 wire) and no dual batch overlap; both are refused at
construction by the prepare/finalize.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.distributed as dist

from .p2p_nccl_all2all import P2pAll2AllEngine, P2pAll2AllPlan, _current_stream

logger = logging.getLogger(__name__)

NIXL_BACKEND = "UCX"
# A stalled peer surfaces as an error instead of a hang (same as the AFD
# connector's NCCL groups).
TRANSFER_TIMEOUT_S = 120.0


def _agent_name(rank: int) -> str:
    return f"p2p-nixl-{rank}"


class P2pNixlAll2AllEngine(P2pAll2AllEngine):
    """``P2pAll2AllEngine`` whose dispatch/combine rows travel as NIXL WRITEs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from nixl import nixl_agent, nixl_agent_config  # noqa: PLC0415

        if self.device.type != "cuda":
            raise RuntimeError("p2p_nixl needs CUDA tensors (NIXL VRAM regions)")
        self.agent = nixl_agent(
            _agent_name(self.rank), nixl_agent_config(backends=[NIXL_BACKEND])
        )
        # (kind, tail, dtype) -> local region of world_size * R rows, and the
        # gather buffer of the same size that dispatch returns (fixed-size
        # launch wants one contiguous buffer; the region's shares are R apart).
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
        self._writes: list[Any] = []
        self._notifs: dict[str, list[bytes]] = {}
        self._register_agents()

    # ---------------------------------------------------------------- setup
    def _register_agents(self) -> None:
        """Exchange agent metadata (collective on the NCCL group)."""
        mine = (self.agent.get_agent_metadata(), self.device.index)
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        self._peer_devices = [info[1] for info in everyone]
        for peer, (metadata, _) in enumerate(everyone):
            if peer == self.rank:
                continue
            # add_remote_agent returns the name as bytes while notifications
            # are keyed by str, so peers are addressed by the known name.
            self.agent.add_remote_agent(metadata)
            self._peer_agents[peer] = _agent_name(peer)

    def _region(self, kind: str, tail: tuple[int, ...], dtype: torch.dtype) -> tuple:
        """Register (once, collectively) the region for one tensor shape.

        Every rank meets the same tensor shapes in the same order (same
        model, same layer), so this collective is matched across ranks.
        """
        key = (kind, tuple(tail), dtype)
        if key in self._regions:
            return key
        rows = self.world_size * self.max_tokens_per_rank
        region = torch.empty((rows, *tail), dtype=dtype, device=self.device)
        send = torch.empty_like(region)
        self.agent.register_memory([region, send])
        self._regions[key] = region
        self._sends[key] = send
        self._gathers[key] = torch.empty_like(region)
        row_bytes = region[0].numel() * region.element_size()
        mine = (region.data_ptr(), row_bytes)
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        self._peer_regions[key] = everyone
        return key

    # ------------------------------------------------------------- transfer
    def _write(self, peer: int, key: tuple, local: torch.Tensor, notif: bytes) -> None:
        """WRITE ``local`` (``[n, ...]``) into this rank's share of ``peer``'s
        region."""
        addr, row_bytes = self._peer_regions[key][peer]
        n = local.shape[0]
        remote = [
            (
                addr + self.rank * self.max_tokens_per_rank * row_bytes,
                n * row_bytes,
                self._peer_devices[peer],
            )
        ]
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs([local]),
            self.agent.get_xfer_descs(remote, "VRAM"),
            self._peer_agents[peer],
            notif,
        )
        if self.agent.transfer(handle) == "ERR":
            raise RuntimeError(f"p2p_nixl: WRITE to rank {peer} failed to post")
        self._writes.append(handle)

    def _finish_writes(self) -> None:
        """Poll every posted WRITE to completion (releases the send buffers)."""
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        for handle in self._writes:
            while (state := self.agent.check_xfer_state(handle)) == "PROC":
                if time.monotonic() > deadline:
                    raise RuntimeError("p2p_nixl: WRITE did not complete in time")
            if state != "DONE":
                raise RuntimeError(f"p2p_nixl: WRITE ended in state {state}")
            self.agent.release_xfer_handle(handle)
        self._writes.clear()

    def _wait_notifs(self, expected: dict[int, bytes]) -> None:
        """Wait for ``expected[peer]`` from every listed peer."""
        missing = {self._peer_agents[p]: n for p, n in expected.items()}
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        while missing:
            for name, notifs in self.agent.get_new_notifs().items():
                self._notifs.setdefault(name, []).extend(notifs)
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            if missing and time.monotonic() > deadline:
                raise RuntimeError(f"p2p_nixl: no notification from {sorted(missing)}")

    def _exchange(
        self,
        key: tuple,
        plan: P2pAll2AllPlan,
        packed: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        out_rows: int,
        notif: bytes,
    ) -> torch.Tensor:
        """Move ``packed`` (destination-major, ``send_counts``, a view of the
        registered send buffer of ``key``) to its owners and return this
        rank's received rows gathered source-major (``recv_counts``) in a
        buffer of ``out_rows`` rows."""
        region, gather = self._regions[key], self._gathers[key]
        R = self.max_tokens_per_rank
        # The sends read ``packed`` on the current stream: finish it first.
        torch.cuda.current_stream(self.device).synchronize()
        offset = 0
        for peer, count in enumerate(send_counts):
            if count:
                share = packed[offset : offset + count]
                if peer == self.rank:
                    region[self.rank * R : self.rank * R + count].copy_(share)
                else:
                    self._write(peer, key, share, notif)
            offset += count
        self._finish_writes()
        self._wait_notifs(
            {p: notif for p, c in enumerate(recv_counts) if c and p != self.rank}
        )
        # Source-major gather into the head of the fixed-size buffer; the
        # shares are at most R rows each so the head always fits. The row
        # count is the sum of what arrives here (num_recv for dispatch,
        # num_send for the partials coming back).
        num_in = sum(recv_counts)
        parts = [region[p * R : p * R + c] for p, c in enumerate(recv_counts)]
        if num_in:
            torch.cat(parts, out=gather[:num_in])
        return gather[:out_rows]

    def dispatch(
        self,
        plan: P2pAll2AllPlan,
        tensors: tuple[torch.Tensor, ...],
        min_rows: int = 0,
    ) -> list[torch.Tensor]:
        """Same contract as the NCCL engine's ``dispatch`` (see there); the
        rows travel as NIXL WRITEs. ``min_rows`` views come from the gather
        buffer, whose rows past ``num_recv`` are handled by ``mask_rows_past``
        exactly as for the NCCL receive buffers."""
        outputs: list[torch.Tensor] = []
        stream = _current_stream(self.device)
        num_send = plan.num_send
        if stream is not None and num_send > 0:
            plan.perm.record_stream(stream)
        notif = f"d{plan.seq}".encode()
        for i, x in enumerate(tensors):
            if x.shape[0] != plan.num_tokens:
                raise ValueError(
                    f"tensor {i} has {x.shape[0]} rows, plan has {plan.num_tokens}"
                )
            if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise RuntimeError("p2p_nixl carries unquantized rows only")
            key = self._region(f"recv{i}", tuple(x.shape[1:]), x.dtype)
            packed = self._sends[key][:num_send]
            if num_send > 0:
                torch.index_select(x, 0, plan.perm, out=packed)
            recv = self._exchange(
                key,
                plan,
                packed,
                plan.send_counts,
                plan.recv_counts,
                max(plan.num_recv, min_rows),
                notif,
            )
            key = (plan.slot, recv.data_ptr())
            if key in self._dirty_rows:
                self._dirty_rows[key] = max(self._dirty_rows[key], plan.num_recv)
            outputs.append(recv)
        return outputs

    def combine_send(self, plan: P2pAll2AllPlan, partial: torch.Tensor) -> torch.Tensor:
        """Return ``partial`` (``[num_recv, ...]``) to the owners as NIXL WRITEs;
        same contract as the NCCL engine's ``combine_send``."""
        if partial.shape[0] != plan.num_recv:
            raise ValueError(
                f"partial has {partial.shape[0]} rows, plan expects {plan.num_recv}"
            )
        # Stage into the registered send buffer. Going back: this rank sends
        # recv_counts (source-major rows it holds) and receives send_counts
        # (its own tokens' partials, perm order).
        key = self._region("back", tuple(partial.shape[1:]), partial.dtype)
        staged = self._sends[key][: plan.num_recv]
        staged.copy_(partial)
        return self._exchange(
            key,
            plan,
            staged,
            plan.recv_counts,
            plan.send_counts,
            plan.num_send,
            f"c{plan.seq}".encode(),
        )


__all__ = ["P2pNixlAll2AllEngine"]
