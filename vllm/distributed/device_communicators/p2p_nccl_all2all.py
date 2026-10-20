# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Point-to-point (selective) MoE all-to-all built on ``torch.distributed``.

This module deliberately imports nothing from vLLM so that the communication
logic can be unit-tested with the gloo backend on CPU.

Per MoE layer the engine performs three steps:

* plan (``begin_plan`` -> ``exchange_counts`` -> ``finish_plan``): decide on
  device which ranks every local token has to visit (one visit per rank,
  however many of that rank's experts the token selected), exchange the
  per-destination counts with one tiny ``all_to_all_single`` and copy them to
  the host. That copy is the single CPU sync of the layer; NCCL needs
  host-side counts to post the receives. The three sub-steps are exposed
  separately so that a caller can move to another stream or yield to another
  micro-batch between them (dual batch overlap).
* ``dispatch``: pack the local tokens in destination order and ship them with
  a variable-split ``all_to_all_single`` into preallocated max-size buffers,
  so no per-layer allocation happens ("max buffer" variant).
* combine (``combine_send`` -> ``combine_accumulate``): send the per-token
  partial outputs back along the reversed splits and accumulate them (fp32)
  into the owner rank's output.

Buffers and count exchanges are kept per ``slot`` so that several
micro-batches can be in flight at the same time (one slot per micro-batch).

Only the GPUs that own at least one selected expert receive a token, so the
bytes on the wire are ``sum_t |ranks(t)| * row`` instead of the
``(world_size - 1) * T * row`` of the all-gather/reduce-scatter backend.
Expert ids below zero (padding) are not sent anywhere and produce zeros.

Usage (Theseus): ``--all2all-backend p2p_nccl --enable-expert-parallel`` with
``--data-parallel-size > 1``; the backend needs no DeepEP/NVSHMEM/RDMA and
runs over plain NCCL (TCP sockets are fine). Micro-batch overlap is
``--ubatch-size N`` (``--enable-dbo`` for N=2). CUDA graphs are disabled
automatically (the count exchange is a host sync). ``num_experts`` must be
divisible by the EP size, EPLB is not supported. ``VLLM_P2P_NCCL_DEBUG=1|2``
prints per-layer plan/dispatch logs (2 also validates received expert ids).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger("vllm.distributed.device_communicators.p2p_nccl_all2all")

# DEBUG(p2p_nccl): off by default. VLLM_P2P_NCCL_DEBUG=1 logs
# per-layer counts, bytes and host-wait time (first 8 calls, then every 64th);
# =2 additionally runs sanity checks that synchronize with the device.
_DEBUG = int(os.environ.get("VLLM_P2P_NCCL_DEBUG", "0") or 0)


def _should_log(n: int) -> bool:
    return n < 8 or n % 64 == 0


# NCCL/gloo have no 8-bit float type; such tensors travel as uint8 views.
_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)


def _as_wire(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in _FP8_DTYPES:
        return x.contiguous().view(torch.uint8)
    return x


def _current_stream(device: torch.device) -> torch.cuda.Stream | None:
    if device.type != "cuda":
        return None
    return torch.cuda.current_stream(device)


@dataclass
class P2pAll2AllPlan:
    """Host-side description of one dispatch/combine round."""

    slot: int
    num_tokens: int
    send_counts: list[int]
    recv_counts: list[int]
    num_send: int
    num_recv: int
    # Local token index of every packed row, in destination-major order.
    perm: torch.Tensor
    # Sequence number of this plan on this rank (diagnostics).
    seq: int = 0
    # Host wait for the count copy, in ms (diagnostics).
    host_wait_ms: float = 0.0


@dataclass
class P2pAll2AllPendingPlan:
    """Device-side layout computed by ``begin_plan``; completed by ``finish_plan``."""

    slot: int
    num_tokens: int
    # argsort of the destination-major keys, length world_size * num_tokens.
    order: torch.Tensor


class P2pAll2AllEngine:
    """Selective token exchange between the ranks of one expert-parallel group.

    Experts are assumed to be assigned contiguously: rank ``r`` owns global
    experts ``[r * num_local_experts, (r + 1) * num_local_experts)``. Ranks
    beyond ``num_experts / num_local_experts`` own no experts and only send
    (e.g. attention ranks of a one-hop AFD deployment); ranks that own
    experts may have no tokens of their own and only receive.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        num_local_experts: int,
        max_tokens_per_rank: int,
        device: torch.device | str,
        combine_group: dist.ProcessGroup | None = None,
    ) -> None:
        if num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        self.group = group
        # Collectives on one communicator must be issued in the same order by
        # every rank. When senders and receivers interleave dispatch and
        # combine differently (e.g. attention ranks issue all dispatches of a
        # layer before the first combine while FFN ranks alternate), the
        # combine must live on its own communicator.
        self.combine_group = combine_group if combine_group is not None else group
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.max_tokens_per_rank = max_tokens_per_rank
        self.device = torch.device(device)
        # Initial rows per buffer. The worst case is max_tokens_per_rank *
        # world_size (every token of every rank lands here), but that is far
        # above the typical traffic, so buffers start at one rank's worth of
        # tokens and grow geometrically on demand (rare after warm-up).
        self.capacity = max_tokens_per_rank
        self._arange_r = torch.arange(world_size, device=self.device)
        self._count_bufs: dict[int, torch.Tensor] = {}
        self._buffers: dict[tuple, torch.Tensor] = {}
        # Ranks without tokens of their own (the FFN role of a one-hop AFD
        # deployment) take part in every exchange but skip the plan
        # arithmetic and the send-side packing: an empty ordering, zero-row
        # send buffers per wire shape, and the slots whose send counts are
        # already zero on device.
        self._empty_order = torch.empty(0, dtype=torch.int64, device=self.device)
        self._empty_sends: dict[tuple, torch.Tensor] = {}
        self._zero_send_slots: set[int] = set()
        self._num_plans = 0
        # mask_rows_past: per (slot, buffer address), how many leading rows
        # may still hold expert ids from an earlier dispatch.
        self._dirty_rows: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ plan
    def _count_buf(self, slot: int) -> torch.Tensor:
        buf = self._count_bufs.get(slot)
        if buf is None:
            buf = torch.zeros(
                2, self.world_size, dtype=torch.int64, device=self.device
            )
            self._count_bufs[slot] = buf
        return buf

    def begin_plan(self, topk_ids: torch.Tensor, slot: int = 0) -> P2pAll2AllPendingPlan:
        """Device-only part of the plan: destinations, send counts, ordering."""
        num_tokens = topk_ids.shape[0]
        world = self.world_size
        device = self.device

        if num_tokens == 0:
            # Nothing to send: zero the send counts once per slot (they stay
            # zero until a call with tokens overwrites them) and skip the
            # destination/ordering arithmetic, which would run ~20 kernels
            # on empty tensors every layer on a receive-only rank.
            if slot not in self._zero_send_slots:
                self._count_buf(slot)[0].zero_()
                self._zero_send_slots.add(slot)
            return P2pAll2AllPendingPlan(
                slot=slot, num_tokens=0, order=self._empty_order
            )
        self._zero_send_slots.discard(slot)

        rank_of = torch.div(
            topk_ids.to(torch.int64), self.num_local_experts, rounding_mode="floor"
        )
        # dest[t, r]: token t has to visit rank r (at most once per rank).
        # Negative ids (padding) match no rank.
        dest = (rank_of.unsqueeze(-1) == self._arange_r).any(dim=1)
        self._count_buf(slot)[0].copy_(dest.sum(dim=0, dtype=torch.int64))

        # Destination-major ordering with a fixed-length sort: the key of
        # pair (r, t) is r * T + t when the token goes to r, otherwise a
        # sentinel that sorts last. No data-dependent shapes on device.
        key_base = self._arange_r.unsqueeze(1) * num_tokens + torch.arange(
            num_tokens, device=device
        ).unsqueeze(0)
        key = torch.where(
            dest.t(), key_base, torch.full_like(key_base, world * num_tokens)
        )
        order = key.reshape(-1).argsort()
        return P2pAll2AllPendingPlan(slot=slot, num_tokens=num_tokens, order=order)

    def exchange_counts(self, pending: P2pAll2AllPendingPlan) -> None:
        """Exchange per-destination counts (collective, on the current stream)."""
        buf = self._count_buf(pending.slot)
        dist.all_to_all_single(buf[1], buf[0], group=self.group)

    def finish_plan(self, pending: P2pAll2AllPendingPlan) -> P2pAll2AllPlan:
        """Copy the counts to the host (the only CPU sync) and build the plan."""
        # The only host sync of the layer.
        t0 = time.perf_counter()
        send_counts, recv_counts = self._count_buf(pending.slot).tolist()
        host_wait_ms = (time.perf_counter() - t0) * 1e3
        num_send = sum(send_counts)
        num_tokens = pending.num_tokens
        stream = _current_stream(self.device)
        if stream is not None and num_tokens > 0:
            # ``order`` was produced on the stream that ran begin_plan (the
            # compute stream under dual batch overlap) and is read here on the
            # current one; keep its memory from being reused before that read.
            pending.order.record_stream(stream)
        perm = (
            pending.order[:num_send] % num_tokens
            if num_tokens > 0
            else self._empty_order
        )
        seq = self._num_plans
        self._num_plans += 1
        plan = P2pAll2AllPlan(
            slot=pending.slot,
            num_tokens=num_tokens,
            send_counts=send_counts,
            recv_counts=recv_counts,
            num_send=num_send,
            num_recv=sum(recv_counts),
            perm=perm,
            seq=seq,
            host_wait_ms=host_wait_ms,
        )
        # DEBUG(p2p_nccl): only with VLLM_P2P_NCCL_DEBUG.
        if _DEBUG and _should_log(seq):
            logger.info(
                "p2p_nccl[r%d] plan#%d slot=%d T=%d send=%s recv=%s "
                "n_send=%d n_recv=%d host_wait=%.3fms",
                self.rank, seq, pending.slot, num_tokens, send_counts,
                recv_counts, num_send, plan.num_recv, host_wait_ms,
            )
            if num_send == 0 and plan.num_recv == 0:
                logger.info(
                    "p2p_nccl[r%d] plan#%d: nothing to send or receive "
                    "(empty all_to_all on this rank)", self.rank, seq,
                )
        return plan

    def plan(self, topk_ids: torch.Tensor, slot: int = 0) -> P2pAll2AllPlan:
        """``begin_plan`` + ``exchange_counts`` + ``finish_plan`` in one go."""
        pending = self.begin_plan(topk_ids, slot)
        self.exchange_counts(pending)
        return self.finish_plan(pending)

    # --------------------------------------------------------------- buffers
    def _buffer(
        self,
        slot: int,
        name: str,
        rows: int,
        tail: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (slot, name, tuple(tail), dtype)
        buf = self._buffers.get(key)
        if buf is None:
            # Buffers first used with zero rows (receive buffers of ranks that
            # own no experts, send buffers of ranks without tokens) stay
            # empty; the others start at the configured capacity.
            initial = max(self.capacity, rows) if rows > 0 else 0
            buf = torch.empty((initial, *tail), dtype=dtype, device=self.device)
            self._buffers[key] = buf
        elif buf.shape[0] < rows:
            new_rows = max(rows, 2 * buf.shape[0], self.capacity)
            logger.info(
                "p2p_nccl[r%d] growing buffer %s (slot %d) from %d to %d rows",
                self.rank, name, slot, buf.shape[0], new_rows,
            )
            buf = torch.empty((new_rows, *tail), dtype=dtype, device=self.device)
            self._buffers[key] = buf
        return buf[:rows]

    def _empty_send(self, tail: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(tail), dtype)
        buf = self._empty_sends.get(key)
        if buf is None:
            buf = torch.empty((0, *tail), dtype=dtype, device=self.device)
            self._empty_sends[key] = buf
        return buf

    def mask_rows_past(self, slot: int, ids: torch.Tensor, num_valid: int) -> None:
        """Set the rows ``num_valid:`` of ``ids`` to -1.

        ``ids`` is a fixed-size view (``dispatch(min_rows=...)``) of a receive
        buffer whose first ``num_valid`` rows the last dispatch on ``slot``
        wrote. Only the rows that may still hold ids from an earlier dispatch
        are rewritten (nothing when this dispatch wrote at least as many), and
        a buffer seen for the first time is filled once. Expert id -1 makes
        the experts kernel skip the row.
        """
        key = (slot, ids.data_ptr())
        dirty = self._dirty_rows.get(key, ids.shape[0])
        if dirty > num_valid:
            ids[num_valid:dirty].fill_(-1)
        self._dirty_rows[key] = num_valid

    # -------------------------------------------------------------- dispatch
    def dispatch(
        self,
        plan: P2pAll2AllPlan,
        tensors: tuple[torch.Tensor, ...],
        min_rows: int = 0,
    ) -> list[torch.Tensor]:
        """Send row ``t`` of every tensor to each rank token ``t`` visits.

        Returns, for every input ``[T, ...]``, a view of a preallocated buffer
        whose first ``num_recv`` rows hold the rows received by this rank,
        grouped by source rank. The view has ``max(num_recv, min_rows)`` rows;
        rows past ``num_recv`` keep whatever the buffer held before (callers
        that pass ``min_rows`` launch the experts kernel at a fixed size and
        mask those rows themselves). The views stay valid until the next
        ``dispatch`` on the same slot.
        """
        outputs: list[torch.Tensor] = []
        stream = _current_stream(self.device)
        num_send = plan.num_send
        if stream is not None and num_send > 0:
            # The inputs may have been produced on another stream (dual batch
            # overlap runs this on the communication stream while the tensors
            # were allocated on the compute stream). Keep their memory from
            # being reused until this stream is done reading them.
            plan.perm.record_stream(stream)
        for i, x in enumerate(tensors):
            if x.shape[0] != plan.num_tokens:
                raise ValueError(
                    f"tensor {i} has {x.shape[0]} rows, plan has {plan.num_tokens}"
                )
            if num_send > 0:
                xw = _as_wire(x)
                if stream is not None:
                    xw.record_stream(stream)
                wire_dtype = xw.dtype
                tail = tuple(xw.shape[1:])
                packed = self._buffer(plan.slot, f"send{i}", num_send, tail, wire_dtype)
                torch.index_select(xw, 0, plan.perm, out=packed)
            else:
                # Receive-only rank: no packing, an empty send of the wire dtype.
                wire_dtype = torch.uint8 if x.dtype in _FP8_DTYPES else x.dtype
                tail = tuple(x.shape[1:])
                packed = self._empty_send(tail, wire_dtype)
            recv = self._buffer(
                plan.slot, f"recv{i}", max(plan.num_recv, min_rows), tail, wire_dtype
            )
            dist.all_to_all_single(
                recv[: plan.num_recv],
                packed,
                output_split_sizes=plan.recv_counts,
                input_split_sizes=plan.send_counts,
                group=self.group,
            )
            # Keep mask_rows_past's record right when a buffer it tracks is
            # also used without a fixed size.
            key = (plan.slot, recv.data_ptr())
            if key in self._dirty_rows:
                self._dirty_rows[key] = max(self._dirty_rows[key], plan.num_recv)
            outputs.append(recv.view(x.dtype) if wire_dtype != x.dtype else recv)
        # DEBUG(p2p_nccl): only with VLLM_P2P_NCCL_DEBUG.
        if _DEBUG and _should_log(plan.seq):
            row_bytes = sum(
                math.prod(x.shape[1:]) * x.element_size() for x in tensors
            )
            remote_out = plan.num_send - plan.send_counts[self.rank]
            remote_in = plan.num_recv - plan.recv_counts[self.rank]
            logger.info(
                "p2p_nccl[r%d] dispatch#%d slot=%d rows_out=%d (remote %d) "
                "rows_in=%d (remote %d) bytes_out=%d bytes_in=%d",
                self.rank, plan.seq, plan.slot, plan.num_send, remote_out,
                plan.num_recv, remote_in, remote_out * row_bytes,
                remote_in * row_bytes,
            )
        return outputs

    # --------------------------------------------------------------- combine
    def combine_send(
        self, plan: P2pAll2AllPlan, partial: torch.Tensor
    ) -> torch.Tensor:
        """Return ``partial`` (``[num_recv, ...]``) to the owners.

        Returns the ``[num_send, ...]`` buffer view that receives this rank's
        partial sums back, in the same order as ``plan.perm``.
        """
        if partial.shape[0] != plan.num_recv:
            raise ValueError(
                f"partial has {partial.shape[0]} rows, plan expects {plan.num_recv}"
            )
        tail = tuple(partial.shape[1:])
        back = self._buffer(plan.slot, "back", plan.num_send, tail, partial.dtype)
        if plan.num_recv > 0:
            partial = partial.contiguous()
            stream = _current_stream(self.device)
            if stream is not None:
                # Same cross-stream hazard as in dispatch(): ``partial`` was
                # computed on the compute stream.
                partial.record_stream(stream)
        dist.all_to_all_single(
            back,
            partial,
            output_split_sizes=plan.send_counts,
            input_split_sizes=plan.recv_counts,
            group=self.combine_group,
        )
        return back

    def combine_accumulate(
        self, plan: P2pAll2AllPlan, back: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Sum the returned partials into ``out`` (``[T, ...]``, overwritten).

        Accumulation is done in fp32 and cast to ``out.dtype`` once.
        """
        if out.shape[0] != plan.num_tokens:
            raise ValueError(
                f"out has {out.shape[0]} rows, plan has {plan.num_tokens}"
            )
        acc = self._buffer(
            plan.slot, "acc", plan.num_tokens, tuple(back.shape[1:]), torch.float32
        )
        acc.zero_()
        acc.index_add_(0, plan.perm, back.to(torch.float32))
        out.copy_(acc.to(out.dtype))
        return out

    def combine(
        self, plan: P2pAll2AllPlan, partial: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """``combine_send`` + ``combine_accumulate`` in one go."""
        back = self.combine_send(plan, partial)
        return self.combine_accumulate(plan, back, out)
