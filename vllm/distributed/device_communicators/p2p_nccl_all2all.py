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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


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
    experts ``[r * num_local_experts, (r + 1) * num_local_experts)``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        num_local_experts: int,
        max_tokens_per_rank: int,
        device: torch.device | str,
    ) -> None:
        if num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.max_tokens_per_rank = max_tokens_per_rank
        self.device = torch.device(device)
        # Worst case: every token of every rank lands on this rank.
        self.capacity = max_tokens_per_rank * world_size
        self._arange_r = torch.arange(world_size, device=self.device)
        self._count_bufs: dict[int, torch.Tensor] = {}
        self._buffers: dict[tuple, torch.Tensor] = {}

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
        send_counts, recv_counts = self._count_buf(pending.slot).tolist()
        num_send = sum(send_counts)
        num_tokens = pending.num_tokens
        perm = (
            pending.order[:num_send] % num_tokens
            if num_tokens > 0
            else pending.order[:0]
        )
        return P2pAll2AllPlan(
            slot=pending.slot,
            num_tokens=num_tokens,
            send_counts=send_counts,
            recv_counts=recv_counts,
            num_send=num_send,
            num_recv=sum(recv_counts),
            perm=perm,
        )

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
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty(
                (max(self.capacity, rows), *tail), dtype=dtype, device=self.device
            )
            self._buffers[key] = buf
        return buf[:rows]

    # -------------------------------------------------------------- dispatch
    def dispatch(
        self, plan: P2pAll2AllPlan, tensors: tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor]:
        """Send row ``t`` of every tensor to each rank token ``t`` visits.

        Returns, for every input ``[T, ...]``, a view ``[num_recv, ...]`` of a
        preallocated buffer holding the rows received by this rank, grouped
        by source rank. The views stay valid until the next ``dispatch`` on
        the same slot.
        """
        outputs: list[torch.Tensor] = []
        for i, x in enumerate(tensors):
            if x.shape[0] != plan.num_tokens:
                raise ValueError(
                    f"tensor {i} has {x.shape[0]} rows, plan has {plan.num_tokens}"
                )
            tail = tuple(x.shape[1:])
            packed = self._buffer(plan.slot, f"send{i}", plan.num_send, tail, x.dtype)
            torch.index_select(x, 0, plan.perm, out=packed)
            recv = self._buffer(plan.slot, f"recv{i}", plan.num_recv, tail, x.dtype)
            dist.all_to_all_single(
                recv,
                packed,
                output_split_sizes=plan.recv_counts,
                input_split_sizes=plan.send_counts,
                group=self.group,
            )
            outputs.append(recv)
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
        dist.all_to_all_single(
            back,
            partial.contiguous(),
            output_split_sizes=plan.send_counts,
            input_split_sizes=plan.recv_counts,
            group=self.group,
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
        acc = torch.zeros(
            (plan.num_tokens, *back.shape[1:]), dtype=torch.float32, device=back.device
        )
        acc.index_add_(0, plan.perm, back.to(torch.float32))
        out.copy_(acc.to(out.dtype))
        return out

    def combine(
        self, plan: P2pAll2AllPlan, partial: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """``combine_send`` + ``combine_accumulate`` in one go."""
        back = self.combine_send(plan, partial)
        return self.combine_accumulate(plan, back, out)
