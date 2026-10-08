# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Point-to-point (selective) MoE all-to-all built on ``torch.distributed``.

This module deliberately imports nothing from vLLM so that the communication
logic can be unit-tested with the gloo backend on CPU.

Per MoE layer the engine performs three steps:

* ``plan``: decide on device which ranks every local token has to visit
  (one visit per rank, however many of that rank's experts the token
  selected), exchange the per-destination counts with one tiny
  ``all_to_all_single`` and copy them to the host. That copy is the single
  CPU sync of the layer; NCCL needs host-side counts to post the receives.
* ``dispatch``: pack the local tokens in destination order and ship them with
  a variable-split ``all_to_all_single`` into preallocated max-size buffers,
  so no per-layer allocation happens ("max buffer" variant).
* ``combine``: send the per-token partial outputs back along the reversed
  splits and accumulate them (fp32) into the owner rank's output.

Only the GPUs that own at least one selected expert receive a token, so the
bytes on the wire are ``sum_t |ranks(t)| * row`` instead of the
``(world_size - 1) * T * row`` of the all-gather/reduce-scatter backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class P2pAll2AllPlan:
    """Host-side description of one dispatch/combine round."""

    num_tokens: int
    send_counts: list[int]
    recv_counts: list[int]
    num_send: int
    num_recv: int
    # Local token index of every packed row, in destination-major order.
    perm: torch.Tensor


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
        self._count_buf = torch.zeros(
            2, world_size, dtype=torch.int64, device=self.device
        )
        self._buffers: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ plan
    def plan(self, topk_ids: torch.Tensor) -> P2pAll2AllPlan:
        """Compute the exchange layout for ``topk_ids`` (``[T, topk]``)."""
        num_tokens = topk_ids.shape[0]
        world = self.world_size
        device = self.device

        rank_of = torch.div(
            topk_ids.to(torch.int64), self.num_local_experts, rounding_mode="floor"
        )
        # dest[t, r]: token t has to visit rank r (at most once per rank).
        dest = (rank_of.unsqueeze(-1) == self._arange_r).any(dim=1)
        self._count_buf[0].copy_(dest.sum(dim=0, dtype=torch.int64))
        dist.all_to_all_single(
            self._count_buf[1], self._count_buf[0], group=self.group
        )
        # The only host sync of the layer.
        send_counts, recv_counts = self._count_buf.tolist()

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
        num_send = sum(send_counts)
        perm = order[:num_send] % num_tokens if num_tokens > 0 else order[:0]

        return P2pAll2AllPlan(
            num_tokens=num_tokens,
            send_counts=send_counts,
            recv_counts=recv_counts,
            num_send=num_send,
            num_recv=sum(recv_counts),
            perm=perm,
        )

    # --------------------------------------------------------------- buffers
    def _buffer(
        self, name: str, rows: int, tail: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        key = (name, tuple(tail), dtype)
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
        by source rank. The views stay valid until the next ``dispatch``.
        """
        outputs: list[torch.Tensor] = []
        for i, x in enumerate(tensors):
            if x.shape[0] != plan.num_tokens:
                raise ValueError(
                    f"tensor {i} has {x.shape[0]} rows, plan has {plan.num_tokens}"
                )
            tail = tuple(x.shape[1:])
            packed = self._buffer(f"send{i}", plan.num_send, tail, x.dtype)
            torch.index_select(x, 0, plan.perm, out=packed)
            recv = self._buffer(f"recv{i}", plan.num_recv, tail, x.dtype)
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
    def combine(
        self, plan: P2pAll2AllPlan, partial: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Return ``partial`` (``[num_recv, ...]``) to the owners, summing into ``out``.

        ``out`` must be ``[T, ...]``; it is overwritten. Accumulation is done
        in fp32 and cast to ``out.dtype`` once.
        """
        if partial.shape[0] != plan.num_recv:
            raise ValueError(
                f"partial has {partial.shape[0]} rows, plan expects {plan.num_recv}"
            )
        if out.shape[0] != plan.num_tokens:
            raise ValueError(
                f"out has {out.shape[0]} rows, plan has {plan.num_tokens}"
            )
        tail = tuple(partial.shape[1:])
        back = self._buffer("back", plan.num_send, tail, partial.dtype)
        dist.all_to_all_single(
            back,
            partial.contiguous(),
            output_split_sizes=plan.send_counts,
            input_split_sizes=plan.recv_counts,
            group=self.group,
        )
        acc = torch.zeros(
            (plan.num_tokens, *tail), dtype=torch.float32, device=partial.device
        )
        acc.index_add_(0, plan.perm, back.to(torch.float32))
        out.copy_(acc.to(out.dtype))
        return out
