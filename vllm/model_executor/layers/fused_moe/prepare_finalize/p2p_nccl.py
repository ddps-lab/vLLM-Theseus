# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/Finalize for the ``p2p_nccl`` all2all backend.

Tokens are sent only to the expert-parallel ranks that own at least one of
their selected experts (see ``P2pAll2AllEngine``). After the dispatch the
local experts kernel runs unchanged: expert ids that do not belong to this
rank are rewritten to an id the ``expert_map`` maps to ``-1`` (the same trick
the DeepEP high-throughput path uses) and their routing weights are zeroed.
The combine returns the weighted partial sums to the owner rank, so the
finalize output is already reduced across the EP group.

The per-layer host sync for the token counts makes this path eager-only.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.p2p_nccl_all2all import (
    P2pAll2AllEngine,
    P2pAll2AllPlan,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)


class P2pNcclPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Selective point-to-point dispatch/combine over NCCL send/recv."""

    def __init__(
        self,
        engine: P2pAll2AllEngine,
        num_dispatchers: int,
        num_experts: int,
        num_local_experts: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.engine = engine
        self._num_dispatchers = num_dispatchers
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.rank = rank
        self.rank_expert_offset = rank * num_local_experts
        self._plan: P2pAll2AllPlan | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if quant_config.quant_dtype is not None and not defer_input_quant:
            raise NotImplementedError(
                "all2all_backend=p2p_nccl currently dispatches unquantized "
                f"activations only (requested {quant_config.quant_dtype})."
            )
        if num_experts != self.num_experts:
            raise ValueError(
                f"layer has {num_experts} experts, backend was built for "
                f"{self.num_experts}"
            )
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        plan = self.engine.plan(topk_ids)
        recv_a1, recv_ids, recv_weights = self.engine.dispatch(
            plan, (a1, topk_ids, topk_weights)
        )
        self._plan = plan

        # Slots of experts that live on other ranks: point them at an expert
        # this rank does not own so that expert_map yields -1, and drop their
        # routing weight so they contribute nothing to the local partial sum.
        is_local = (
            torch.div(
                recv_ids.to(torch.int64), self.num_local_experts, rounding_mode="floor"
            )
            == self.rank
        )
        foreign = self.num_experts - 1 if self.rank_expert_offset == 0 else 0
        expert_topk_ids = torch.where(
            is_local, recv_ids, torch.full_like(recv_ids, foreign)
        )
        expert_topk_weights = torch.where(
            is_local, recv_weights, torch.zeros_like(recv_weights)
        )
        return recv_a1, None, None, expert_topk_ids, expert_topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        plan = self._plan
        assert plan is not None, "finalize called without a matching prepare"
        self._plan = None

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        if fused_expert_output.numel() != 0:
            partial = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            partial = fused_expert_output.new_zeros((0, output.shape[-1]))

        self.engine.combine(plan, partial.contiguous(), output)
