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

Activations are always dispatched unquantized; when the experts kernel wants
quantized inputs they are quantized after the dispatch, on the received rows
(same order as DeepEP high-throughput for non-block quantization). Since
every quantization scheme vLLM uses here is per-token, per-group or static,
the result is identical to quantizing before the dispatch.

Dual batch overlap: ``prepare_async``/``finalize_async`` follow the DeepEP
pattern. The count exchange is queued on the communication stream and the
thread yields to the other micro-batch *before* the host copy of the counts,
so the GPU has the other micro-batch's work to run while the counts travel.
Each micro-batch uses its own engine slot (buffers and counts).
"""

from __future__ import annotations

from collections.abc import Callable

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
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)


def _noop_hook() -> None:
    # NCCL transfers need no explicit completion call. Returning a hook
    # still makes the modular kernel yield once more under DBO, which lets
    # the other micro-batch compute while our transfer is in flight.
    return None


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
        # One in-flight plan per micro-batch slot.
        self._plans: dict[int, P2pAll2AllPlan] = {}

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

    def supports_async(self) -> bool:
        return True

    # --------------------------------------------------------------- prepare
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
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

        slot = dbo_current_ubatch_id()
        pending = self.engine.begin_plan(topk_ids, slot)
        # Under DBO: hand the CPU to the other micro-batch and move to the
        # communication stream. Without DBO these calls are no-ops.
        dbo_yield_and_switch_from_compute_to_comm()
        self.engine.exchange_counts(pending)
        plan = self.engine.finish_plan(pending)
        recv_a1, recv_ids, recv_weights = self.engine.dispatch(
            plan, (a1, topk_ids, topk_weights)
        )
        self._plans[slot] = plan
        dbo_switch_to_compute_sync()

        quantize_after = quant_config.quant_dtype is not None and not defer_input_quant
        a1_scale = (
            quant_config.a1_gscale
            if quant_config.quant_dtype == "nvfp4"
            else quant_config.a1_scale
        )

        def receiver() -> mk.PrepareResultType:
            # Slots of experts that live on other ranks: point them at an
            # expert this rank does not own so that expert_map yields -1, and
            # drop their routing weight so they contribute nothing locally.
            is_local = (
                torch.div(
                    recv_ids.to(torch.int64),
                    self.num_local_experts,
                    rounding_mode="floor",
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

            a1q: torch.Tensor = recv_a1
            a1q_scale: torch.Tensor | None = None
            if quantize_after and recv_a1.numel() != 0:
                a1q, a1q_scale = moe_kernel_quantize_input(
                    recv_a1,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=quant_config.per_act_token_quant,
                    block_shape=quant_config.block_shape,
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                    mx_alignment=getattr(quant_config, "mx_alignment", 0),
                )
            return a1q, a1q_scale, None, expert_topk_ids, expert_topk_weights

        return _noop_hook, receiver

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
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        hook()
        return receiver()

    # -------------------------------------------------------------- finalize
    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        slot = dbo_current_ubatch_id()
        plan = self._plans.pop(slot, None)
        assert plan is not None, "finalize called without a matching prepare"

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
        partial = partial.contiguous()

        dbo_yield_and_switch_from_compute_to_comm()
        back = self.engine.combine_send(plan, partial)
        dbo_switch_to_compute()

        def receiver() -> None:
            dbo_switch_to_comm()
            self.engine.combine_accumulate(plan, back, output)
            dbo_yield_and_switch_from_comm_to_compute()

        return _noop_hook, receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        hook, receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        hook()
        receiver()
