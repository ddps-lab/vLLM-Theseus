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

When the experts kernel wants quantized inputs, activations are quantized
before the dispatch and the quantized values travel on the wire (8-bit
tensors are sent as uint8 views). Per-token or per-group scales are sent
along as an extra tensor; static scales are replicated on every rank and are
not sent. The one exception is a dynamic per-tensor scale (fp8/int8 without
static input scale, block shape or per-token quantization): it depends on
the sender's batch, so those activations travel unquantized and are
quantized after the dispatch, as the DeepEP backend does.

Dual batch overlap: ``prepare_async``/``finalize_async`` follow the DeepEP
pattern. The count exchange is queued on the communication stream and the
thread yields to the other micro-batch *before* the host copy of the counts,
so the GPU has the other micro-batch's work to run while the counts travel.
Each micro-batch uses its own engine slot (buffers and counts).
"""

from __future__ import annotations

import os
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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)


logger = init_logger(__name__)

# DEBUG(p2p_nccl): remove after GPU validation (see p2p_nccl_all2all.py).
_DEBUG = int(os.environ.get("VLLM_P2P_NCCL_DEBUG", "0") or 0)


def quantize_before_dispatch(quant_config: FusedMoEQuantConfig) -> bool:
    """Whether the activations can be quantized on the sending rank.

    Scales that are static (replicated on every rank) or that have one row
    per token (per-token, per-block, nvfp4/mxfp4 groups) survive the
    dispatch. A dynamic per-tensor scale (fp8/int8 without a static input
    scale, block shape or per-token quantization) depends on the sender's
    whole batch and is not sent, so those activations travel unquantized and
    are quantized on the receiving rank, as the DeepEP backend does.
    """
    quant_dtype = quant_config.quant_dtype
    if quant_dtype is None:
        return False
    if isinstance(quant_dtype, str):
        # nvfp4 / mxfp4: per-group scales with a static global scale.
        return True
    if quant_config.per_act_token_quant or quant_config.block_shape is not None:
        return True
    return quant_config.a1_scale is not None


def _quantize_for_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Quantize ``a1`` before the dispatch.

    Returns ``(a1q, scale_to_send, scale)``. ``scale_to_send`` is the scale
    tensor when it has one row per token (it travels with the tokens),
    otherwise ``None`` (static scale, replicated on every rank). When the
    quantization cannot happen on the sender (see
    ``quantize_before_dispatch``) ``a1`` is returned unchanged.
    """
    if defer_input_quant or not quantize_before_dispatch(quant_config):
        return a1, None, None
    input_sf = (
        quant_config.a1_gscale if quant_config.use_nvfp4_w4a4 else quant_config.a1_scale
    )
    # Scale swizzling pads to multiples of 128 rows, which would break the
    # per-token split; swizzle after the dispatch instead.
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=False,
        mx_alignment=getattr(quant_config, "mx_alignment", 0),
    )
    per_row = (
        a1q_scale is not None
        and a1q_scale.ndim > 0
        and a1q_scale.shape[0] == a1.shape[0]
    )
    return a1q, (a1q_scale if per_row else None), a1q_scale


def _quantize_after_dispatch(
    recv_a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize received rows whose scale could not travel (see above)."""
    if recv_a1.numel() == 0:
        return recv_a1, None
    return moe_kernel_quantize_input(
        recv_a1,
        quant_config.a1_scale,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=quant_config.is_scale_swizzled,
        mx_alignment=getattr(quant_config, "mx_alignment", 0),
    )


def _scale_after_dispatch(
    recv_scale: torch.Tensor | None,
    scale: torch.Tensor | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor | None:
    if recv_scale is None:
        return scale
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        if recv_scale.element_size() == 1:
            recv_scale = recv_scale.view(torch.uint8)
        recv_scale = nvfp4_block_scale_interleave(recv_scale)
    return recv_scale


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
        self._num_prepares = 0  # DEBUG(p2p_nccl)

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

        a1q, scale_to_send, a1q_scale = _quantize_for_dispatch(
            a1, quant_config, defer_input_quant
        )
        quantize_on_receive = (
            a1q is a1
            and not defer_input_quant
            and quant_config.quant_dtype is not None
        )
        tensors: list[torch.Tensor] = [a1q, topk_ids, topk_weights]
        if scale_to_send is not None:
            tensors.append(scale_to_send)

        slot = dbo_current_ubatch_id()
        # DEBUG(p2p_nccl): remove after GPU validation.
        if _DEBUG and self._num_prepares < 8:
            logger.info(
                "p2p_nccl P/F rank=%d prepare#%d dbo=%s slot=%d T=%d topk=%d "
                "ids_dtype=%s quant_dtype=%s defer_input_quant=%s wire_dtype=%s "
                "quantize_on_receive=%s",
                self.rank, self._num_prepares, dbo_enabled(), slot,
                a1.shape[0], topk_ids.shape[1], topk_ids.dtype,
                quant_config.quant_dtype, defer_input_quant, a1q.dtype,
                quantize_on_receive,
            )
        self._num_prepares += 1
        pending = self.engine.begin_plan(topk_ids, slot)
        # Under DBO: hand the CPU to the other micro-batch and move to the
        # communication stream. Without DBO these calls are no-ops.
        dbo_yield_and_switch_from_compute_to_comm()
        self.engine.exchange_counts(pending)
        plan = self.engine.finish_plan(pending)
        received = self.engine.dispatch(plan, tuple(tensors))
        recv_a1, recv_ids, recv_weights = received[0], received[1], received[2]
        recv_scale = received[3] if scale_to_send is not None else None
        self._plans[slot] = plan
        dbo_switch_to_compute_sync()

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
            # DEBUG(p2p_nccl): remove after GPU validation. Level 2 checks
            # synchronize with the device.
            if _DEBUG >= 2 and recv_ids.numel() > 0:
                lo, hi = int(recv_ids.min()), int(recv_ids.max())
                assert 0 <= lo and hi < self.num_experts, (
                    f"p2p_nccl: received expert ids out of range [{lo}, {hi}]"
                )
                assert bool(is_local.any(dim=1).all()), (
                    "p2p_nccl: received a token that selects none of this "
                    "rank's experts"
                )
            foreign = self.num_experts - 1 if self.rank_expert_offset == 0 else 0
            expert_topk_ids = torch.where(
                is_local, recv_ids, torch.full_like(recv_ids, foreign)
            )
            expert_topk_weights = torch.where(
                is_local, recv_weights, torch.zeros_like(recv_weights)
            )

            if quantize_on_receive:
                # Dynamic per-tensor scale: quantize the received rows here.
                recv_a1q, recv_a1q_scale = _quantize_after_dispatch(
                    recv_a1, quant_config
                )
                return (
                    recv_a1q,
                    recv_a1q_scale,
                    None,
                    expert_topk_ids,
                    expert_topk_weights,
                )
            recv_a1q_scale = _scale_after_dispatch(recv_scale, a1q_scale, quant_config)
            return recv_a1, recv_a1q_scale, None, expert_topk_ids, expert_topk_weights

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
        # DEBUG(p2p_nccl): remove after GPU validation.
        if _DEBUG and plan.seq < 8:
            logger.info(
                "p2p_nccl P/F rank=%d finalize plan#%d slot=%d expert_out=%s "
                "output=%s",
                self.rank, plan.seq, slot, tuple(fused_expert_output.shape),
                tuple(output.shape),
            )

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
