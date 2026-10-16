# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/Finalize for the ``prerouted`` all2all backend.

Used by expert-parallel ranks whose input tokens were already delivered to
the rank that owns (some of) their selected experts by an external
transport, e.g. the AFD FFN role behind a one-hop attention-to-expert
connector. No communication happens here: the received global expert ids
and routing weights go to the local experts kernel unchanged. Slots of
experts that live on other ranks are skipped by the kernel itself through
``expert_map`` (which maps them to ``-1``), exactly as in the
allgather/reduce-scatter path, so no masking pass is needed before the
kernel. The finalize output is the per-token partial sum over the local
experts; it is reported as reduced because the external transport sums the
partials of the different ranks on the token owner.

When the transport delivers activations that were already quantized on the
sender, the caller stores them in ``prequantized`` before the forward (see
``set_prequantized``) and passes a placeholder of the unquantized dtype as
the kernel input; the placeholder only determines the output dtype and the
token count.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)


class PreroutedPrepareAndFinalize(MoEPrepareAndFinalizeNoDPEPModular):
    """No-communication prepare/finalize for pre-routed tokens."""

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        rank: int,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.rank = rank
        self.rank_expert_offset = rank * num_local_experts
        self._num_dispatchers = num_dispatchers
        # (a1q, a1q_scale) delivered already quantized by the external
        # transport. Consumed (and cleared) by the next ``prepare``.
        self.prequantized: tuple[torch.Tensor, torch.Tensor | None] | None = None

    def set_prequantized(
        self, a1q: torch.Tensor, a1q_scale: torch.Tensor | None
    ) -> None:
        self.prequantized = (a1q, a1q_scale)

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
        if num_experts != self.num_experts:
            raise ValueError(
                f"layer has {num_experts} experts, backend was built for "
                f"{self.num_experts}"
            )
        if self.prequantized is not None:
            # The transport delivered quantized activations; do not re-quantize.
            a1q, a1q_scale = self.prequantized
            self.prequantized = None
            if a1q.shape[0] != a1.shape[0]:
                raise ValueError(
                    f"prequantized input has {a1q.shape[0]} rows, "
                    f"placeholder has {a1.shape[0]}"
                )
            assert not apply_router_weight_on_input, (
                "prerouted: apply_router_weight_on_input with pre-quantized input"
            )
        else:
            a1q, a1q_scale, _, _, _ = super().prepare(
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )
        # The ids are global and the experts kernel skips the slots whose
        # expert_map entry is -1 (align/sum kernels and the Triton kernel all
        # do), so the received ids and weights are passed through as they are.
        return a1q, a1q_scale, None, topk_ids, topk_weights
