# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[theseus p2p_delta] Prepare/Finalize of the ``p2p_nixl`` backend with
delta dispatch (``vllm/distributed/device_communicators/p2p_delta.py``).

Every EP rank is both an "Attention" rank, which sends its tokens, and an
"FFN" rank, which runs its experts for every rank's tokens. Per (token,
destination rank) row it sends, instead of the MoE input ``x``, the bytes of
the row's format: ``h'`` (the residual after attention) or its difference
from the state both sides keep for that destination, or ``x`` quantized.
The receiving rank rebuilds ``h'`` and normalizes it with the layer's
``post_attention_layernorm`` (``norm_from_residual``: the fused add + norm
kernel the sender ran).

A rank's own rows are not encoded: their exact ``h'`` (``x`` in the x
modes), expert ids and weights go on the GPU after the received rows, and
their partial sums go into their dispatch place of the combine; the states
cover the other ranks' rows only. The partial sums are summed in the
dispatch order as in ``P2pNcclPrepareAndFinalize``, so ``full_bf16`` and
``delta_fp32`` give the plain path's expert inputs and outputs bit for bit.
With combine compression (``"combine": {...}``) the peers' partial sums come
back compressed (with error feedback, kept by the rank that computed them)
and are rebuilt on the sending rank; the own rows' partial sums stay exact.

The model registers each MoE layer's norm and records ``h'`` per
micro-batch (``P2pDeltaLayers``, ``deepseek_v2.py``); the MoE runner binds
each prepare/finalize to its layer and routed scale. Settings:
``--additional-config '{"p2p_delta": {"delta_dispatch": {...}}}'``, checked
by ``p2p_delta_configs``.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.p2p_delta import (
    MAX_ATTENTION_RANKS,
    AttentionDeltaState,
    CombineConfig,
    DeltaDispatchConfig,
    FFNDeltaState,
    PackedDispatch,
)
from vllm.distributed.device_communicators.p2p_nccl_all2all import P2pAll2AllPlan
from vllm.distributed.device_communicators.p2p_nixl_all2all import (
    P2pNixlAll2AllEngine,
    _profile_range,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fixed_launch
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.p2p_nccl import (
    P2pNcclPrepareAndFinalize,
    _noop_hook,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_switch_to_comm,
    dbo_switch_to_comm_sync,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_compute_to_comm,
)

logger = init_logger(__name__)

# Dual batch overlap: two micro-batches at most.
MAX_UBATCHES = 2


def p2p_delta_configs(vllm_config: Any) -> tuple[DeltaDispatchConfig, CombineConfig]:
    """The delta dispatch and combine settings of ``vllm_config``; with delta
    dispatch on, refuse what the aggregated path does not support."""
    cfg = DeltaDispatchConfig.from_vllm_config(vllm_config)
    combine = CombineConfig.from_vllm_config(vllm_config)
    if combine.enabled and not cfg.enabled:
        raise ValueError("p2p_delta: combine compression needs delta dispatch")
    if not cfg.enabled:
        return cfg, combine
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    needs = {
        "all2all backend p2p_nixl": parallel.all2all_backend == "p2p_nixl",
        "expert parallelism": parallel.enable_expert_parallel,
        "tensor parallel size 1": parallel.tensor_parallel_size == 1,
        "pipeline parallel size 1": parallel.pipeline_parallel_size == 1,
        "no EPLB": not parallel.enable_eplb,
        "no sequence parallel MoE": not parallel.use_sequence_parallel_moe,
        f"at most {MAX_UBATCHES} micro-batches": parallel.num_ubatches <= MAX_UBATCHES,
        "a bf16 model": model is not None and model.dtype == torch.bfloat16,
        "--enforce-eager": model is not None and model.enforce_eager,
        "unquantized weights": vllm_config.quant_config is None,
    }
    missing = [need for need, ok in needs.items() if not ok]
    if missing:
        raise NotImplementedError(
            f"p2p_delta (delta dispatch {cfg.mode}) needs " + ", ".join(missing)
        )
    return cfg, combine


def engine_delta_kwargs(vllm_config: Any, hidden_dim: int) -> dict[str, Any]:
    """``P2pNixlAll2AllEngine`` arguments of delta dispatch ({} when off)."""
    cfg, combine = p2p_delta_configs(vllm_config)
    if not cfg.enabled:
        return {}
    return {
        "delta_row_bytes": cfg.max_row_bytes(hidden_dim),
        "combine_row_bytes": (
            combine.max_row_bytes(hidden_dim) if combine.enabled else 0
        ),
        "settings": repr((cfg, combine)),
    }


class P2pDeltaLayers:
    """The model side of delta dispatch on one rank: the post-attention norm
    of every MoE layer and, per micro-batch, the ``h'`` recorded for the MoE
    layer it runs next."""

    def __init__(self) -> None:
        self.norms: dict[int, nn.Module] = {}
        self._h: dict[int, tuple[int, torch.Tensor]] = {}

    def register(self, layer_idx: int, norm: nn.Module) -> None:
        if layer_idx in self.norms:
            raise ValueError(f"p2p_delta: MoE layer {layer_idx} registered twice")
        self.norms[layer_idx] = norm

    def record(self, layer_idx: int, h: torch.Tensor) -> None:
        """``h'`` of MoE layer ``layer_idx`` for the current micro-batch."""
        self._h[dbo_current_ubatch_id()] = (layer_idx, h)

    def take(self, layer_idx: int, ubatch: int) -> torch.Tensor:
        got = self._h.pop(ubatch, None)
        if got is None or got[0] != layer_idx:
            raise RuntimeError(
                f"p2p_delta: micro-batch {ubatch} recorded no h' for MoE layer "
                f"{layer_idx} (last: {None if got is None else got[0]})"
            )
        return got[1]


_LAYERS = P2pDeltaLayers()


def p2p_delta_layers() -> P2pDeltaLayers:
    """The model side of this process (one EP rank)."""
    return _LAYERS


def norm_from_residual(norm: nn.Module, out: torch.Tensor, h: torch.Tensor) -> None:
    """Write ``Norm(h)`` into ``out`` the way the sending rank computed it.

    The sender normalizes ``h'`` in the fused add + norm of
    ``post_attention_layernorm(x, residual)``. Calling the same module with a
    zero input and ``h`` as the residual runs that kernel on the same values
    (``0 + h = h``), so the result matches it bit for bit; the plain norm is
    another kernel whose reduction order differs. ``h`` is rewritten in place
    with the same values. (Same as afd-plugin's ``norm_from_residual``.)
    """
    out.zero_()
    normed, _ = norm(out, h)
    out.copy_(normed)


class OwnRowsFFNDeltaState(FFNDeltaState):
    """``FFNDeltaState`` with the rank's own rows: exact, after the received
    rows in the experts' input, outside the states."""

    def decode_with_own(
        self,
        layer_idx: int,
        stage: int,
        data: torch.Tensor,
        norm_into: Callable[[torch.Tensor, torch.Tensor], object],
        src: torch.Tensor,
        own_tokens: torch.Tensor,
        out_rows: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``decode`` with the rows ``src[own_tokens]`` (``h'``, or ``x`` when
        the mode sends ``x``) after the received ones: as they are in the x
        modes, else normalized in the same call as the received rows. The
        ``h`` buffer is sized for both before ``decode``, which then reuses
        it (``FFNDeltaState._buffer``) for the received rows."""
        own = int(own_tokens.shape[0])
        if self.cfg.sends_x:
            x = self.decode(layer_idx, stage, data, norm_into, out_rows, dtype)
            num_in = int(self._rows[stage].shape[0])
            if own:
                torch.index_select(src, 0, own_tokens, out=x[num_in : num_in + own])
            return x
        h = self._buffer(self._h, stage, out_rows, src.device, dtype)
        xbuf = self._buffer(self._x, stage, out_rows, src.device, dtype)
        normed: list[int] = []

        def norm_with_own(out: torch.Tensor, received: torch.Tensor) -> None:
            num_in = int(received.shape[0])
            if received.data_ptr() != h.data_ptr() or out.data_ptr() != xbuf.data_ptr():
                raise RuntimeError("p2p_delta: decode did not reuse the sized buffers")
            torch.index_select(src, 0, own_tokens, out=h[num_in : num_in + own])
            norm_into(xbuf[: num_in + own], h[: num_in + own])
            normed.append(num_in)

        x = self.decode(layer_idx, stage, data, norm_with_own, out_rows, dtype)
        if not normed and own:
            # No received rows: decode returned before the norm.
            torch.index_select(src, 0, own_tokens, out=h[:own])
            norm_into(x[:own], h[:own])
        return x


class P2pDeltaRank:
    """One EP rank's delta dispatch, shared by the prepare/finalize of every
    MoE layer: the settings, the model side and, from the first dispatch
    on, both roles' states."""

    def __init__(
        self,
        cfg: DeltaDispatchConfig,
        combine: CombineConfig,
        layers: P2pDeltaLayers,
        hidden: int,
        max_tokens: int,
        world_size: int,
        rank: int,
    ) -> None:
        if world_size > MAX_ATTENTION_RANKS:
            raise NotImplementedError(
                f"p2p_delta: the row words hold {MAX_ATTENTION_RANKS} ranks at "
                f"most, the EP group has {world_size}"
            )
        self.cfg, self.combine, self.layers = cfg, combine, layers
        self.hidden, self.max_tokens = hidden, max_tokens
        self.world_size, self.rank = world_size, rank
        # The smallest and largest row of the mode, in bytes.
        self.row_bytes = (cfg.min_row_bytes(hidden), cfg.max_row_bytes(hidden))
        self.routed_scale: float | None = None
        self._bound: set[int] = set()
        self._attn: AttentionDeltaState | None = None
        self._ffn: OwnRowsFFNDeltaState | None = None

    def bind(self, layer_idx: int, routed_scale: float) -> None:
        """A prepare/finalize serves MoE layer ``layer_idx``, whose runner
        scales the combined output by ``routed_scale``."""
        if self.routed_scale is not None and routed_scale != self.routed_scale:
            raise NotImplementedError(
                f"p2p_delta: MoE layer {layer_idx} has routed scale "
                f"{routed_scale}, the others {self.routed_scale}"
            )
        if self._attn is not None and layer_idx not in self._bound:
            raise RuntimeError(f"p2p_delta: MoE layer {layer_idx} bound too late")
        self.routed_scale = routed_scale
        self._bound.add(layer_idx)

    def states(self) -> tuple[AttentionDeltaState, OwnRowsFFNDeltaState]:
        """Both roles' states (made at the first call)."""
        if self._attn is None or self._ffn is None:
            layers = sorted(self.layers.norms)
            if not layers or set(layers) != self._bound:
                raise RuntimeError(
                    f"p2p_delta: the model registered MoE layers {layers}, the "
                    f"prepare/finalize serve {sorted(self._bound)} (is the "
                    "model wired for delta dispatch?)"
                )
            assert self.routed_scale is not None
            self._attn = AttentionDeltaState(
                self.cfg,
                layers,
                self.routed_scale,
                self.world_size,
                self.max_tokens,
                self.hidden,
                self.combine,
            )
            self._ffn = OwnRowsFFNDeltaState(
                self.cfg,
                layers,
                self.routed_scale,
                self.hidden,
                self.world_size,
                self.max_tokens,
                self.combine,
            )
        return self._attn, self._ffn


_RANKS: weakref.WeakKeyDictionary[P2pNixlAll2AllEngine, P2pDeltaRank] = (
    weakref.WeakKeyDictionary()
)


def p2p_delta_rank(engine: Any, vllm_config: Any, hidden: int) -> P2pDeltaRank | None:
    """The delta dispatch of ``engine``'s rank (one per engine), None when
    the engine carries plain rows."""
    if not getattr(engine, "delta_row_bytes", 0):
        return None
    rank = _RANKS.get(engine)
    if rank is None:
        cfg, combine = p2p_delta_configs(vllm_config)
        rank = P2pDeltaRank(
            cfg,
            combine,
            p2p_delta_layers(),
            hidden,
            engine.max_tokens_per_rank,
            engine.world_size,
            engine.rank,
        )
        _RANKS[engine] = rank
        # The run's record that delta dispatch is on, with which settings.
        logger.info(
            "p2p_delta EP rank %d/%d: %s, %s; states of %d tokens per rank",
            engine.rank,
            engine.world_size,
            cfg,
            combine,
            engine.max_tokens_per_rank,
        )
    return rank


@dataclasses.dataclass
class _Round:
    """What a slot's finalize needs from its prepare."""

    plan: P2pAll2AllPlan
    packed: PackedDispatch
    # Rows this rank sends itself, and their place in the dispatch order.
    own: int
    start: int


class P2pDeltaPrepareAndFinalize(P2pNcclPrepareAndFinalize):
    """``P2pNcclPrepareAndFinalize`` over a ``P2pNixlAll2AllEngine`` with
    delta dispatch (see the module docstring)."""

    def __init__(
        self,
        engine: P2pNixlAll2AllEngine,
        num_dispatchers: int,
        num_experts: int,
        num_local_experts: int,
        rank: int,
        delta: P2pDeltaRank,
    ) -> None:
        super().__init__(engine, num_dispatchers, num_experts, num_local_experts, rank)
        self.nixl = engine
        self.delta = delta
        self.layer_idx: int | None = None
        self._rounds: dict[int, _Round] = {}

    def bind_layer(self, runner: Any) -> None:
        """Serve the MoE layer of ``runner`` (``MoERunner``): its index and the
        routed scale it applies to the combined output."""
        if (
            runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
        ):
            raise NotImplementedError(
                "p2p_delta: routed input/output transforms (latent MoE)"
            )
        self.layer_idx = int(runner.layer_id)
        self.delta.bind(self.layer_idx, float(runner.routed_scaling_factor))

    def _check_sizes(self, info: list[tuple[int, int]]) -> None:
        """Each peer's bytes fit its rows' formats (a host check)."""
        low, high = self.delta.row_bytes
        for peer, (count, size) in enumerate(info):
            if peer != self.rank and not count * low <= size <= count * high:
                raise RuntimeError(
                    f"p2p_delta: {count} rows of {size} bytes from rank {peer} "
                    f"cannot be {self.delta.cfg.mode} rows"
                )

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
        if (
            apply_router_weight_on_input
            or defer_input_quant
            or quant_config.quant_dtype is not None
        ):
            raise NotImplementedError(
                "p2p_delta sends unquantized rows and applies the router "
                "weights after the experts"
            )
        layer = self.layer_idx
        if layer is None:
            raise RuntimeError("p2p_delta: prepare/finalize not bound to a layer")
        slot = dbo_current_ubatch_id()
        delta, engine = self.delta, self.nixl
        attn, ffn = delta.states()
        src = a1 if delta.cfg.sends_x else delta.layers.take(layer, slot)
        if src.shape != a1.shape or src.dtype != a1.dtype:
            raise RuntimeError(
                f"p2p_delta: h' {tuple(src.shape)} {src.dtype} does not match "
                f"the MoE input {tuple(a1.shape)} {a1.dtype}"
            )
        pending = self._begin_plan(topk_ids, slot)
        # As for the plain path's deferred-receive engines: no yield before
        # the plan wait, the receiver waits after the modular kernel's yield.
        dbo_switch_to_comm_sync()
        plan = engine.finish_plan(pending)
        num_tokens = plan.num_tokens
        counts = list(plan.send_counts)
        own = counts[self.rank]
        start = sum(counts[: self.rank])
        end = start + own
        counts[self.rank] = 0
        num = plan.num_send - own
        order, perm = pending.order, plan.perm
        # The peers' entries of the destination-major order and their tokens
        # (perm = order % num_tokens), this rank's own block cut out: views
        # unless the block lies between other ranks' blocks.
        if not own or end == plan.num_send:
            remote, tokens = order[:num], perm[:num]
        elif start == 0:
            remote, tokens = order[own : plan.num_send], perm[own:]
        else:
            remote = torch.cat((order[:start], order[end : plan.num_send]))
            tokens = torch.cat((perm[:start], perm[end:]))
        fixed = self._fixed_launch_ok()
        with _profile_range(f"p2p_delta.encode[{slot}]"):
            packed = attn.prepare(layer, slot, remote, num_tokens)
            if packed.totals is not None:
                # Rows of several sizes: the byte totals reach host memory
                # once the stream gets here.
                engine.wait_stream("delta row sizes")
            nbytes = attn.ffn_bytes(packed, counts)
            out_bytes, out_words, out_ids, out_weights = engine.delta_packs(slot)
            if num:
                attn.encode(packed, src, num, out_bytes, out_words, self.rank)
                torch.index_select(topk_ids, 0, tokens, out=out_ids[:num])
                torch.index_select(topk_weights, 0, tokens, out=out_weights[:num])
        recv_bytes, recv_words, recv_ids, recv_weights = engine.dispatch_delta(
            plan, counts, nbytes, min_rows=self._fixed_rows if fixed else 0
        )
        self._rounds[slot] = _Round(plan, packed, own, start)
        dbo_switch_to_compute_sync()

        def receiver() -> mk.PrepareResultType:
            info = engine.delta_dispatch_wait(plan, own)
            num_recv = plan.num_recv
            assert num_recv is not None
            num_in = num_recv - own
            self._check_sizes(info)
            total = sum(size for _, size in info)
            if fixed:
                ids, weights = recv_ids, recv_weights
            else:
                ids, weights = recv_ids[:num_recv], recv_weights[:num_recv]
            own_tokens = perm[start:end]
            if own:
                torch.index_select(topk_ids, 0, own_tokens, out=ids[num_in:num_recv])
                torch.index_select(
                    topk_weights, 0, own_tokens, out=weights[num_in:num_recv]
                )
            ffn.receive(slot, recv_words[:num_in])
            norm = delta.layers.norms[layer]
            with _profile_range(f"p2p_delta.decode[{slot}]"):
                x = ffn.decode_with_own(
                    layer,
                    slot,
                    recv_bytes[:total],
                    lambda out, h: norm_from_residual(norm, out, h),
                    src,
                    own_tokens,
                    out_rows=int(ids.shape[0]),
                    dtype=a1.dtype,
                )
            if fixed:
                # Fixed-size launch: the rows past num_recv get expert id -1
                # and the experts kernel learns how many rows are real.
                engine.mask_rows_past(slot, ids, num_recv)
                fixed_launch.set_valid_rows(num_recv, int(ids.shape[0]))
            else:
                fixed_launch.clear_valid_rows()
            return x, None, None, ids, weights

        return _noop_hook, receiver

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
        r = self._rounds.pop(slot, None)
        assert r is not None, "finalize called without a matching prepare"
        plan = r.plan
        num_recv = plan.num_recv
        assert num_recv is not None
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        # The real-row count was for this layer's experts call only.
        fixed_launch.clear_valid_rows()
        if fused_expert_output.shape[0] > num_recv:
            # Fixed-size launch: only the first num_recv rows are real.
            fused_expert_output = fused_expert_output[:num_recv]
            topk_weights = topk_weights[:num_recv]
            topk_ids = topk_ids[:num_recv]
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
        attn, ffn = self.delta.states()
        # The peers' rows' partial sums (compressed with combine compression)
        # into their states, on the stream the next layer's decode runs on.
        ffn.after_experts(slot, partial)
        compressed, row_bytes = None, 0
        if self.delta.combine.enabled:
            row_bytes = ffn.combine_row_bytes(slot)
            compressed = ffn.combine_bytes(slot, num_recv - r.own)
        dbo_yield_and_switch_from_compute_to_comm()
        back = self.nixl.delta_combine_send(plan, partial, r.own, compressed, row_bytes)
        dbo_switch_to_compute()

        def receiver() -> None:
            dbo_switch_to_comm()
            data = self.nixl.delta_combine_wait(plan)
            # Each peer's returned partial sums into its states; the own
            # block between them (rows to this rank) has none.
            num = plan.num_send - r.own
            if data is not None:
                # Compressed: rebuilt in fp32, into the states as they are and
                # around the own block in the model dtype.
                partials = attn.decode_combine(slot, data, num, r.packed.layer_pos)
                attn.after_combine(r.packed, num, partials)
                if r.start:
                    back[: r.start].copy_(partials[: r.start])
                if num > r.start:
                    back[r.start + r.own : plan.num_send].copy_(partials[r.start :])
            else:
                if r.start:
                    attn.after_combine(r.packed, r.start, back[: r.start])
                if num > r.start:
                    rest = dataclasses.replace(r.packed, flat=r.packed.flat[r.start :])
                    attn.after_combine(
                        rest, num - r.start, back[r.start + r.own : plan.num_send]
                    )
            self.nixl.delta_accumulate(plan, back, output)
            dbo_switch_to_compute_sync()

        return _noop_hook, receiver


__all__ = [
    "OwnRowsFFNDeltaState",
    "P2pDeltaLayers",
    "P2pDeltaPrepareAndFinalize",
    "P2pDeltaRank",
    "engine_delta_kwargs",
    "norm_from_residual",
    "p2p_delta_configs",
    "p2p_delta_layers",
    "p2p_delta_rank",
]
