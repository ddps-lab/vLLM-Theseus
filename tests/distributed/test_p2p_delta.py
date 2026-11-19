# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[theseus p2p_delta] Delta dispatch of the aggregated ``p2p_nixl`` path on
CPU: every EP rank in one process over a fake NIXL (host memory copies).

The lossless modes must give the plain path's (``off``) expert inputs and
combined outputs bit for bit, with each rank's own rows after the received
ones, and both sides' states must follow the closed loop of ``p2p_delta.py``
(checked against a direct reference, so a fault both sides share is caught
too).
"""

import ctypes
import sys
import threading
import types

import pytest
import torch

from vllm.distributed.device_communicators.p2p_delta import (
    CombineConfig,
    DeltaDispatchConfig,
)
from vllm.distributed.device_communicators.p2p_nixl_all2all import (
    P2pNixlAll2AllEngine,
)
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    p2p_delta as pf_delta,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    p2p_nccl as pf_nccl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.p2p_delta import (
    P2pDeltaLayers,
    P2pDeltaPrepareAndFinalize,
    P2pDeltaRank,
    engine_delta_kwargs,
    p2p_delta_configs,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.p2p_nccl import (
    P2pNcclPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

HIDDEN = 256
TOP_K = 3
NUM_EXPERTS = 8
MAX_TOKENS = 8
MOE_LAYERS = (1, 2, 3, 5)
EPS = 1e-6


class FakeNixl:
    """A NIXL stand-in: a WRITE is a host memory copy done when posted, and
    notifications land in the peer's inbox."""

    def __init__(self) -> None:
        self.agents: dict[str, FakeAgent] = {}
        self.module = types.ModuleType("nixl")
        self.module.nixl_agent = lambda name, config: FakeAgent(self, name)
        self.module.nixl_agent_config = lambda **kwargs: kwargs


class FakeAgent:
    def __init__(self, world: FakeNixl, name: str) -> None:
        self.world, self.name = world, name
        self.inbox: dict[str, list[bytes]] = {}
        world.agents[name] = self

    def register_memory(self, tensors):
        pass

    def get_agent_metadata(self):
        return self.name

    def add_remote_agent(self, metadata):
        return metadata.encode()

    def get_xfer_descs(self, descs, mem_type):
        return list(descs)

    def initialize_xfer(self, op, src, dst, peer, notif):
        assert op == "WRITE"
        return src, dst, peer, notif

    def transfer(self, handle):
        src, dst, peer, notif = handle
        for (s_addr, s_len, _), (d_addr, d_len, _) in zip(src, dst, strict=True):
            assert s_len == d_len
            ctypes.memmove(d_addr, s_addr, s_len)
        self.send_notif(peer, notif)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass

    def send_notif(self, peer, notif):
        self.world.agents[peer].inbox.setdefault(self.name, []).append(notif)

    def get_new_notifs(self):
        new, self.inbox = self.inbox, {}
        return new


def make_engines(monkeypatch, world, num_slots, kwargs_of=lambda rank: {}):
    """One engine per rank (extra arguments ``kwargs_of(rank)``), built in
    threads: the agent exchange is an all-gather over the ranks."""
    nixl = FakeNixl()
    monkeypatch.setitem(sys.modules, "nixl", nixl.module)
    barrier = threading.Barrier(world)
    infos: list = [None] * world

    def all_gather_object(out, obj, group=None):
        infos[group] = obj
        barrier.wait()
        out[:] = infos

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    engines: list = [None] * world
    errors: list = []

    def build(rank):
        try:
            engines[rank] = P2pNixlAll2AllEngine(
                group=rank,
                rank=rank,
                world_size=world,
                num_local_experts=NUM_EXPERTS // world,
                max_tokens_per_rank=MAX_TOKENS,
                device="cpu",
                hidden_dim=HIDDEN,
                top_k=TOP_K,
                act_dtype=torch.bfloat16,
                num_slots=num_slots,
                **kwargs_of(rank),
            )
        except BaseException as e:
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=build, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return engines


class FusedAddNorm(torch.nn.Module):
    """RMSNorm with a residual as the CUDA fused add + norm computes it: the
    sum rounded to bf16 first, then normalized; both inputs are rewritten in
    place."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x, residual):
        s = (x.float() + residual.float()).to(torch.bfloat16)
        var = s.float().pow(2).mean(-1, keepdim=True)
        out = (s.float() * torch.rsqrt(var + EPS)).to(torch.bfloat16) * self.weight
        residual.copy_(s)
        x.copy_(out)
        return x, residual


def owner_table(world, tables=None, layer=None):
    """The expert -> rank table of a layer: ``tables[layer]``, or the default
    contiguous one (expert id // experts per rank)."""
    if tables is not None:
        return tables[layer]
    return torch.arange(NUM_EXPERTS) // (NUM_EXPERTS // world)


def experts(rank, world, x, ids, weights, coef, owner=None):
    """Rank ``rank``'s experts (``owner``: the expert -> rank table, default
    contiguous) on the rows ``x`` (expert ``e`` scales by ``coef[e]``),
    reduced over the top-k in fp32, in bf16. Row by row, so a row's result
    does not depend on its position."""
    owner = owner_table(world) if owner is None else owner
    acc = torch.zeros(x.shape, dtype=torch.float32)
    for k in range(ids.shape[1]):
        e = ids[:, k].long()
        mine = (e >= 0) & (owner[e.clamp(0, NUM_EXPERTS - 1)] == rank)
        c = coef[e.clamp(0, NUM_EXPERTS - 1)] * mine
        acc += (weights[:, k] * c).unsqueeze(1) * x.float()
    return acc.to(torch.bfloat16)


def make_forwards(world, ubatches, norm, gen):
    """Two forwards of inputs per (MoE layer, rank, micro-batch): the MoE input
    x, h' (x = Norm(h')), expert ids and weights. Token counts differ per
    rank, micro-batch and forward; one rank has none in one micro-batch."""
    forwards = []
    for f in range(2):
        tokens = {
            (r, u): (MAX_TOKENS - (r + 2 * u + f) % 4) if (r, u, f) != (1, 0, 1) else 0
            for r in range(world)
            for u in range(ubatches)
        }
        residual = {
            key: torch.randn(t, HIDDEN, generator=gen).to(torch.bfloat16)
            for key, t in tokens.items()
        }
        layers = {}
        for layer in MOE_LAYERS:
            data = {}
            for key, t in tokens.items():
                attn = (0.5 * torch.randn(t, HIDDEN, generator=gen)).to(torch.bfloat16)
                x, h = norm(attn, residual[key].clone())
                picks = [
                    torch.randperm(NUM_EXPERTS, generator=gen)[:TOP_K] for _ in range(t)
                ]
                ids = (
                    torch.stack(picks).to(torch.int32)
                    if t
                    else torch.empty(0, TOP_K, dtype=torch.int32)
                )
                weights = torch.rand(t, TOP_K, generator=gen)
                data[key] = (x, h, ids, weights)
                residual[key] = h
            layers[layer] = data
        forwards.append(layers)
    return forwards


class Stub:
    """The MoE runner attributes ``bind_layer`` reads."""

    def __init__(self, layer, scale):
        self.layer_id = layer
        self.routed_scaling_factor = scale
        self.routed_input_transform = None
        self.routed_output_transform = None


def run(
    monkeypatch,
    world,
    ubatches,
    fixed,
    forwards,
    coef,
    mode,
    scale,
    norm,
    combine=None,
    tables=None,
):
    """Run every forward through one world (``mode`` "off", a delta mode or a
    ``delta_dispatch`` mapping; ``combine`` the combine settings; ``tables``
    per-layer expert -> rank tables). Returns per (forward, layer, rank,
    micro-batch) the output, the expert input rows (x, ids, weights; the
    first num_recv) and the receive counts, and for stateful modes both
    sides' states after every layer."""
    if mode == "off":
        engines = make_engines(monkeypatch, world, ubatches)
    else:
        raw = {"mode": mode} if isinstance(mode, str) else dict(mode)
        cfg = DeltaDispatchConfig.from_mapping(raw)
        ccfg = CombineConfig.from_mapping(combine)
        settings = {
            "delta_row_bytes": cfg.max_row_bytes(HIDDEN),
            "combine_row_bytes": ccfg.max_row_bytes(HIDDEN) if ccfg.enabled else 0,
            "settings": repr((raw, combine)),
        }
        engines = make_engines(monkeypatch, world, ubatches, lambda rank: settings)
        ranks = []
        for r in range(world):
            layers = P2pDeltaLayers()
            for layer in MOE_LAYERS:
                layers.register(layer, norm)
            ranks.append(P2pDeltaRank(cfg, ccfg, layers, HIDDEN, MAX_TOKENS, world, r))
    pfs = []
    for r, engine in enumerate(engines):
        per_layer = {}
        for layer in MOE_LAYERS:
            args = (engine, world, NUM_EXPERTS, NUM_EXPERTS // world, r)
            if mode == "off":
                pf = P2pNcclPrepareAndFinalize(*args)
            else:
                pf = P2pDeltaPrepareAndFinalize(*args, delta=ranks[r])
                pf.bind_layer(Stub(layer, scale))
            if fixed:
                pf._fixed_experts = types.SimpleNamespace()
            if tables is not None:
                pf.bind_expert_owner(tables[layer])
            per_layer[layer] = pf
        pfs.append(per_layer)
    current = {"u": 0}
    monkeypatch.setattr(pf_nccl, "dbo_current_ubatch_id", lambda: current["u"])
    monkeypatch.setattr(pf_delta, "dbo_current_ubatch_id", lambda: current["u"])
    out: dict = {}
    states: dict = {}
    for f, layers in enumerate(forwards):
        for layer, data in layers.items():
            prepared = {}
            for u in range(ubatches):
                current["u"] = u
                for r in range(world):
                    x, h, ids, weights = data[r, u]
                    if mode != "off":
                        ranks[r].layers.record(layer, h)
                    prepared[r, u] = pfs[r][layer].prepare_async(
                        x,
                        weights,
                        ids,
                        NUM_EXPERTS,
                        None,
                        False,
                        FUSED_MOE_UNQUANTIZED_CONFIG,
                    )
            finals = {}
            for u in range(ubatches):
                current["u"] = u
                for r in range(world):
                    pf = pfs[r][layer]
                    hook, receiver = prepared[r, u]
                    hook()
                    x_in, _, _, ids_in, w_in = receiver()
                    plan = pf._plans[u] if mode == "off" else pf._rounds[u].plan
                    n = plan.num_recv
                    owner = owner_table(world, tables, layer)
                    fused = experts(r, world, x_in, ids_in, w_in, coef, owner)
                    result = torch.empty(data[r, u][0].shape, dtype=torch.bfloat16)
                    finals[r, u] = pf.finalize_async(
                        result, fused, w_in, ids_in, False, TopKWeightAndReduceNoOP()
                    )
                    out[f, layer, r, u] = {
                        "x": x_in[:n].clone(),
                        "ids": ids_in[:n].clone(),
                        "weights": w_in[:n].clone(),
                        "recv_counts": list(plan.recv_counts),
                        "out": result,
                    }
            for u in range(ubatches):
                current["u"] = u
                for r in range(world):
                    hook, receiver = finals[r, u]
                    hook()
                    receiver()
            if mode != "off":
                states[f, layer] = [
                    (
                        {u: s.clone() for u, s in rank._attn._state.items()},
                        {u: s.clone() for u, s in rank._ffn._state.items()},
                    )
                    if rank._attn is not None and rank.cfg.stateful
                    else None
                    for rank in ranks
                ]
    return out, states


def reference_states(world, ubatches, forwards, coef, scale, tables=None):
    """The closed loop of ``p2p_delta.py`` computed directly: per (rank,
    micro-batch, destination, token) the state after each layer of the
    forward, for the rows the forward has visited so far (a forward starts
    with first visits). A token's own rank keeps none."""
    states: dict = {}
    for f, layers in enumerate(forwards):
        ref: dict = {}
        for layer, data in layers.items():
            owner = owner_table(world, tables, layer)
            for (r, u), (x, h, ids, weights) in data.items():
                owners = owner[ids.long()]
                for t in range(h.shape[0]):
                    for dest in sorted(set(owners[t].tolist())):
                        if dest == r:
                            continue
                        key = (r, u, dest, t)
                        if key not in ref:
                            s = h[t].float()
                        else:
                            s = ref[key]
                            s = s + (h[t].float() - s)
                        part = experts(
                            dest,
                            world,
                            x[t : t + 1],
                            ids[t : t + 1],
                            weights[t : t + 1],
                            coef,
                            owner,
                        )[0]
                        ref[key] = s + scale * part.float()
            states[f, layer] = dict(ref)
    return states


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
@pytest.mark.parametrize("ubatches", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
def test_lossless_modes_match_off(monkeypatch, world, ubatches, mode, fixed):
    """Expert inputs and outputs equal the plain path's bit for bit (own rows
    after the received ones); the states follow the reference."""
    gen = torch.Generator().manual_seed(1000 * world + 10 * ubatches + fixed)
    norm = FusedAddNorm(
        (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    )
    coef = 0.2 * torch.randn(NUM_EXPERTS, generator=gen)
    scale = 1.0 if world == 2 else 0.5
    forwards = make_forwards(world, ubatches, norm, gen)
    off, _ = run(
        monkeypatch, world, ubatches, fixed, forwards, coef, "off", scale, norm
    )
    delta, states = run(
        monkeypatch, world, ubatches, fixed, forwards, coef, mode, scale, norm
    )
    assert_matches_off(off, delta)
    if mode == "delta_fp32":
        assert_states_follow_reference(world, ubatches, forwards, coef, scale, states)


def assert_matches_off(off, delta):
    """The delta world's expert inputs (own rows after the received ones) and
    outputs equal the plain world's bit for bit."""
    for key, plain in off.items():
        got = delta[key]
        r = key[2]
        counts = plain["recv_counts"]
        assert got["recv_counts"] == counts
        a, own = sum(counts[:r]), counts[r]
        for name in ("x", "ids", "weights"):
            p = plain[name]
            expected = torch.cat((p[:a], p[a + own :], p[a : a + own]))
            assert torch.equal(got[name], expected), (key, name)
        assert torch.equal(got["out"], plain["out"]), key


def assert_states_follow_reference(
    world, ubatches, forwards, coef, scale, states, tables=None
):
    """Both sides' states equal the direct reference; own rows keep none."""
    ref = reference_states(world, ubatches, forwards, coef, scale, tables)
    for (f, layer), per_rank in states.items():
        tokens = {key: v[1].shape[0] for key, v in forwards[f][layer].items()}
        for (r, u, dest, t), s in ref[f, layer].items():
            attn = per_rank[r][0][u]
            ffn = per_rank[dest][1][u]
            n = tokens[r, u]
            assert torch.equal(attn[dest * n + t], s), (f, layer, r, u, dest, t)
            assert torch.equal(ffn[r * MAX_TOKENS + t], s), (f, layer, r, u, dest, t)
        for r, (attn_states, ffn_states) in enumerate(per_rank):
            for u in attn_states:
                # A rank's own rows never touch the states. (The Attention
                # side's rows move with the token count, so its own block is
                # clean only in the first forward.)
                assert not ffn_states[u][r * MAX_TOKENS : (r + 1) * MAX_TOKENS].any()
                if f == 0:
                    n = tokens[r, u]
                    assert not attn_states[u][r * n : (r + 1) * n].any()


def random_tables(world, gen):
    """Per MoE layer a random expert -> rank table, as many experts on every
    rank (no copies)."""
    tables = {}
    for layer in MOE_LAYERS:
        owner = torch.empty(NUM_EXPERTS, dtype=torch.int64)
        owner[torch.randperm(NUM_EXPERTS, generator=gen)] = torch.arange(
            NUM_EXPERTS
        ) // (NUM_EXPERTS // world)
        tables[layer] = owner
    return tables


@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
@pytest.mark.parametrize("world", [2, 4])
def test_placement_tables_route_each_layer(monkeypatch, world, mode):
    """With per-layer expert -> rank tables a rank receives exactly the
    tokens that pick one of its experts in that layer, and the delta path
    still gives the plain path's inputs and outputs and reference states."""
    ubatches = 2
    gen = torch.Generator().manual_seed(31 + world)
    norm = FusedAddNorm(
        (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    )
    coef = 0.2 * torch.randn(NUM_EXPERTS, generator=gen)
    tables = random_tables(world, gen)
    forwards = make_forwards(world, ubatches, norm, gen)
    args = (monkeypatch, world, ubatches, True, forwards, coef)
    off, _ = run(*args, "off", 1.0, norm, tables=tables)
    delta, states = run(*args, mode, 1.0, norm, tables=tables)
    for (f, layer, r, u), plain in off.items():
        for src in range(world):
            ids = forwards[f][layer][src, u][2].long()
            picks = (tables[layer][ids] == r).any(-1) if ids.numel() else ids
            assert plain["recv_counts"][src] == int(picks.sum())
    assert_matches_off(off, delta)
    if mode == "delta_fp32":
        assert_states_follow_reference(
            world, ubatches, forwards, coef, 1.0, states, tables
        )


def test_contiguous_table_is_the_default(monkeypatch):
    """The table expert id // experts per rank routes as no table does."""
    world, ubatches = 4, 2
    gen = torch.Generator().manual_seed(5)
    norm = FusedAddNorm(
        (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    )
    coef = 0.2 * torch.randn(NUM_EXPERTS, generator=gen)
    forwards = make_forwards(world, ubatches, norm, gen)
    tables = {layer: owner_table(world) for layer in MOE_LAYERS}
    args = (monkeypatch, world, ubatches, True, forwards, coef)
    plain, _ = run(*args, "off", 1.0, norm)
    tabled, _ = run(*args, "delta_fp32", 1.0, norm, tables=tables)
    assert_matches_off(plain, tabled)


def visited_rows(world, forwards, tables=None):
    """Per (forward, layer): the (rank, micro-batch, destination, token) rows
    the forward has sent to another rank so far."""
    out: dict = {}
    for f, layers in enumerate(forwards):
        seen: set = set()
        for layer, data in layers.items():
            for (r, u), (_, _, ids, _) in data.items():
                owners = owner_table(world, tables, layer)[ids.long()]
                for t in range(ids.shape[0]):
                    seen |= {(r, u, d, t) for d in set(owners[t].tolist()) if d != r}
            out[f, layer] = set(seen)
    return out


LOSSY = [
    {"mode": "x_int8"},
    {"mode": "x_nvfp4"},
    {"mode": "delta_int8", "first": "int8"},
    {"mode": "delta_nvfp4", "first": "nvfp4"},
    # First visits in bf16: rows of two sizes in one layer.
    {"mode": "delta_nvfp4"},
    {"mode": "delta_nvfp4", "first": "nvfp4", "int8_layers": [1, 3]},
    {"mode": "x_nvfp4", "int8_layers": [0, 2]},
]


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize(
    "settings", LOSSY, ids=lambda s: "-".join(str(v) for v in s.values())
)
@pytest.mark.parametrize("world", [2, 4])
def test_lossy_modes_keep_own_rows_exact(monkeypatch, world, settings, fixed):
    """With compressed rows a rank's own rows still reach its experts exact,
    the routing data travels as it is, the peers' rows come close, and both
    sides' states stay equal."""
    ubatches = 2
    gen = torch.Generator().manual_seed(77 + world + fixed)
    norm = FusedAddNorm(
        (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    )
    coef = 0.2 * torch.randn(NUM_EXPERTS, generator=gen)
    forwards = make_forwards(world, ubatches, norm, gen)
    args = (monkeypatch, world, ubatches, fixed, forwards, coef)
    off, _ = run(*args, "off", 1.0, norm)
    delta, states = run(*args, settings, 1.0, norm)
    worst, changed = 0.0, 0
    for key, plain in off.items():
        got = delta[key]
        r = key[2]
        counts = plain["recv_counts"]
        assert got["recv_counts"] == counts
        a, own = sum(counts[:r]), counts[r]
        num_in = sum(counts) - own
        for name in ("ids", "weights"):
            p = plain[name]
            assert torch.equal(
                got[name], torch.cat((p[:a], p[a + own :], p[a : a + own]))
            )
        assert torch.equal(got["x"][num_in:], plain["x"][a : a + own]), key
        peers = torch.cat((plain["x"][:a], plain["x"][a + own :])).float()
        if num_in:
            err = (got["x"][:num_in].float() - peers).norm(dim=-1)
            worst = max(worst, float((err / peers.norm(dim=-1)).max()))
            changed += int((err > 0).sum())
    assert 0 < worst < 0.3 and changed
    if not DeltaDispatchConfig.from_mapping(settings).stateful:
        return
    for (f, layer), rows in visited_rows(world, forwards).items():
        per_rank = states[f, layer]
        tokens = {key: v[1].shape[0] for key, v in forwards[f][layer].items()}
        for r, u, dest, t in rows:
            attn = per_rank[r][0][u][dest * tokens[r, u] + t]
            assert torch.equal(attn, per_rank[dest][1][u][r * MAX_TOKENS + t])


def test_ranks_must_share_settings(monkeypatch):
    cfg = DeltaDispatchConfig.from_mapping({"mode": "full_bf16"})
    size = cfg.max_row_bytes(HIDDEN)
    with pytest.raises(RuntimeError, match="delta dispatch settings"):
        make_engines(
            monkeypatch,
            2,
            1,
            lambda rank: {"delta_row_bytes": size, "settings": "ab"[rank]},
        )


def test_layers_hand_over_h_per_layer_and_micro_batch(monkeypatch):
    layers = P2pDeltaLayers()
    h = torch.zeros(2, HIDDEN)
    monkeypatch.setattr(pf_delta, "dbo_current_ubatch_id", lambda: 1)
    layers.record(3, h)
    with pytest.raises(RuntimeError, match="layer 4"):
        layers.take(4, 1)
    layers.record(3, h)
    assert layers.take(3, 1) is h
    with pytest.raises(RuntimeError):
        layers.take(3, 1)


def vllm_config(**changes):
    parallel = dict(
        all2all_backend="p2p_nixl",
        enable_expert_parallel=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        num_ubatches=2,
    )
    model = dict(dtype=torch.bfloat16, enforce_eager=True)
    config = dict(
        additional_config={"p2p_delta": {"delta_dispatch": {"mode": "delta_fp32"}}},
        quant_config=None,
    )
    for key, value in changes.items():
        for part in (parallel, model, config):
            if key in part:
                part[key] = value
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(**parallel),
        model_config=types.SimpleNamespace(**model),
        **config,
    )


def test_configs_accept_the_supported_setup():
    cfg, combine = p2p_delta_configs(vllm_config())
    assert cfg.mode == "delta_fp32" and not combine.enabled
    kwargs = engine_delta_kwargs(vllm_config(), HIDDEN)
    assert kwargs["delta_row_bytes"] == 4 * HIDDEN
    assert engine_delta_kwargs(vllm_config(additional_config={}), HIDDEN) == {}


@pytest.mark.parametrize(
    "change, need",
    [
        ({"all2all_backend": "p2p_nccl"}, "p2p_nixl"),
        ({"tensor_parallel_size": 2}, "tensor parallel"),
        ({"enforce_eager": False}, "enforce-eager"),
        ({"dtype": torch.float16}, "bf16"),
        ({"enable_eplb": True}, "EPLB"),
        ({"use_sequence_parallel_moe": True}, "sequence parallel"),
        ({"num_ubatches": 4}, "micro-batches"),
        ({"quant_config": object()}, "unquantized"),
    ],
)
def test_configs_refuse_unsupported_setups(change, need):
    with pytest.raises(NotImplementedError, match=need):
        p2p_delta_configs(vllm_config(**change))


def test_combine_compression_sizes_the_engine():
    config = vllm_config(
        additional_config={
            "p2p_delta": {
                "delta_dispatch": {"mode": "delta_nvfp4", "first": "nvfp4"},
                "combine": {"mode": "nvfp4", "int8_layers": [0]},
            }
        }
    )
    kwargs = engine_delta_kwargs(config, HIDDEN)
    # The widest combine row: int8 on the listed layer.
    assert kwargs["combine_row_bytes"] == HIDDEN + HIDDEN // 128 * 4
    alone = vllm_config(additional_config={"p2p_delta": {"combine": {"mode": "int8"}}})
    with pytest.raises(ValueError, match="needs delta dispatch"):
        p2p_delta_configs(alone)


COMBINED = [
    ({"mode": "x_int8"}, {"mode": "int8", "ef": False}),
    ({"mode": "x_nvfp4"}, {"mode": "nvfp4", "ef": False}),
    ({"mode": "delta_int8", "first": "int8"}, {"mode": "int8", "ef": True}),
    ({"mode": "delta_nvfp4", "first": "nvfp4"}, {"mode": "nvfp4", "ef": True}),
    (
        {"mode": "delta_nvfp4", "first": "nvfp4", "int8_layers": [1, 3]},
        {"mode": "nvfp4", "ef": True, "int8_layers": [1, 3]},
    ),
]


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize(
    "dispatch, combine",
    COMBINED,
    ids=lambda s: "-".join(str(v) for v in s.values()),
)
@pytest.mark.parametrize("world", [2, 4])
def test_combine_compression_keeps_own_rows_exact(
    monkeypatch, world, dispatch, combine, fixed
):
    """With compressed partial sums a token served by its own rank only gets
    the plain path's output exactly, the others come close, and both sides'
    states (built from what was sent and what was rebuilt) stay equal."""
    ubatches = 2
    gen = torch.Generator().manual_seed(91 + world + fixed)
    norm = FusedAddNorm(
        (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    )
    coef = 0.2 * torch.randn(NUM_EXPERTS, generator=gen)
    forwards = make_forwards(world, ubatches, norm, gen)
    args = (monkeypatch, world, ubatches, fixed, forwards, coef)
    off, _ = run(*args, "off", 1.0, norm)
    delta, states = run(*args, dispatch, 1.0, norm, combine)
    alone, errors = 0, []
    for (f, layer, r, u), plain in off.items():
        got = delta[f, layer, r, u]["out"]
        ids = forwards[f][layer][r, u][2].long()
        only_own = (ids // (NUM_EXPERTS // world) == r).all(-1)
        assert torch.equal(got[only_own], plain["out"][only_own])
        alone += int(only_own.sum())
        if (~only_own).any():
            p, g = plain["out"][~only_own].float(), got[~only_own].float()
            errors.append((g - p).norm(dim=-1) / p.norm(dim=-1).clamp_min(1e-6))
    # Relative errors, per token: the largest ones come from partial sums that
    # nearly cancel, so the median is checked.
    errors = torch.cat(errors)
    assert (errors > 0).any() and float(errors.median()) < 0.2
    # With two experts a rank (4 ranks) no token keeps all its top-3 home.
    assert alone or world == 4
    if not DeltaDispatchConfig.from_mapping(dispatch).stateful:
        return
    for (f, layer), rows in visited_rows(world, forwards).items():
        per_rank = states[f, layer]
        tokens = {key: v[1].shape[0] for key, v in forwards[f][layer].items()}
        for r, u, dest, t in rows:
            attn = per_rank[r][0][u][dest * tokens[r, u] + t]
            assert torch.equal(attn, per_rank[dest][1][u][r * MAX_TOKENS + t])


def encode_decode(module, mode, first, seed):
    """One encode and decode of rows of several (destination, token) pairs,
    first visits and revisits mixed, with the CPU reference of ``module``."""
    gen = torch.Generator().manual_seed(seed)
    raw = {"mode": mode} if first is None else {"mode": mode, "first": first}
    cfg = module.DeltaDispatchConfig.from_mapping(raw)
    tokens, dests, pos, count = 5, 3, 2, 6
    src = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    state = torch.randn(dests * tokens, HIDDEN, generator=gen)
    last = torch.randint(-1, pos, (dests * tokens,), generator=gen, dtype=torch.int32)
    flat = torch.arange(dests * tokens)
    codes = cfg.row_codes(pos, count, last[flat]).to(torch.int32)
    sizes = torch.tensor([module.row_bytes(int(c), HIDDEN) for c in codes])
    offsets = torch.cumsum(sizes, 0) - sizes
    out = torch.zeros(int(sizes.sum()), dtype=torch.uint8)
    meta = torch.zeros(len(flat), dtype=torch.int32)
    ffn_state = state.clone()
    module.encode_rows_reference(
        cfg,
        src,
        state if cfg.stateful else None,
        last,
        pos,
        flat,
        codes,
        offsets,
        tokens,
        1,
        out,
        meta,
    )
    h = torch.empty(len(flat), HIDDEN, dtype=torch.bfloat16)
    module.decode_rows_reference(
        cfg, out, meta, offsets, flat, ffn_state if cfg.stateful else None, h
    )
    return out, meta, state, last, h, ffn_state


def combine_round(module, code, seed):
    """One compressed combine (with error feedback) and its decode."""
    gen = torch.Generator().manual_seed(seed)
    n = 7
    partial = torch.randn(n, HIDDEN, generator=gen).to(torch.bfloat16)
    err = 0.01 * torch.randn(n, HIDDEN, generator=gen)
    out = torch.zeros(n * module.row_bytes(code, HIDDEN), dtype=torch.uint8)
    deq = torch.empty(n, HIDDEN)
    module.encode_combine_reference(code, partial, torch.arange(n), err, out, deq)
    back = torch.empty(n, HIDDEN)
    module.decode_combine_reference(code, out, back)
    return out, deq, err, back


def test_copy_matches_afd_plugin():
    """The vLLM copy gives the plugin's bytes, words, states and rebuilt
    values (``p2p_delta.py`` keeps the two in step; skipped without the
    plugin)."""
    plugin = pytest.importorskip("afd_plugin.delta_dispatch")
    from vllm.distributed.device_communicators import p2p_delta as copy

    modes = [
        ("full_bf16", None),
        ("delta_fp32", None),
        ("delta_int8", "int8"),
        ("delta_int4", None),
        ("delta_int3", None),
        ("delta_nvfp4", None),
        ("delta_nvfp4", "nvfp4"),
        ("x_int4", None),
        ("x_int8", None),
        ("x_nvfp4", None),
    ]
    for seed, (mode, first) in enumerate(modes):
        ours = encode_decode(copy, mode, first, seed)
        theirs = encode_decode(plugin, mode, first, seed)
        for a, b in zip(ours, theirs, strict=True):
            assert torch.equal(a, b), (mode, first)
    for code in (copy.D_INT8, copy.D_INT4, copy.D_NVFP4):
        ours = combine_round(copy, code, code)
        theirs = combine_round(plugin, code, code)
        for a, b in zip(ours, theirs, strict=True):
            assert torch.equal(a, b), code
