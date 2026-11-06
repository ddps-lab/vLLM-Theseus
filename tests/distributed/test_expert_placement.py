# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.device_communicators.p2p_nccl_all2all import P2pAll2AllEngine
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    EXPERT_PLACEMENT_FORMAT,
    ExpertPlacement,
    determine_expert_map,
    expert_placement_for_layer,
    load_expert_placement,
)


def verify_round_robin_pattern(expert_map, ep_rank, ep_size, global_num_experts):
    """Verify that the expert map follows the round_robin pattern."""
    # Calculate expected local experts (supporting non-divisible cases)
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size

    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Expected expert IDs for this rank in round_robin pattern
    # For non-divisible cases, ranks with extra experts start earlier
    expected_expert_ids = []
    for expert_idx in range(local_num_experts):
        global_expert_id = ep_rank + expert_idx * ep_size
        expected_expert_ids.append(global_expert_id)

    # Check that only expected experts are mapped to this rank
    for global_expert_id in range(global_num_experts):
        if global_expert_id in expected_expert_ids:
            local_expert_id = expert_map[global_expert_id]
            expected_local_id = expected_expert_ids.index(global_expert_id)
            assert local_expert_id == expected_local_id, (
                f"Global expert {global_expert_id} should map to local expert "
                f"{expected_local_id}, got {local_expert_id}"
            )
        else:
            assert expert_map[global_expert_id] == -1, (
                f"Global expert {global_expert_id} should not be mapped to this rank"
            )

    # Verify that all local expert IDs are consecutive starting from 0
    local_expert_ids = [expert_map[global_id] for global_id in expected_expert_ids]
    expected_local_ids = list(range(local_num_experts))
    assert local_expert_ids == expected_local_ids, (
        f"Expected local expert IDs {expected_local_ids}, got {local_expert_ids}"
    )


@pytest.mark.parametrize("expert_placement_strategy", ["round_robin"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_expert_placement_various_sizes(expert_placement_strategy, world_size):
    """Test round_robin expert placement with various expert counts."""

    # Test with different global_num_experts values
    # Include both divisible and non-divisible cases
    if world_size == 2:
        test_cases = [
            (4, 2),  # 4 experts (divisible)
            (8, 2),  # 8 experts (divisible)
            (9, 2),  # 9 experts (non-divisible)
            (16, 2),  # 16 experts (divisible)
            (17, 2),  # 17 experts (non-divisible)
        ]
    elif world_size == 4:
        test_cases = [
            (8, 4),  # 8 experts (divisible)
            (16, 4),  # 16 experts (divisible)
            (18, 4),  # 18 experts (non-divisible)
            (32, 4),  # 32 experts (divisible)
            (33, 4),  # 33 experts (non-divisible)
        ]
    else:
        test_cases = []

    for test_global_experts, test_ep_size in test_cases:
        # Ensure ep_size matches world_size
        assert test_ep_size == world_size, (
            f"ep_size {test_ep_size} must equal world_size {world_size}"
        )

        # Test each rank
        for ep_rank in range(world_size):
            # Calculate expected local experts
            base_experts = test_global_experts // test_ep_size
            remainder = test_global_experts % test_ep_size
            if ep_rank < remainder:
                expected_test_local = base_experts + 1
            else:
                expected_test_local = base_experts

            test_local_experts, test_expert_map, _ = determine_expert_map(
                ep_size=test_ep_size,
                ep_rank=ep_rank,
                global_num_experts=test_global_experts,
                expert_placement_strategy=expert_placement_strategy,
            )

            assert test_local_experts == expected_test_local, (
                f"For {test_global_experts} experts on {test_ep_size} ranks, "
                f"rank {ep_rank}: expected {expected_test_local} local"
                f"experts, got {test_local_experts}"
            )

            if test_expert_map is not None:
                assert test_expert_map.shape == (test_global_experts,), (
                    f"Expected expert map shape ({test_global_experts},), "
                    f"got {test_expert_map.shape}"
                )

                # Verify round_robin pattern for this test case
                verify_round_robin_pattern(
                    test_expert_map, ep_rank, test_ep_size, test_global_experts
                )


@pytest.mark.parametrize("expert_placement_strategy", ["round_robin"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_expert_placement_edge_cases(expert_placement_strategy, world_size):
    """Test edge cases for round_robin expert placement."""

    # Test case 1: ep_size = 1 (should return None for expert_map)
    local_num_experts, expert_map, _ = determine_expert_map(
        ep_size=1,
        ep_rank=0,
        global_num_experts=8,
        expert_placement_strategy=expert_placement_strategy,
    )
    assert local_num_experts == 8, "For ep_size=1, should get all experts"
    assert expert_map is None, "For ep_size=1, expert_map should be None"

    # Test case 2: ep_size = 0 (should raise assertion)
    with pytest.raises(AssertionError):
        determine_expert_map(
            ep_size=0,
            ep_rank=0,
            global_num_experts=8,
            expert_placement_strategy=expert_placement_strategy,
        )


def test_determine_expert_map_comprehensive():
    """Test of determine_expert_map function with various configurations."""

    # Test cases: (ep_size, ep_rank, global_num_experts,
    # expert_placement_strategy, expected_local, expected_map_pattern)
    test_cases = [
        # Round robin placement tests
        (
            2,
            0,
            8,
            "round_robin",
            4,
            [0, -1, 1, -1, 2, -1, 3, -1],
        ),  # rank 0 gets even experts
        (
            2,
            1,
            8,
            "round_robin",
            4,
            [-1, 0, -1, 1, -1, 2, -1, 3],
        ),  # rank 1 gets odd experts
        (
            2,
            0,
            9,
            "round_robin",
            5,
            [0, -1, 1, -1, 2, -1, 3, -1, 4],
        ),  # rank 0 gets 5 experts (even + last)
        (
            2,
            1,
            9,
            "round_robin",
            4,
            [-1, 0, -1, 1, -1, 2, -1, 3, -1],
        ),  # rank 1 gets 4 experts (odd)
        # 4-rank tests
        (
            4,
            0,
            8,
            "round_robin",
            2,
            [0, -1, -1, -1, 1, -1, -1, -1],
        ),  # rank 0 gets experts 0, 4
        (
            4,
            1,
            8,
            "round_robin",
            2,
            [-1, 0, -1, -1, -1, 1, -1, -1],
        ),  # rank 1 gets experts 1, 5
        (
            4,
            2,
            8,
            "round_robin",
            2,
            [-1, -1, 0, -1, -1, -1, 1, -1],
        ),  # rank 2 gets experts 2, 6
        (
            4,
            3,
            8,
            "round_robin",
            2,
            [-1, -1, -1, 0, -1, -1, -1, 1],
        ),  # rank 3 gets experts 3, 7
    ]

    for (
        ep_size,
        ep_rank,
        global_num_experts,
        expert_placement_strategy,
        expected_local,
        expected_map_pattern,
    ) in test_cases:
        local_num_experts, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy=expert_placement_strategy,
        )

        assert local_num_experts == expected_local, (
            f"ep_size={ep_size}, ep_rank={ep_rank}, "
            f"global_num_experts={global_num_experts}, "
            f"expert_placement_strategy={expert_placement_strategy}: "
            f"expected {expected_local} local experts, got {local_num_experts}"
        )

        if expected_map_pattern is None:
            assert expert_map is None, "Expected expert_map to be None"
        else:
            assert expert_map is not None, "Expected expert_map to not be None"
            actual_map = expert_map.tolist()
            assert actual_map == expected_map_pattern, (
                f"ep_size={ep_size}, ep_rank={ep_rank}, "
                f"global_num_experts={global_num_experts}, "
                f"expert_placement_strategy={expert_placement_strategy}: "
                f"expected map {expected_map_pattern}, got {actual_map}"
            )


@pytest.mark.parametrize("counts", [[48, 16], [16, 48], [3, 1, 4]])
def test_placement_counts_give_contiguous_blocks(counts):
    """Rank r owns the counts[r] experts that follow the lower ranks'."""
    start = 0
    for ep_rank, count in enumerate(counts):
        local_num_experts, expert_map, _ = determine_expert_map(
            len(counts), ep_rank, sum(counts), expert_placement_counts=counts
        )
        assert local_num_experts == count
        expected = [-1] * sum(counts)
        expected[start : start + count] = range(count)
        assert expert_map.tolist() == expected
        start += count


@pytest.mark.parametrize(("global_num_experts", "ep_size"), [(64, 2), (9, 2), (8, 3)])
def test_even_placement_counts_match_default(global_num_experts, ep_size):
    """Counts of the default split reproduce it, remainder on the first ranks."""
    for ep_rank in range(ep_size):
        default = determine_expert_map(ep_size, ep_rank, global_num_experts)
        counts = [
            determine_expert_map(ep_size, r, global_num_experts)[0]
            for r in range(ep_size)
        ]
        explicit = determine_expert_map(
            ep_size, ep_rank, global_num_experts, expert_placement_counts=counts
        )
        assert explicit[0] == default[0]
        assert torch.equal(explicit[1], default[1])


@pytest.mark.parametrize(
    ("counts", "strategy"),
    [([48, 15], "linear"), ([32, 16, 16], "linear"), ([48, 16], "round_robin")],
)
def test_invalid_placement_counts_raise(counts, strategy):
    with pytest.raises(ValueError):
        determine_expert_map(2, 0, 64, strategy, expert_placement_counts=counts)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expert_placement_counts": [48, 0]},
        {
            "expert_placement_counts": [48, 16],
            "expert_placement_strategy": "round_robin",
        },
        {"expert_placement_counts": [48, 16], "all2all_backend": "p2p_nccl"},
        {"expert_placement_counts": [48, 16], "enable_ep_weight_filter": True},
    ],
)
def test_placement_counts_config_rejects_unsupported(overrides):
    with pytest.raises(ValueError):
        ParallelConfig(**overrides)


def test_p2p_engine_routes_by_placement_counts():
    """Tokens visit the ranks that own their experts; padding visits none."""
    # Two expert ranks owning experts [0, 3) and [3, 4), one rank without.
    engine = P2pAll2AllEngine(
        group=None,
        rank=2,
        world_size=3,
        num_local_experts=2,
        max_tokens_per_rank=4,
        device="cpu",
        expert_counts=[3, 1],
    )
    topk_ids = torch.tensor([[0, 3], [3, -1], [1, 2], [-1, -1]])
    pending = engine.begin_plan(topk_ids)
    assert engine._count_buf(0)[0].tolist() == [2, 2, 0]
    # Destination-major order: rank 0 gets tokens 0 and 2, rank 1 tokens 0, 1.
    assert (pending.order[:4] % 4).tolist() == [0, 2, 0, 1]


# [theseus expert placement]
def test_placement_experts_map_in_list_order():
    """Local id i is the i-th listed expert, whatever the order."""
    local_num_experts, expert_map, expert_mask = determine_expert_map(
        4, 1, 8, return_expert_mask=True, expert_placement_experts=[6, 1, 3]
    )
    assert local_num_experts == 3
    assert expert_map.dtype == torch.int32
    assert expert_map.tolist() == [-1, 1, -1, 2, -1, -1, 0, -1]
    assert expert_mask.tolist() == [0, 1, 0, 1, 0, 0, 1, 0, 0]


@pytest.mark.parametrize(("global_num_experts", "ep_size"), [(64, 4), (9, 2), (10, 4)])
def test_contiguous_placement_experts_match_default(global_num_experts, ep_size):
    """The blocks of the default linear split, listed, reproduce its maps."""
    for ep_rank in range(ep_size):
        default = determine_expert_map(ep_size, ep_rank, global_num_experts)
        experts = torch.where(default[1] >= 0)[0].tolist()
        explicit = determine_expert_map(
            ep_size, ep_rank, global_num_experts, expert_placement_experts=experts
        )
        assert explicit[0] == default[0]
        assert torch.equal(explicit[1], default[1])


@pytest.mark.parametrize(
    ("experts", "strategy", "counts"),
    [
        ([1, 1], "linear", None),
        ([0, 8], "linear", None),
        ([-1, 2], "linear", None),
        ([], "linear", None),
        ([0, 1], "linear", [4, 4]),
        ([0, 1], "round_robin", None),
    ],
)
def test_invalid_placement_experts_raise(experts, strategy, counts):
    with pytest.raises(ValueError):
        determine_expert_map(
            2,
            0,
            8,
            strategy,
            expert_placement_counts=counts,
            expert_placement_experts=experts,
        )


def write_placement(tmp_path, layers, **fields):
    data = {
        "format": EXPERT_PLACEMENT_FORMAT,
        "num_experts": 4,
        "num_ranks": 2,
        "layers": layers,
        **fields,
    }
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_placement_file_round_trip(tmp_path):
    """Every layer's lists come back per rank; the layer comes from the name."""
    path = write_placement(tmp_path, {"5": [[3, 0], [1, 2]], "7": [[0, 1, 2], [3]]})
    assert load_expert_placement(path) == ExpertPlacement(
        num_experts=4,
        num_ranks=2,
        layers={5: ((3, 0), (1, 2)), 7: ((0, 1, 2), (3,))},
    )
    rank0_layer5 = expert_placement_for_layer(
        path, "model.layers.5.mlp.experts", 0, 2, 4, "allgather_reducescatter"
    )
    rank1_layer7 = expert_placement_for_layer(
        path, "model.layers.7.mlp.experts", 1, 2, 4, "allgather_reducescatter"
    )
    assert rank0_layer5 == [3, 0]
    assert rank1_layer7 == [3]


def test_placement_file_allows_copies(tmp_path):
    """An expert listed on two ranks is local on both."""
    path = write_placement(tmp_path, {"1": [[0, 1, 2], [2, 3]]})
    maps = [
        determine_expert_map(
            2,
            rank,
            4,
            expert_placement_experts=expert_placement_for_layer(
                path, "model.layers.1.mlp.experts", rank, 2, 4, "prerouted"
            ),
        )[1].tolist()
        for rank in range(2)
    ]
    assert maps == [[0, 1, 2, -1], [-1, -1, 0, 1]]


def test_placement_has_copies():
    placement = ExpertPlacement(
        num_experts=4, num_ranks=2, layers={1: ((0, 1, 2), (2, 3)), 2: ((0, 1), (2, 3))}
    )
    assert placement.has_copies(1)
    assert not placement.has_copies(2)


@pytest.mark.parametrize(
    ("backend", "layers", "ok"),
    [
        ("allgather_reducescatter", {"1": [[0, 1, 2], [2, 3]]}, False),
        ("prerouted", {"1": [[0, 1, 2], [2, 3]]}, True),
        ("allgather_reducescatter", {"1": [[0, 1], [2, 3]]}, True),
    ],
)
def test_placement_copies_need_prerouted(tmp_path, backend, layers, ok):
    """Only the sender of prerouted tokens can pick one copy per token; with
    another backend every copy would compute it and the sum count it twice."""
    path = write_placement(tmp_path, layers)
    name = "model.layers.1.mlp.experts"
    if ok:
        experts = expert_placement_for_layer(path, name, 0, 2, 4, backend)
        assert experts == layers["1"][0]
    else:
        with pytest.raises(ValueError, match="prerouted"):
            expert_placement_for_layer(path, name, 0, 2, 4, backend)


@pytest.mark.parametrize(
    ("fields", "layers"),
    [
        ({"format": "theseus-expert-placement-v0"}, {"1": [[0, 1], [2, 3]]}),
        ({"num_ranks": 0}, {"1": []}),
        ({}, {"1": [[0, 1], [2]]}),
        ({}, {"1": [[0, 1, 2, 3]]}),
        ({}, {"1": [[0, 0, 1], [2, 3]]}),
        ({}, {"1": [[0, 1, 4], [2, 3]]}),
        ({}, {"1": [[0, 1, 2, 3], []]}),
        ({}, {"x": [[0, 1], [2, 3]]}),
    ],
)
def test_invalid_placement_file_raises(tmp_path, fields, layers):
    path = write_placement(tmp_path, layers, **fields)
    with pytest.raises(ValueError):
        load_expert_placement(path)


@pytest.mark.parametrize(
    ("layer_name", "ep_size", "global_num_experts"),
    [
        ("model.layers.2.mlp.experts", 2, 4),
        ("model.layers.1.mlp.experts", 4, 4),
        ("model.layers.1.mlp.experts", 2, 8),
    ],
)
def test_placement_file_mismatch_raises(
    tmp_path, layer_name, ep_size, global_num_experts
):
    """Missing layer, other EP size or other expert count."""
    path = write_placement(tmp_path, {"1": [[0, 1], [2, 3]]})
    with pytest.raises(ValueError):
        expert_placement_for_layer(
            path, layer_name, 0, ep_size, global_num_experts, "prerouted"
        )


def test_placement_file_config_reads_no_file():
    path = "/nonexistent/placement.json"
    assert ParallelConfig(expert_placement_file=path).expert_placement_file == path


@pytest.mark.parametrize(
    "overrides",
    [
        {"expert_placement_counts": [48, 16]},
        {"expert_placement_strategy": "round_robin"},
        {"enable_ep_weight_filter": True},
    ],
)
def test_placement_file_config_rejects_unsupported(overrides):
    with pytest.raises(ValueError):
        ParallelConfig(expert_placement_file="placement.json", **overrides)


def test_placement_file_backend_is_checked_per_layer(tmp_path):
    """AFD with microbatching validates the config under a temporary all2all
    backend, so the backend is checked where a MoE layer reads the file."""
    ParallelConfig(
        expert_placement_file="placement.json", all2all_backend="deepep_low_latency"
    )
    path = write_placement(tmp_path, {"1": [[0, 1], [2, 3]]})
    name = "model.layers.1.mlp.experts"
    for backend in ("p2p_nccl", "deepep_low_latency"):
        with pytest.raises(ValueError, match="all2all backends"):
            expert_placement_for_layer(path, name, 0, 2, 4, backend)
    for backend in ("allgather_reducescatter", "prerouted"):
        assert expert_placement_for_layer(path, name, 0, 2, 4, backend) == [0, 1]


def test_p2p_engine_routes_by_per_call_owner():
    """A per-call expert -> rank table replaces the engine's own routing."""
    engine = P2pAll2AllEngine(
        group=None,
        rank=2,
        world_size=3,
        num_local_experts=2,
        max_tokens_per_rank=4,
        device="cpu",
        expert_counts=[3, 1],
    )
    topk_ids = torch.tensor([[0, 3], [3, -1], [1, 2], [-1, -1]])
    # Experts 1 and 2 on rank 0, experts 0 and 3 on rank 1.
    owner = torch.tensor([1, 0, 0, 1])
    pending = engine.begin_plan(topk_ids, expert_owner=owner)
    assert engine._count_buf(0)[0].tolist() == [1, 2, 0]
    # Destination-major order: rank 0 gets token 2, rank 1 tokens 0 and 1.
    assert (pending.order[:3] % 4).tolist() == [2, 0, 1]


@pytest.mark.parametrize("expert_counts", [None, [40, 8, 8, 8]])
def test_p2p_engine_default_owner_table_matches_default(expert_counts):
    """The table of the engine's own routing gives the same counts and order."""
    engine = P2pAll2AllEngine(
        group=None,
        rank=4,
        world_size=5,
        num_local_experts=16,
        max_tokens_per_rank=32,
        device="cpu",
        expert_counts=expert_counts,
    )
    generator = torch.Generator().manual_seed(0)
    topk_ids = torch.randint(-1, 64, (32, 6), generator=generator)
    if expert_counts is None:
        owner = torch.arange(64) // 16
    else:
        owner = torch.repeat_interleave(torch.arange(4), torch.tensor(expert_counts))
    default = engine.begin_plan(topk_ids)
    default_counts = engine._count_buf(0)[0].clone()
    explicit = engine.begin_plan(topk_ids, expert_owner=owner)
    assert torch.equal(engine._count_buf(0)[0], default_counts)
    assert torch.equal(explicit.order, default.order)


def make_rank_of_engine():
    # Two expert ranks owning experts [0, 3) and [3, 4), one rank without.
    return P2pAll2AllEngine(
        group=None,
        rank=2,
        world_size=3,
        num_local_experts=2,
        max_tokens_per_rank=4,
        device="cpu",
        expert_counts=[3, 1],
    )


def test_p2p_engine_routes_by_per_pair_ranks():
    """Each (token, expert) pair goes where rank_of says, not to the owner."""
    engine = make_rank_of_engine()
    topk_ids = torch.tensor([[0, 3], [3, 1], [1, 2], [-1, -1]])
    # Token 0: both experts on rank 1 (one visit); token 1: expert 3's copy on
    # rank 0, expert 1 nowhere; token 2: both ranks; token 3: padding.
    rank_of = torch.tensor([[1, 1], [0, -1], [1, 0], [-1, -1]])
    pending = engine.begin_plan(topk_ids, rank_of=rank_of)
    assert engine._count_buf(0)[0].tolist() == [2, 2, 0]
    # Destination-major order of rank * 4 + token: rank 0 gets tokens 1, 2,
    # rank 1 tokens 0, 2; the rest is padding (3 * 4).
    assert pending.order[:4].tolist() == [1, 2, 4, 6]
    assert (pending.order[4:] == 12).all()


def test_p2p_engine_per_pair_ranks_match_owner_table():
    """rank_of read from an expert -> rank table gives that table's plan."""
    engine = P2pAll2AllEngine(
        group=None,
        rank=4,
        world_size=5,
        num_local_experts=16,
        max_tokens_per_rank=32,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(0)
    topk_ids = torch.randint(-1, 64, (32, 6), generator=generator)
    owner = torch.randint(0, 4, (64,), generator=generator)
    rank_of = torch.where(topk_ids >= 0, owner[topk_ids.clamp(min=0)], -1)
    by_owner = engine.begin_plan(topk_ids, expert_owner=owner)
    owner_counts = engine._count_buf(0)[0].clone()
    by_pair = engine.begin_plan(topk_ids, rank_of=rank_of)
    assert torch.equal(engine._count_buf(0)[0], owner_counts)
    assert torch.equal(by_pair.order, by_owner.order)


@pytest.mark.parametrize(
    ("rank_of_shape", "with_owner"), [((4, 2), True), ((4,), False), ((4, 1), False)]
)
def test_p2p_engine_rejects_bad_rank_of(rank_of_shape, with_owner):
    engine = make_rank_of_engine()
    topk_ids = torch.tensor([[0, 3], [3, 1], [1, 2], [-1, -1]])
    owner = torch.tensor([1, 0, 0, 1]) if with_owner else None
    with pytest.raises(ValueError):
        engine.begin_plan(
            topk_ids, expert_owner=owner, rank_of=torch.zeros(rank_of_shape).long()
        )
