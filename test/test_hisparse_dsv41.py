"""DeepSeek-V4.1 (HiSparse L0/L1) geometry and access tests.

L0: global KV is 890 B/token instead of the legacy
``num_layers * bytes_per_token_per_layer`` (which over-estimates DSV4.1 by
~92x).  L1: source layers are grouped by compression ratio and the indexer
selects ``index_topk=512`` compressed positions per group; a ratio-2 group
emits a new compressed entry every two decode steps.
"""

import math
import random
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from hisim.simulation.sglang.hisparse_sim import (
    DSV41_FP4_PROFILE,
    DSV41_PROFILE,
    DSV41_VLLM_BF16_PROFILE,
    HiSparseSimConfig,
    HiSparseSimulator,
    KVProfile,
    _WorkingSet,
    hot_pool_geometry_for_budget,
    physical_pool_tokens_for_budget,
)


def _dsv41_config(**overrides):
    kwargs = dict(
        enable=True,
        kv_profile="dsv41",
        top_k=512,
        device_buffer_size=1024,
        host_to_device_ratio=8,
        churn=0.05,
        recent_ratio=0.5,
        access_model="sticky",
        seed=7,
    )
    kwargs.update(overrides)
    return HiSparseSimConfig(**kwargs)


def _make_sim(config=None, *, physical_tokens=20_000, tp_size=1, **kwargs):
    config = config or _dsv41_config()
    return HiSparseSimulator(
        config,
        num_layers=40,
        bytes_per_token_per_layer=576.0,
        physical_pool_tokens=physical_tokens,
        page_size=1,
        read_bandwidth=64e9,
        write_bandwidth=64e9,
        tp_size=tp_size,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Profile / configuration
# ---------------------------------------------------------------------------


def test_dsv41_profile_geometry():
    assert DSV41_PROFILE.offload_bytes_per_token == 890.0
    assert DSV41_PROFILE.source_layer_ratios == (2, 2, 2, 1)
    assert DSV41_PROFILE.index_topk == 512
    # 3 x ratio 2 and 1 x ratio 1 source layers, 222.5 B/token each.
    assert DSV41_PROFILE.io_groups() == [(1, 222.5), (2, 1335.0)]


def test_dsv41_config_caps_top_k_and_defaults_buffer():
    cfg = HiSparseSimConfig(enable=True, kv_profile="dsv41", top_k=2048)
    assert cfg.top_k == 512
    # device_buffer_size 0 -> 2 * top_k; default 1024, not 4096.
    assert cfg.device_buffer_size == 1024
    assert cfg.padded_buffer_size == 1025


def test_dsv41_config_keeps_explicit_smaller_top_k():
    cfg = HiSparseSimConfig(enable=True, kv_profile="dsv41", top_k=256)
    assert cfg.top_k == 256
    assert cfg.device_buffer_size == 512


def test_profile_from_mapping_string_and_dict():
    cfg = HiSparseSimConfig.from_mapping({"kv_profile": "dsv41", "top_k": 512})
    assert cfg.kv_profile is DSV41_PROFILE
    assert cfg.top_k == 512

    cfg = HiSparseSimConfig.from_mapping(
        {
            "kv_profile": {
                "name": "custom",
                "offload_bytes_per_token": 1000.0,
                "source_layer_ratios": [4, 1],
                "index_topk": 128,
            }
        }
    )
    assert cfg.kv_profile.name == "custom"
    assert cfg.kv_profile.source_layer_ratios == (4, 1)
    assert cfg.top_k == 128


def test_profile_value_none_is_legacy():
    cfg = HiSparseSimConfig(enable=True, kv_profile=None)
    assert cfg.kv_profile is None
    assert cfg.top_k == 2048
    assert cfg.device_buffer_size == 4096


# ---------------------------------------------------------------------------
# L0: capacity / staging bytes
# ---------------------------------------------------------------------------


def test_l0_hbm_and_host_bytes_use_890_per_token():
    sim = _make_sim(physical_tokens=20_000)
    assert sim.hbm_kv_bytes == pytest.approx(20_000 * 890.0)
    assert sim.host_kv_bytes == pytest.approx(20_000 * 8 * 890.0)
    # Without the profile, the legacy geometry would report ~92x more HBM.
    legacy = HiSparseSimulator(
        HiSparseSimConfig(enable=True, top_k=512, device_buffer_size=1024),
        num_layers=40,
        # V4.1 has no kv_lora_rank, so the legacy path takes the GQA branch:
        # 2 * 1 KV head * 512 head_dim * 2 B = 2048 B/token/layer.
        bytes_per_token_per_layer=2048.0,
        physical_pool_tokens=20_000,
    )
    assert legacy.hbm_kv_bytes == pytest.approx(20_000 * 40 * 2048.0)
    assert legacy.hbm_kv_bytes / sim.hbm_kv_bytes > 90


def test_l0_128k_context_is_megabytes_not_gigabytes():
    # 128K tokens, host ratio 8: persistent KV is ~116.6 MB.
    sim = _make_sim(physical_tokens=20_000)
    logical = 128 * 1024
    assert logical * 890.0 == pytest.approx(116_654_080.0)
    # The old GQA geometry would have said ~10.7 GB.
    assert logical * 40 * 2048.0 > 10.7e9
    assert sim.physical_pool_tokens * 890.0 < 20e6


def test_bf16_physical_pool_uses_allocator_slope():
    p = DSV41_VLLM_BF16_PROFILE
    swa = 8 * p.swa_window_tokens * p.swa_bytes_per_token
    compressor = 8 * p.compressor_bytes_per_request
    tokens = physical_pool_tokens_for_budget(
        p, 80e9, max_num_reqs=8, tp_size=8
    )
    assert tokens == int((80e9 - swa - compressor) // 4096)
    assert tokens == 19_520_242
    # The old payload denominator would have produced 27,714,007 tokens.
    assert tokens < int((80e9 - swa - compressor) // 2885)


def test_hot_pool_geometry_resolves_request_clamp_and_overhead():
    p = DSV41_VLLM_BF16_PROFILE
    budget = 64 * 1024 * 1024
    tokens, reqs = hot_pool_geometry_for_budget(
        p, budget, requested_reqs=8, padded_buffer_size=1025, tp_size=8
    )
    assert reqs == 6
    assert tokens == 8128
    cfg = _dsv41_config(kv_profile="dsv41_vllm_bf16")
    sim = _make_sim(
        cfg,
        physical_tokens=tokens,
        tp_size=8,
        max_num_reqs=reqs,
    )
    assert sim.max_num_reqs == reqs
    assert sim.hbm_kv_bytes == pytest.approx(budget)


def test_requested_reqs_are_per_dp_worker():
    from hisim.simulation.sglang.sglang_hook import _requested_reqs_for_profile

    p = DSV41_VLLM_BF16_PROFILE
    dp8 = SimpleNamespace(
        max_running_requests=64,
        enable_dp_attention=True,
        dp_size=8,
    )
    assert _requested_reqs_for_profile(dp8, p, fallback=4096) == 8
    no_dp = SimpleNamespace(
        max_running_requests=64,
        enable_dp_attention=False,
        dp_size=8,
    )
    assert _requested_reqs_for_profile(no_dp, p, fallback=4096) == 64
    default = SimpleNamespace(
        max_running_requests=None,
        enable_dp_attention=True,
        dp_size=8,
    )
    assert _requested_reqs_for_profile(default, p, fallback=4096) == 8


def test_prompt_staging_matches_bytes_per_token():
    sim = _make_sim()
    sim.on_request_start("r0", 1024)
    # ratio-2: 512 entries * 1335 B; ratio-1: 1024 entries * 222.5 B.
    assert sim.staging_bytes == int(1024 * 890.0)


def test_prompt_staging_subtracts_reused_prefix():
    sim = _make_sim()
    sim.on_request_start("r0", 1024, prefix_len=512)
    assert sim.staging_bytes == int(512 * 890.0)
    assert sim.cur_host_tokens == 512
    assert sim.prefix_reused_tokens == 512
    # The full prompt still determines the initial compressed candidate count.
    groups = {g.ratio: g for g in sim.requests["r0"].groups}
    assert groups[1].working.prev_seq_len == 0
    sim.on_decode_step("r0", 1024)
    assert groups[1].working.prev_seq_len == 1024
    assert groups[2].working.prev_seq_len == 512


def test_odd_prompt_staging_uses_floor_compression():
    profile = KVProfile(
        name="ratio2-floor",
        offload_bytes_per_token=10.0,
        source_layer_ratios=(2,),
        index_topk=4,
    )
    cfg = _dsv41_config(
        kv_profile=profile,
        top_k=4,
        device_buffer_size=8,
    )
    sim = _make_sim(cfg, physical_tokens=10_000)
    sim.on_request_start("r0", 65)
    assert sim.staging_bytes == 32 * 20
    assert sim.on_decode_step("r0", 65).d2h_bytes == 0
    # Entry 32 completes now and is backed up exactly once.
    assert sim.on_decode_step("r0", 66).d2h_bytes == 20
    assert sim.staging_bytes + sim.d2h_bytes == 66 * 10

    # The hook can report past_len=P+1 on the first decode after prefill.  That
    # completed entry must be backed up immediately instead of being mistaken
    # for part of the staged prompt.
    hook_like = _make_sim(cfg, physical_tokens=10_000)
    hook_like.on_request_start("r2", 65)
    assert hook_like.on_decode_step("r2", 66).d2h_bytes == 20
    assert hook_like.staging_bytes + hook_like.d2h_bytes == 66 * 10

    # A ratio-2 group below one full entry has no candidate at all; it must not
    # invent position -1 as a newest page.
    tiny = _make_sim(cfg, physical_tokens=10_000)
    tiny.on_request_start("r1", 1)
    assert tiny.staging_bytes == 0
    assert tiny.on_decode_step("r1", 1).h2d_bytes == 0
    assert tiny.on_decode_step("r1", 2).d2h_bytes == 20
    assert -1 not in tiny.requests["r1"].groups[0].working.hot


def test_hook_request_start_uses_resolved_prefix(monkeypatch):
    from hisim.simulation.sglang.sglang_hook import (
        C_SchedulerHook,
        _start_hisparse_request,
    )

    sim = _make_sim()
    monkeypatch.setattr(HiSparseSimulator, "_instance", sim)
    monkeypatch.setitem(C_SchedulerHook.REQUEST_STATS, "prefix-r", None)
    req = SimpleNamespace(
        rid="prefix-r",
        origin_input_ids=list(range(1024)),
        # Req.prefix_indices is a Tensor in sglang; truthiness must not be used.
        prefix_indices=torch.arange(512, dtype=torch.int64),
    )
    _start_hisparse_request(req)
    assert sim.staging_bytes == int(512 * 890.0)
    assert sim.cur_host_tokens == 512
    assert sim.prefix_reused_tokens == 512

    empty = _make_sim()
    monkeypatch.setattr(HiSparseSimulator, "_instance", empty)
    monkeypatch.setitem(C_SchedulerHook.REQUEST_STATS, "empty-r", None)
    empty_req = SimpleNamespace(
        rid="empty-r",
        origin_input_ids=list(range(1024)),
        prefix_indices=torch.empty(0, dtype=torch.int64),
    )
    _start_hisparse_request(empty_req)
    assert empty.staging_bytes == int(1024 * 890.0)
    assert empty.prefix_reused_tokens == 0


def test_tp_scaling_divides_profile_bytes():
    sim = _make_sim(tp_size=2)
    assert sim.offload_bytes_per_token == pytest.approx(445.0)
    assert sim.io_groups == [(1, pytest.approx(111.25)), (2, pytest.approx(667.5))]
    assert sim.hbm_kv_bytes == pytest.approx(20_000 * 445.0)
    assert sim.host_kv_bytes == pytest.approx(20_000 * 8 * 445.0)


def test_l2_indexer_overhead_is_optional():
    profile = KVProfile(
        name="dsv41-indexer",
        offload_bytes_per_token=890.0,
        source_layer_ratios=(2, 2, 2, 1),
        index_topk=512,
        indexer_bytes_per_token=544.0,
    )
    sim = _make_sim(_dsv41_config(kv_profile=profile))
    assert sim.indexer_bytes_per_token == pytest.approx(544.0)
    assert sim.hbm_kv_bytes == pytest.approx(20_000 * (890.0 + 544.0))
    # Host only holds the global KV, not the HBM-resident indexer.
    assert sim.host_kv_bytes == pytest.approx(20_000 * 8 * 890.0)

    default = _make_sim()
    assert default.indexer_bytes_per_token == 0.0
    assert default.hbm_kv_bytes == pytest.approx(20_000 * 890.0)


# ---------------------------------------------------------------------------
# L1: compressed positions and cadence
# ---------------------------------------------------------------------------


def test_compressed_working_set_cadence():
    sim = _make_sim()
    prompt = 64
    sim.on_request_start("r0", prompt)
    for step in range(10):
        sim.on_decode_step("r0", prompt + step)

    # First decode step seeds both groups; the next 9 steps are counted.
    # Compression uses floor(past_len / ratio): ratio 1 emits every step,
    # ratio 2 emits at 66, 68, 70 and 72.  The whole sequence is smaller than
    # the device buffer, so the direct-short path has no main/index H2D and the
    # previous newest page ages into the regular LRU for free.
    assert sim.h2d_bytes == 0
    assert sim.main_h2d_bytes == 0
    assert sim.d2h_bytes_by_ratio == {
        1: 9 * int(222.5),
        2: 4 * 1335,
    }
    assert sim.new_compressed_entries == 9 + 4

    groups = {g.ratio: g for g in sim.requests["r0"].groups}
    assert groups[1].prev_comp_len == prompt + 9
    assert groups[2].prev_comp_len == (prompt + 9) // 2


def test_top_k_is_selected_on_compressed_positions():
    sim = _make_sim()
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    group = {g.ratio: g for g in sim.requests["r0"].groups}[2]
    # ratio-2 candidates: floor(4096 / 2) = 2048; top_k is still 512.
    assert group.working.prev_seq_len == 2048
    assert group.working.prev_n == 512
    assert sim.config.top_k == 512


def test_selection_can_change_when_compression_len_is_unchanged():
    cfg = _dsv41_config(
        churn=1.0,
        top_k=64,
        device_buffer_size=64,
        selection_cadence="every_step",
    )
    sim = _make_sim(cfg, physical_tokens=10_000)
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    group = {g.ratio: g for g in sim.requests["r0"].groups}[2]
    before = group.working.prev_seq_len
    before_d2h = group.d2h_bytes
    cost = sim.on_decode_step("r0", prompt + 1)  # floor(4097/2) unchanged
    assert group.working.prev_seq_len == before
    assert group.d2h_bytes == before_d2h
    # The query changed even though no compressed entry was produced.
    assert cost.misses > 0
    assert cost.h2d_bytes > 0


def test_compression_boundary_cadence_is_explicit_sensitivity_mode():
    cfg = _dsv41_config(
        churn=1.0,
        top_k=64,
        device_buffer_size=64,
        selection_cadence="compression_boundary",
    )
    sim = _make_sim(cfg, physical_tokens=10_000)
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    group = {g.ratio: g for g in sim.requests["r0"].groups}[2]
    before = group.working.prev_seq_len
    before_misses = group.misses
    sim.on_decode_step("r0", prompt + 1)
    assert group.working.prev_seq_len == before
    assert group.misses == before_misses


def test_h2d_bytes_are_charged_per_compressed_entry():
    cfg = _dsv41_config(
        churn=1.0,
        top_k=64,
        device_buffer_size=64,
        host_to_device_ratio=8,
    )
    sim = _make_sim(cfg, physical_tokens=10_000)
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    groups = {g.ratio: g for g in sim.requests["r0"].groups}
    assert groups[1].misses > 0 and groups[2].misses > 0
    expected = sum(int(g.misses * g.entry_bytes) for g in groups.values())
    expected_bytes = sum(g.misses * g.entry_bytes for g in groups.values())
    assert sim.h2d_bytes == expected
    assert sim.h2d_bytes_by_ratio == {
        1: int(groups[1].misses * 222.5),
        2: int(groups[2].misses * 1335.0),
    }
    assert sim.misses_bytes == pytest.approx(expected_bytes)


def test_hit_rate_is_byte_weighted():
    sim = _make_sim()
    sim.on_request_start("r0", 1024)
    for step in range(8):
        sim.on_decode_step("r0", 1024 + step)
    metrics = sim.metrics()
    assert metrics["hisparse_hit_rate_bytes"] == pytest.approx(
        metrics["hisparse_hits_bytes"]
        / (metrics["hisparse_hits_bytes"] + metrics["hisparse_misses_bytes"])
    )
    assert metrics["hisparse_io_groups"] == [[1, 222.5], [2, 1335.0]]
    assert metrics["hisparse_kv_profile"] == "dsv41"
    assert metrics["hisparse_profile_num_layers"] == 40


def test_churn_one_does_not_reselect_dropped_positions():
    ws = _WorkingSet(
        top_k=64,
        capacity=64,
        access_model="sticky",
        churn=1.0,
        recent_ratio=0.5,
        rng=random.Random(7),
        hot_buffer_init="earliest",
        lru_mode=True,
        newest_page=True,
    )
    ws.first_step(4096)
    before = set(ws.sticky)
    ws.step(4097)
    after = set(ws.sticky)
    assert before and after
    assert before.isdisjoint(after)


def test_hot_buffer_seed_modes_and_newest_page():
    def seeded(mode):
        ws = _WorkingSet(
            top_k=4,
            capacity=4,
            access_model="sticky",
            churn=0.0,
            recent_ratio=0.5,
            rng=random.Random(1),
            hot_buffer_init=mode,
            lru_mode=True,
            newest_page=True,
        )
        ws._seed_hot_buffer(10)
        return list(ws.hot)

    assert seeded("earliest") == [0, 1, 2, 3]
    assert seeded("latest") == [6, 7, 8, 9]
    assert seeded("empty") == []
    # The newest entry is served by its reserved page, not the regular LRU.
    ws = _WorkingSet(
        top_k=2,
        capacity=4,
        access_model="recency",
        churn=0.0,
        recent_ratio=1.0,
        rng=random.Random(1),
        hot_buffer_init="empty",
        lru_mode=True,
        newest_page=True,
    )
    hits, misses = ws.first_step(10)
    assert hits == 1
    assert misses == 1
    # Position 8 was loaded; position 9 is represented by the newest page.
    assert list(ws.hot) == [8]


def test_direct_short_has_no_swap_in_and_ages_newest_for_free():
    ws = _WorkingSet(
        top_k=2,
        capacity=4,
        access_model="recency",
        churn=0.0,
        recent_ratio=1.0,
        rng=random.Random(0),
        hot_buffer_init="earliest",
        lru_mode=True,
        newest_page=True,
    )
    for seq_len in (2, 3, 4):
        assert ws.step(seq_len) == (2, 0)
    assert ws.prev_newest == 3


def test_lru_hit_moves_to_mru():
    ws = _WorkingSet(
        top_k=4,
        capacity=4,
        access_model="sticky",
        churn=0.0,
        recent_ratio=0.5,
        rng=random.Random(1),
        hot_buffer_init="empty",
        lru_mode=True,
        newest_page=False,
    )
    ws.hot = OrderedDict((pos, None) for pos in (0, 1, 2, 3))
    hits, misses = ws._access_selected({0}, seq_len=10)
    assert (hits, misses) == (1, 0)
    assert list(ws.hot) == [1, 2, 3, 0]


def test_lru_misses_are_inserted_before_hit_block():
    ws = _WorkingSet(
        top_k=2,
        capacity=4,
        access_model="sticky",
        churn=0.0,
        recent_ratio=0.0,
        rng=random.Random(1),
        hot_buffer_init="empty",
        lru_mode=True,
        newest_page=False,
    )
    ws.hot = OrderedDict((pos, None) for pos in (0, 1, 2, 3))
    hits, misses = ws._access_selected({0, 4}, seq_len=10)
    assert (hits, misses) == (1, 1)
    # Miss 4 sits immediately below hit 0, not after it in token order.
    assert list(ws.hot) == [2, 3, 4, 0]


def test_recency_selects_most_recent_top_k():
    ws = _WorkingSet(
        top_k=8,
        capacity=16,
        access_model="recency",
        churn=0.0,
        recent_ratio=0.5,
        rng=random.Random(1),
        hot_buffer_init="latest",
        lru_mode=True,
        newest_page=True,
    )
    for seq_len in (100, 101):
        hits, misses = ws.step(seq_len)
        assert hits + misses == 8


def test_recent_ratio_zero_has_no_implicit_recent_entry():
    ws = _WorkingSet(
        top_k=8,
        capacity=8,
        access_model="sticky",
        churn=0.0,
        recent_ratio=0.0,
        rng=random.Random(1),
    )
    assert ws._recent_count(8) == 0


# ---------------------------------------------------------------------------
# L2: .22 vLLM-Ascend BF16 measured layout
# ---------------------------------------------------------------------------


def test_vllm_bf16_profile_geometry():
    profile = DSV41_VLLM_BF16_PROFILE
    assert profile.offload_bytes_per_token == 2885.0
    assert profile.bytes_scope == "per_rank"
    assert profile.source_layer_ids == (2, 8, 14, 20)
    assert profile.source_layer_bytes_per_token == (577.0, 577.0, 577.0, 1154.0)
    assert profile.source_layer_bytes_sum == 2885.0
    assert sum(profile.source_layer_main_kv_bytes_per_token) == 2560.0
    assert sum(profile.source_layer_index_bytes_per_token) == 325.0
    # ratio-2 group: 3 * 577 * 2; ratio-1 group: 1154.
    assert profile.io_groups() == [(1, 1154.0), (2, 3462.0)]
    assert profile.io_channel_groups() == [
        (1, 1024.0, 130.0),
        (2, 3072.0, 390.0),
    ]
    assert profile.main_kv_bytes_per_token == 2560.0
    assert profile.index_bytes_per_token == 325.0
    # The index is embedded in 2885, so there is no separate HBM indexer.
    assert profile.indexer_bytes_per_token == 0.0
    # Capacity uses the allocator slope, not the 2885 payload.
    assert profile.capacity_bytes_per_token == 4096.0
    assert profile.scheduler_block_tokens == 128
    assert profile.page_bytes == 131072
    assert profile.slot_count == 4
    assert profile.allocator_bytes_per_token == 4096.0
    assert profile.payload_bytes_per_block == 369280.0


def test_vllm_bf16_source_to_consumer_topology():
    profile = DSV41_VLLM_BF16_PROFILE
    consumers = dict(profile.source_consumers)
    assert consumers[2] == (2, 3, 4, 5, 6, 7)
    assert consumers[8] == (8, 9, 10, 11, 12, 13)
    assert consumers[14] == (14, 15, 16, 17, 18, 19)
    assert consumers[20] == tuple(range(20, 40))
    assert profile.index_storage_layers == (2, 8, 14, 20)
    assert profile.index_compute_layers == (2, 8, 14, 20, 24, 28, 32, 36)
    assert profile.candidate_blocks == 2048
    assert profile.candidate_block_tokens == 8
    assert profile.candidate_source_layer == 20
    assert profile.candidate_filter_layers == (24, 28, 32, 36)


@pytest.mark.parametrize(
    "tokens,blocks,byte_count",
    [
        (128, 22, 22 * 524288),
        (8192, 405, 405 * 524288),
        (1048576, 8533, 4473749504),
    ],
)
def test_vllm_bf16_admission_formula(tokens, blocks, byte_count):
    profile = DSV41_VLLM_BF16_PROFILE
    assert profile.admission_blocks(tokens) == blocks
    assert profile.admission_bytes(tokens) == byte_count
    # The steady-state overhead (10 SWA groups + compressor + null block)
    # is independent of context once L >= swa_admission_tokens.
    expected_overhead = (blocks - math.ceil(tokens / 128)) * 524288
    assert profile.admission_overhead_bytes(tokens) == expected_overhead


def test_vllm_bf16_admission_overhead_is_341_blocks_at_4223():
    profile = DSV41_VLLM_BF16_PROFILE
    assert profile.swa_admission_tokens == 128 - 1 + 2 * 2048
    assert profile.admission_blocks(4223) == 374
    assert profile.admission_overhead_bytes(4223) == 341 * 524288


@pytest.mark.parametrize("tp_size", [1, 8])
def test_vllm_bf16_bytes_are_per_rank_not_divided_by_tp(tp_size):
    cfg = _dsv41_config(kv_profile="dsv41_vllm_bf16")
    sim = _make_sim(
        cfg,
        physical_tokens=20_000,
        tp_size=tp_size,
        max_num_reqs=8,
    )
    assert sim.bytes_scope == "per_rank"
    assert sim.offload_bytes_per_token == pytest.approx(2885.0)
    assert sim.indexer_bytes_per_token == 0.0
    assert sim.io_groups == [(1, pytest.approx(1154.0)), (2, pytest.approx(3462.0))]

    # Steady SWA ring and compressor state are separate from the long-KV
    # payload and are also per rank.
    assert sim.hbm_swa_bytes == pytest.approx(8 * 128 * 40960.0)
    assert sim.hbm_compressor_bytes == pytest.approx(8 * 393216.0)
    expected_hbm = (
        20_000 * 4096.0
        + 8 * 128 * 40960.0
        + 8 * 393216.0
    )
    assert sim.hbm_kv_bytes == pytest.approx(expected_hbm)
    # Host stores the 2885 payload, not the allocator footprint.
    assert sim.host_kv_bytes == pytest.approx(20_000 * 8 * 2885.0)

    sim.on_request_start("r0", 1024)
    assert sim.staging_bytes == int(1024 * 2885.0)


def test_vllm_bf16_io_is_charged_per_source_layer_group():
    cfg = _dsv41_config(
        kv_profile="dsv41_vllm_bf16",
        churn=1.0,
        top_k=64,
        device_buffer_size=64,
        host_to_device_ratio=8,
    )
    sim = _make_sim(cfg, physical_tokens=10_000, tp_size=8, max_num_reqs=8)
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    groups = {g.ratio: g for g in sim.requests["r0"].groups}
    assert groups[1].entry_bytes == pytest.approx(1154.0)
    assert groups[2].entry_bytes == pytest.approx(3462.0)
    assert groups[1].main_entry_bytes == pytest.approx(1024.0)
    assert groups[1].index_entry_bytes == pytest.approx(130.0)
    assert groups[2].main_entry_bytes == pytest.approx(3072.0)
    assert groups[2].index_entry_bytes == pytest.approx(390.0)
    assert groups[1].misses > 0 and groups[2].misses > 0
    expected_main = sum(int(g.misses * g.main_entry_bytes) for g in groups.values())
    expected_index = sum(
        int((g.hits + g.misses) * g.index_entry_bytes) for g in groups.values()
    )
    assert sim.h2d_bytes == expected_main + expected_index
    assert sim.main_h2d_bytes == expected_main
    assert sim.index_h2d_bytes == expected_index
    assert sim.index_scan_bytes == expected_index
    assert sim.h2d_bytes_by_ratio == {
        1: int(
            groups[1].misses * groups[1].main_entry_bytes
            + (groups[1].hits + groups[1].misses) * groups[1].index_entry_bytes
        ),
        2: int(
            groups[2].misses * groups[2].main_entry_bytes
            + (groups[2].hits + groups[2].misses) * groups[2].index_entry_bytes
        ),
    }
    assert sim.staging_bytes == int(prompt * 2885.0)

    metrics = sim.metrics()
    assert metrics["hisparse_runtime_target"] == "vllm_ascend"
    assert metrics["hisparse_capacity_bytes_per_token"] == 4096.0
    assert metrics["hisparse_io_channel_groups"] == [
        [1, 1024.0, 130.0],
        [2, 3072.0, 390.0],
    ]
    assert metrics["hisparse_main_kv_h2d_bytes"] == expected_main
    assert metrics["hisparse_index_h2d_bytes"] == expected_index
    assert metrics["hisparse_kv_profile"] == "dsv41_vllm_bf16"
    assert metrics["hisparse_bytes_scope"] == "per_rank"
    assert metrics["hisparse_source_layer_bytes_per_token"] == [
        577.0,
        577.0,
        577.0,
        1154.0,
    ]
    assert metrics["hisparse_source_layer_bytes_sum"] == 2885.0
    assert metrics["hisparse_admission_128_blocks"] == 22
    assert metrics["hisparse_admission_8192_blocks"] == 405
    assert metrics["hisparse_admission_1m_blocks"] == 8533
    assert metrics["hisparse_admission_1m_bytes"] == 4473749504
    assert metrics["hisparse_hbm_admission_overhead_bytes"] == 8 * 341 * 524288
    assert metrics["hisparse_io_groups"] == [[1, 1154.0], [2, 3462.0]]


# ---------------------------------------------------------------------------
# L2: FP4/CSA2 topology reuse (bytes still approximate)
# ---------------------------------------------------------------------------


def test_fp4_csa2_profile_uses_known_topology_and_indexer_assumption():
    profile = DSV41_FP4_PROFILE
    assert profile.name == "dsv41_fp4_csa2"
    assert profile.bytes_scope == "global"
    assert profile.offload_bytes_per_token == 890.0
    assert profile.indexer_bytes_per_token == 544.0
    assert profile.io_groups() == [(1, 222.5), (2, 1335.0)]
    assert profile.source_layer_ids == (2, 8, 14, 20)
    assert profile.source_consumers == DSV41_VLLM_BF16_PROFILE.source_consumers
    assert profile.index_storage_layers == (2, 8, 14, 20)
    assert profile.index_compute_layers == (2, 8, 14, 20, 24, 28, 32, 36)
    assert profile.candidate_blocks == 2048
    assert profile.candidate_block_tokens == 8
    assert profile.candidate_source_layer == 20
    assert profile.swa_layers == 40
    assert profile.swa_bytes_per_token == 0.0  # FP4 SWA bytes not measured yet
    assert profile.compressor_layers == (2, 8, 14)
    assert profile.compressor_bytes_per_request == 0.0
    assert profile.admission_bytes(1_048_576) == 0  # allocator layout unknown


def test_fp4_csa2_global_bytes_scale_with_tp():
    cfg = _dsv41_config(kv_profile="dsv41_fp4_csa2")
    sim = _make_sim(cfg, physical_tokens=20_000, tp_size=2)
    assert sim.bytes_scope == "global"
    assert sim.offload_bytes_per_token == pytest.approx(445.0)
    assert sim.indexer_bytes_per_token == pytest.approx(272.0)
    assert sim.io_groups == [
        (1, pytest.approx(111.25)),
        (2, pytest.approx(667.5)),
    ]
    assert sim.hbm_kv_bytes == pytest.approx(20_000 * (445.0 + 272.0))
    assert sim.host_kv_bytes == pytest.approx(20_000 * 8 * 445.0)
    assert sim.hbm_swa_bytes == 0.0
    assert sim.hbm_compressor_bytes == 0.0

    metrics = sim.metrics()
    assert metrics["hisparse_kv_profile"] == "dsv41_fp4_csa2"
    assert metrics["hisparse_bytes_scope"] == "global"
    assert metrics["hisparse_source_layer_ids"] == [2, 8, 14, 20]
    assert metrics["hisparse_indexer_bytes_per_token"] == pytest.approx(272.0)
    assert metrics["hisparse_admission_1m_bytes"] == 0


def test_new_profile_string_aliases():
    assert (
        HiSparseSimConfig.from_mapping({"kv_profile": "dsv41_vllm_bf16"}).kv_profile
        is DSV41_VLLM_BF16_PROFILE
    )
    assert (
        HiSparseSimConfig.from_mapping({"kv_profile": "dsv41_fp4_csa2"}).kv_profile
        is DSV41_FP4_PROFILE
    )


# ---------------------------------------------------------------------------
# Legacy regression (no profile == pre-DSV4.1 behaviour)
# ---------------------------------------------------------------------------


def test_legacy_geometry_and_layers_unchanged():
    sim = HiSparseSimulator(
        HiSparseSimConfig(
            enable=True,
            top_k=64,
            device_buffer_size=128,
            host_to_device_ratio=2,
            churn=0.05,
            recent_ratio=0.5,
            access_model="sticky",
            seed=7,
        ),
        num_layers=4,
        bytes_per_token_per_layer=576.0,
        physical_pool_tokens=100_000,
    )
    assert sim.hbm_kv_bytes == pytest.approx(100_000 * 4 * 576.0)
    assert sim.host_kv_bytes == pytest.approx(200_000 * 4 * 576.0)
    sim.on_request_start("r0", 32)
    cost = sim.on_decode_step("r0", 32)
    assert cost.hits + cost.misses == 32 * 4
    metrics = sim.metrics()
    assert metrics["hisparse_kv_profile"] == "legacy"
    assert metrics["hisparse_index_topk"] == 0
    assert metrics["hisparse_io_groups"] == []
