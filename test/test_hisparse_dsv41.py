"""DeepSeek-V4.1 (HiSparse L0/L1) geometry and access tests.

L0: global KV is 890 B/token instead of the legacy
``num_layers * bytes_per_token_per_layer`` (which over-estimates DSV4.1 by
~92x).  L1: source layers are grouped by compression ratio and the indexer
selects ``index_topk=512`` compressed positions per group; a ratio-2 group
emits a new compressed entry every two decode steps.
"""

import math

import pytest

from hisim.simulation.sglang.hisparse_sim import (
    DSV41_PROFILE,
    HiSparseSimConfig,
    HiSparseSimulator,
    KVProfile,
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


def test_prompt_staging_matches_bytes_per_token():
    sim = _make_sim()
    sim.on_request_start("r0", 1024)
    # ratio-2: 512 entries * 1335 B; ratio-1: 1024 entries * 222.5 B.
    assert sim.staging_bytes == int(1024 * 890.0)


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
    # ratio 1 emits an entry every step, ratio 2 every other step.
    assert sim.h2d_bytes == 0  # buffer (1025) holds both compressed prompts
    assert sim.d2h_bytes_by_ratio == {
        1: 9 * int(222.5),
        2: 5 * 1335,
    }
    assert sim.new_compressed_entries == 9 + 5

    groups = {g.ratio: g for g in sim.requests["r0"].groups}
    assert groups[1].prev_comp_len == prompt + 9
    assert groups[2].prev_comp_len == math.ceil((prompt + 9) / 2)


def test_top_k_is_selected_on_compressed_positions():
    sim = _make_sim()
    prompt = 4096
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)
    group = {g.ratio: g for g in sim.requests["r0"].groups}[2]
    # ratio-2 candidates: ceil(4096 / 2) = 2048; top_k is still 512.
    assert group.working.prev_seq_len == 2048
    assert group.working.prev_n == 512
    assert sim.config.top_k == 512


def test_cadence_does_not_churn_on_non_advance_steps():
    # churn=1.0 would force misses every call; the ratio-2 group must not be
    # stepped on odd tokens, so its hit count stays flat there.
    sim = _make_sim(_dsv41_config(churn=1.0), physical_tokens=10_000)
    prompt = 513
    sim.on_request_start("r0", prompt)
    sim.on_decode_step("r0", prompt)  # ratio-2 comp_len = 257
    group = {g.ratio: g for g in sim.requests["r0"].groups}[2]
    before = group.working.prev_seq_len
    before_d2h = group.d2h_bytes
    cost = sim.on_decode_step("r0", prompt + 1)  # ratio-2 comp_len unchanged
    assert group.working.prev_seq_len == before
    assert group.d2h_bytes == before_d2h
    assert cost.misses == 0
    assert cost.h2d_bytes == 0


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
    assert sim.h2d_bytes == expected
    assert sim.h2d_bytes_by_ratio == {
        1: int(groups[1].misses * 222.5),
        2: int(groups[2].misses * 1335.0),
    }
    assert sim.misses_bytes == expected


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
