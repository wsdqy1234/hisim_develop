import math

import pytest

from hisim.simulation.sglang.hisparse_sim import (
    HiSparseSimConfig,
    HiSparseSimulator,
    StepCost,
    load_hisparse_config,
    model_info_kv_geometry,
)


def _make_sim(**kwargs):
    cfg = HiSparseSimConfig(
        enable=True,
        top_k=kwargs.pop("top_k", 64),
        device_buffer_size=kwargs.pop("device_buffer_size", 0),
        host_to_device_ratio=kwargs.pop("host_to_device_ratio", 2),
        churn=kwargs.pop("churn", 0.05),
        recent_ratio=kwargs.pop("recent_ratio", 0.5),
        access_model=kwargs.pop("access_model", "sticky"),
        seed=kwargs.pop("seed", 7),
        **kwargs,
    )
    return HiSparseSimulator(
        cfg,
        num_layers=4,
        bytes_per_token_per_layer=576.0,
        physical_pool_tokens=100_000,
        page_size=1,
        read_bandwidth=64e9,
        write_bandwidth=64e9,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = HiSparseSimConfig()
    assert cfg.top_k == 2048
    assert cfg.device_buffer_size == 2 * 2048
    assert cfg.padded_buffer_size == 2 * 2048 + 1
    assert cfg.host_to_device_ratio == 2
    assert cfg.critical_fraction == 1.0


def test_config_from_mapping_and_env():
    cfg = HiSparseSimConfig.from_mapping(
        {"enable": True, "top_k": 128, "churn": "0.1"},
        {"host_to_device_ratio": "4", "device_buffer_size": "512"},
    )
    assert cfg.enable is True
    assert cfg.top_k == 128
    assert cfg.device_buffer_size == 512
    assert cfg.host_to_device_ratio == 4
    assert cfg.churn == pytest.approx(0.1)


def test_config_clamps_invalid_values():
    cfg = HiSparseSimConfig(
        top_k=100, device_buffer_size=10, churn=2.0, recent_ratio=-1.0
    )
    assert cfg.device_buffer_size == 100
    assert cfg.churn == 1.0
    assert cfg.recent_ratio == 0.0


def test_load_hisparse_config_env_override(monkeypatch):
    monkeypatch.setenv("HISIM_HISPARSE", "1")
    monkeypatch.setenv("HISIM_HISPARSE_TOP_K", "256")
    cfg = load_hisparse_config()
    assert cfg.enable is True
    assert cfg.top_k == 256


# ---------------------------------------------------------------------------
# Geometry / capacity (mirrors sglang's HiSparse allocator)
# ---------------------------------------------------------------------------


def test_pool_geometry_matches_sglang():
    sim = _make_sim(top_k=100, device_buffer_size=200)
    # padded_buffer_size = device_buffer + page_size
    assert sim.padded_buffer_size == 201
    # logical / host pool = physical * host_to_device_ratio
    assert sim.logical_pool_tokens == 200_000
    assert sim.host_pool_tokens == 200_000
    # concurrency = physical tokens // padded buffer
    assert sim.max_reqs_by_hot_pool == 100_000 // 201
    # HBM KV bytes are the physical hot pool only
    assert sim.hbm_kv_bytes == pytest.approx(100_000 * 4 * 576.0)
    assert sim.host_kv_bytes == pytest.approx(200_000 * 4 * 576.0)


def test_model_info_geometry_mla_and_mha():
    class M:
        num_hidden_layers = 61
        kv_lora_rank = 512
        qk_rope_head_dim = 64

    layers, per_layer = model_info_kv_geometry(M(), tp_size=1, itemsize=1)
    assert layers == 61
    assert per_layer == 576

    class MHA:
        num_hidden_layers = 32
        kv_lora_rank = 0
        num_key_value_heads = 8
        head_dim = 128

    layers, per_layer = model_info_kv_geometry(MHA(), tp_size=2, itemsize=2)
    assert layers == 32
    assert per_layer == 2 * 4 * 128 * 2


# ---------------------------------------------------------------------------
# Request lifecycle / host pool
# ---------------------------------------------------------------------------


def test_request_lifecycle_and_staging_bytes():
    sim = _make_sim()
    sim.on_request_start("r1", 1024)
    assert sim.cur_host_tokens == 1024
    assert sim.staging_bytes == 1024 * 4 * 576
    assert sim.peak_host_tokens == 1024

    for step in range(10):
        sim.on_decode_step("r1", 1024 + step)
    # one extra host token per decode step
    assert sim.cur_host_tokens == 1034
    assert sim.decode_steps == 10

    sim.on_request_finish("r1")
    assert sim.cur_host_tokens == 0
    assert sim.finished_requests == 1
    assert "r1" not in sim.requests


def test_duplicate_start_and_finish_are_idempotent():
    sim = _make_sim()
    sim.on_request_start("r1", 10)
    sim.on_request_start("r1", 10)
    assert len(sim.requests) == 1
    sim.on_request_finish("r1")
    sim.on_request_finish("r1")
    assert sim.finished_requests == 1


# ---------------------------------------------------------------------------
# Access model / hit rate
# ---------------------------------------------------------------------------


def _run_steps(sim, rid, prompt_len, steps):
    sim.on_request_start(rid, prompt_len)
    for step in range(steps):
        sim.on_decode_step(rid, prompt_len + step)
    return sim.hits, sim.misses


def test_sticky_model_has_high_hit_rate():
    sim = _make_sim(top_k=64, device_buffer_size=128, churn=0.05)
    hits, misses = _run_steps(sim, "r1", 4096, 200)
    rate = hits / (hits + misses)
    # 5% churn over a 2x oversized buffer: the great majority must hit.
    assert rate > 0.85, rate


def test_uniform_model_has_low_hit_rate():
    sim = _make_sim(
        top_k=64, device_buffer_size=128, access_model="uniform", seed=11
    )
    hits, misses = _run_steps(sim, "r1", 4096, 200)
    rate = hits / (hits + misses)
    assert rate < 0.15, rate


def test_recency_model_always_hits_for_short_sequences():
    sim = _make_sim(
        top_k=64, device_buffer_size=128, access_model="recency"
    )
    hits, misses = _run_steps(sim, "r1", 100, 50)
    assert misses == 0
    assert hits > 0


def test_churn_increases_misses():
    low = _make_sim(top_k=64, device_buffer_size=128, churn=0.01, seed=3)
    high = _make_sim(top_k=64, device_buffer_size=128, churn=0.5, seed=3)
    _run_steps(low, "r1", 4096, 200)
    _run_steps(high, "r1", 4096, 200)
    assert high.misses > low.misses * 2


def test_seed_makes_the_model_deterministic():
    a = _make_sim(seed=42, churn=0.2)
    b = _make_sim(seed=42, churn=0.2)
    _run_steps(a, "r1", 2048, 50)
    _run_steps(b, "r1", 2048, 50)
    assert (a.hits, a.misses) == (b.hits, b.misses)


# ---------------------------------------------------------------------------
# Traffic / timing
# ---------------------------------------------------------------------------


def test_bytes_and_latency_accounting():
    sim = _make_sim(top_k=64, device_buffer_size=128, churn=1.0)
    sim.on_request_start("r1", 512)
    costs = [sim.on_decode_step("r1", 512 + i) for i in range(20)]

    total_misses = sum(c.misses for c in costs)
    assert total_misses > 0
    assert sim.h2d_bytes == total_misses * 576
    assert all(c.load_dur > 0 for c in costs if c.misses > 0)
    # eager backup of one token per decode step after the first
    assert sim.d2h_bytes == 19 * 4 * 576


def test_shared_index_prefetch_scales_critical_latency():
    base = _make_sim(top_k=64, device_buffer_size=128, churn=0.2, seed=5)
    shared = _make_sim(
        top_k=64,
        device_buffer_size=128,
        churn=0.2,
        shared_index_freq=4,
        seed=5,
    )
    base.on_request_start("r1", 4096)
    shared.on_request_start("r1", 4096)
    b = base.on_decode_batch([("r1", 4096)])
    s = shared.on_decode_batch([("r1", 4096)])
    # Same traffic, one quarter of it on the critical path.
    assert s.h2d_bytes == b.h2d_bytes
    assert s.critical_h2d_bytes == int(s.h2d_bytes / 4)
    assert s.load_dur < b.load_dur


def test_decode_batch_aggregates_requests():
    sim = _make_sim(churn=0.2)
    for rid in ("a", "b", "c"):
        sim.on_request_start(rid, 1024)
    cost = sim.on_decode_batch([("a", 1024), ("b", 2048), ("c", 3000)])
    assert isinstance(cost, StepCost)
    assert sim.decode_steps == 3
    assert sim.decode_batches == 1
    assert cost.h2d_bytes == sum(
        sim.requests[rid].h2d_bytes for rid in ("a", "b", "c")
    )


def test_decode_batch_charges_launch_overhead_once():
    # One swap-in kernel serves the whole batch: t0 is paid once, not per
    # request, and the copies share the host<->device bandwidth.
    sim = _make_sim(top_k=64, device_buffer_size=128, churn=1.0)
    rids = ("a", "b", "c", "d")
    for rid in rids:
        sim.on_request_start(rid, 512)
    cost = sim.on_decode_batch([(rid, 512) for rid in rids])
    assert cost.h2d_bytes > 0
    assert math.isclose(
        cost.load_dur,
        6.67e-6 + cost.h2d_bytes / (64e9 * 0.85),
        rel_tol=1e-9,
    )


def test_metrics_payload():
    sim = _make_sim()
    sim.on_request_start("r1", 100)
    for i in range(5):
        sim.on_decode_step("r1", 100 + i)
    sim.on_request_finish("r1")
    m = sim.metrics()
    assert m["hisparse_enabled"] == 1
    assert m["hisparse_requests"] == 1
    assert m["hisparse_finished_requests"] == 1
    assert m["hisparse_decode_steps"] == 5
    assert 0 <= m["hisparse_hit_rate"] <= 1
    assert m["hisparse_hbm_kv_bytes"] == int(100_000 * 4 * 576)
    assert m["hisparse_host_kv_bytes"] == int(200_000 * 4 * 576)
