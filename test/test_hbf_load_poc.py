"""Standalone PoC: HBF Load timing injection independent of L3.

Tests the mechanism by which HBF Load timing is computed from host_hit_length
metadata (not from mock_start_loading) and consumed by the simulation clock.

This proves HBF→HBM restore latency can be modeled WITHOUT:
- Real tensor copy (load_queue=0 in simulation)
- L3 storage (no storage backend needed)
- mock_start_loading (never called in sim mode)

The mechanism:
1. host_hit_length (from prefix match) tells how many tokens are in L2
2. Adapter's capacity stats determine HBF/DRAM split
3. _hbf_transfer_duration computes load timing (fast formula: max(latency, bw))
4. StateManager records HBF/DRAM load duration (max, not sum — parallel)
5. wrapped_process_batch_result pops durations, combined_load = max(l2, hbf)
6. Clock advancement: step(max(combined_load - last_inference, 0)) + inference
   → if load > inference, overflow delays next round → affects TPOT

Run: cd /home/d00954154/HiSim_Qwen3 && PYTHONPATH=src python3 test/test_hbf_load_poc.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from hisim.simulation.sglang.hbf_tiered_pool import (
    TieredHostKVCache, MockHBFPool, PlacementPolicyV0,
)
from hisim.simulation.manager.state import StateManager
from hisim.simulation.types import HBFConfig, PlatformConfig
from hisim.simulation.manager import ConfigManager


def setup_mock_config(hbf_read_latency_us=10.0, hbf_write_latency_us=100.0,
                      hbf_read_bw_gb=512.0):
    """Set up ConfigManager with HBF config for timing functions."""
    hbf = HBFConfig(
        capacity_gb=64,
        read_bandwidth_gb=hbf_read_bw_gb,
        write_bandwidth_gb=100,
        read_latency_us=hbf_read_latency_us,
        write_latency_us=hbf_write_latency_us,
    )
    pc = PlatformConfig(
        device="H20",
        enable_hbf=True,
        hbf=hbf,
        memory_read_bandwidth_gb=16,
        memory_write_bandwidth_gb=16,
    )
    ConfigManager._platform_config = pc
    # Also set kv_cache_bytes for bytes_per_token
    ConfigManager._kv_cache_bytes = 32768  # ~32KB per token (Qwen3-0.6B)
    return pc


def compute_host_load_timing(host_hit, adapter, bytes_per_token=32768):
    """Simulate the load timing injection (from sglang_hook line ~2397).

    This is the code that runs when host_hit_length > 0 during prefix match.
    Computes HBF/DRAM split from adapter capacity stats and records timing.
    """
    from hisim.simulation.sglang.sglang_hook import (
        _record_hbf_load_timing, _record_l2_load_timing,
    )
    if adapter is None:
        _record_l2_load_timing(host_hit * bytes_per_token, host_hit)
        return {"hbf_hit": 0, "dram_hit": host_hit}

    stats = adapter.capacity_stats()
    hbf_used = stats["hbf_used"]
    dram_used = stats["dram_used"]
    total_used = hbf_used + dram_used
    if total_used == 0:
        return {"hbf_hit": 0, "dram_hit": 0}

    hbf_ratio = hbf_used / total_used
    hbf_hit = int(host_hit * hbf_ratio)
    dram_hit = host_hit - hbf_hit

    if hbf_hit > 0:
        _record_hbf_load_timing(hbf_hit * bytes_per_token)
    if dram_hit > 0:
        _record_l2_load_timing(dram_hit * bytes_per_token, dram_hit)

    return {"hbf_hit": hbf_hit, "dram_hit": dram_hit}


def simulate_clock_advancement(combined_load_dur, inference_dur, last_inference_dur):
    """Simulate the OVERLAP_SCHEDULE clock advancement (sglang_hook line ~2562).

    step_global_clock(max(combined_load - last_inference, 0))  # load overflow
    step_global_clock(inference)                                # inference
    """
    load_overflow = max(combined_load_dur - last_inference_dur, 0)
    StateManager.step_global_clock(load_overflow)
    StateManager.step_global_clock(inference_dur)
    return load_overflow


def test_hbf_load_timing_normal():
    """Test 1: Normal latency (10μs) — HBF load timing recorded correctly."""
    print("\n--- Test 1: Normal latency (10μs) ---")
    StateManager.reset()
    setup_mock_config(hbf_read_latency_us=10.0)
    StateManager.set_current_inference_dur(0.001)  # 1ms inference
    StateManager._last_inference_dur = 0.001  # last inference also 1ms

    # Create adapter with HBF(100) + DRAM(50), backup 40 tokens to HBF, 10 to DRAM
    hbf_pool = MockHBFPool(logical_size=100)
    dram_pool = MockHBFPool(logical_size=50)
    adapter = TieredHostKVCache(hbf_pool, dram_pool, PlacementPolicyV0(hbf_ratio=0.8))
    adapter.alloc(40)  # 40 tokens to HBF
    adapter.alloc(10)  # 10 tokens to DRAM (via fallback or ratio)

    # Simulate: 20 tokens need load-back from L2
    host_hit = 20
    result = compute_host_load_timing(host_hit, adapter)
    print(f"  host_hit={host_hit} → HBF={result['hbf_hit']}, DRAM={result['dram_hit']}")

    # Pop durations
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    print(f"  HBF load dur={hbf_load*1e6:.1f}μs, DRAM load dur={l2_load*1e6:.1f}μs, combined(max)={combined*1e6:.1f}μs")

    # Verify timing recorded
    assert hbf_load > 0, "HBF load timing should be > 0"
    assert l2_load > 0, "DRAM load timing should be > 0"
    # HBF: max(10μs, hbf_hit * 32768 / 512GB/s)
    expected_hbf = max(10e-6, result["hbf_hit"] * 32768 / 512e9)
    assert abs(hbf_load - expected_hbf) < 1e-12, f"HBF: {hbf_load} vs {expected_hbf}"

    # Clock advancement
    load_overflow = simulate_clock_advancement(combined, 0.001, 0.001)
    print(f"  load_overflow={load_overflow*1e6:.1f}μs (combined={combined*1e6:.1f} vs inference=1000μs)")
    # With 10μs load and 1ms inference, load < inference → overflow=0 (overlapped)
    assert load_overflow == 0, "10μs load < 1ms inference → no overflow"
    print("  ✅ Normal latency: load overlapped with inference, no TPOT impact")


def test_hbf_load_timing_high_latency():
    """Test 2: High latency (10ms) — load overflow delays clock → TPOT impact."""
    print("\n--- Test 2: High latency (10ms) ---")
    StateManager.reset()
    setup_mock_config(hbf_read_latency_us=10000.0)  # 10ms!
    StateManager.set_current_inference_dur(0.001)  # 1ms inference
    StateManager._last_inference_dur = 0.001

    hbf_pool = MockHBFPool(logical_size=100)
    dram_pool = MockHBFPool(logical_size=50)
    adapter = TieredHostKVCache(hbf_pool, dram_pool, PlacementPolicyV0(hbf_ratio=0.8))
    adapter.alloc(40)
    adapter.alloc(10)

    host_hit = 20
    result = compute_host_load_timing(host_hit, adapter)
    print(f"  host_hit={host_hit} → HBF={result['hbf_hit']}, DRAM={result['dram_hit']}")

    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    print(f"  HBF load={hbf_load*1e3:.2f}ms, DRAM load={l2_load*1e3:.2f}ms, combined={combined*1e3:.2f}ms")

    # With 10ms latency, HBF load = max(10ms, bytes/bw) = 10ms (latency dominates)
    assert abs(hbf_load - 0.01) < 1e-9, f"HBF load should be 10ms, got {hbf_load*1e3:.2f}ms"

    # Clock advancement
    load_overflow = simulate_clock_advancement(combined, 0.001, 0.001)
    print(f"  load_overflow={load_overflow*1e3:.2f}ms (combined={combined*1e3:.2f}ms vs inference=1ms)")
    # 10ms load > 1ms inference → overflow = 10ms - 1ms = 9ms
    assert load_overflow > 0, "10ms load > 1ms inference → overflow > 0"
    expected_overflow = max(0, combined - 0.001)
    assert abs(load_overflow - expected_overflow) < 1e-12, f"overflow: {load_overflow} vs {expected_overflow}"
    print(f"  ✅ High latency: load overflow={load_overflow*1e3:.1f}ms delays next round → TPOT increases")


def test_hbf_load_split_accuracy():
    """Test 3: HBF/DRAM split proportional to capacity usage."""
    print("\n--- Test 3: HBF/DRAM split accuracy ---")
    StateManager.reset()
    setup_mock_config()

    # HBF has 80 tokens, DRAM has 20 tokens → 80% HBF
    hbf_pool = MockHBFPool(logical_size=1000)
    dram_pool = MockHBFPool(logical_size=500)
    adapter = TieredHostKVCache(hbf_pool, dram_pool, PlacementPolicyV0(hbf_ratio=1.0))
    adapter.alloc(80)  # all to HBF
    # Manually alloc to DRAM
    dram_idx = dram_pool.alloc(20)
    adapter._dram_used += 20

    host_hit = 50
    result = compute_host_load_timing(host_hit, adapter)
    print(f"  HBF_used=80, DRAM_used=20, host_hit=50 → HBF={result['hbf_hit']}, DRAM={result['dram_hit']}")
    # 80% HBF → 40 tokens HBF, 10 tokens DRAM
    assert result["hbf_hit"] == 40, f"expected 40 HBF, got {result['hbf_hit']}"
    assert result["dram_hit"] == 10, f"expected 10 DRAM, got {result['dram_hit']}"
    print("  ✅ Split proportional to capacity usage (80% HBF, 20% DRAM)")


def test_load_overlap_semantics():
    """Test 4: Load/inference overlap — don't mechanically add load to TPOT."""
    print("\n--- Test 4: Load/inference overlap semantics ---")
    StateManager.reset()
    setup_mock_config(hbf_read_latency_us=500.0)  # 500μs

    hbf_pool = MockHBFPool(logical_size=100)
    dram_pool = MockHBFPool(logical_size=50)
    adapter = TieredHostKVCache(hbf_pool, dram_pool, PlacementPolicyV0(hbf_ratio=0.8))
    adapter.alloc(40)
    adapter.alloc(10)

    # Case A: load < inference → fully overlapped, overflow=0
    StateManager.reset()
    setup_mock_config(hbf_read_latency_us=500.0)
    StateManager.set_current_inference_dur(0.002)  # 2ms inference
    StateManager._last_inference_dur = 0.002
    compute_host_load_timing(20, adapter)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    overflow_a = simulate_clock_advancement(combined, 0.002, 0.002)
    print(f"  Case A: load={combined*1e6:.0f}μs, inference=2000μs → overflow={overflow_a*1e6:.0f}μs")
    assert overflow_a == 0, "load < inference → no overflow"

    # Case B: load > inference → overflow delays
    StateManager.reset()
    setup_mock_config(hbf_read_latency_us=500.0)
    StateManager.set_current_inference_dur(0.0001)  # 100μs inference (short)
    StateManager._last_inference_dur = 0.0001
    compute_host_load_timing(20, adapter)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    overflow_b = simulate_clock_advancement(combined, 0.0001, 0.0001)
    print(f"  Case B: load={combined*1e6:.0f}μs, inference=100μs → overflow={overflow_b*1e6:.0f}μs")
    assert overflow_b > 0, "load > inference → overflow > 0"

    print("  ✅ Overlap semantics: load overlapped with inference, only overflow delays TPOT")


def test_baseline_no_hbf():
    """Test 5: HBF disabled → no HBF load timing (backward compat)."""
    print("\n--- Test 5: Baseline (HBF off) ---")
    StateManager.reset()
    pc = PlatformConfig(device="H20", enable_hbf=False, hbf=None,
                        memory_read_bandwidth_gb=16, memory_write_bandwidth_gb=16)
    ConfigManager._platform_config = pc

    # No adapter → all load goes to DRAM
    result = compute_host_load_timing(20, None)
    print(f"  host_hit=20, no adapter → HBF={result['hbf_hit']}, DRAM={result['dram_hit']}")
    assert result["hbf_hit"] == 0
    assert result["dram_hit"] == 20

    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    print(f"  HBF load={hbf_load} (should be 0), DRAM load={l2_load*1e6:.1f}μs (should be >0)")
    assert hbf_load == 0, "no HBF → hbf_load should be 0"
    assert l2_load > 0, "DRAM load should be recorded"
    print("  ✅ Baseline: HBF off → all load via DRAM, no HBF timing")


def run_all():
    tests = [
        ("Normal latency (10μs)", test_hbf_load_timing_normal),
        ("High latency (10ms)", test_hbf_load_timing_high_latency),
        ("HBF/DRAM split accuracy", test_hbf_load_split_accuracy),
        ("Load/inference overlap", test_load_overlap_semantics),
        ("Baseline (HBF off)", test_baseline_no_hbf),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
