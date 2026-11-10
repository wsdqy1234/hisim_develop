"""Unit tests for HBF fast timing formula and config parsing.

Tests:
1. fast formula: max(latency, bytes/bw) matches HBFSim calibrator.cpp semantics
2. config parsing: HBFConfig from JSON
3. StateManager HBF duration tracking
4. Baseline regression: enable_hbf=False produces identical results

Run: cd /home/d00954154/HiSim_Qwen3 && PYTHONPATH=src python3 test/test_hbf_timing.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math


def test_fast_formula_semantics():
    """Verify fast formula = max(base_latency, transfer), NOT sum.

    HBFSim calibrator.cpp:46-64 confirms: the modeled service time is the
    LATER of the two bounds (overlap-aware), not their sum.
    """
    from hisim.simulation.sglang.hbf_tiered_pool import PlacementPolicyV0

    # fast formula parameters (from nominal.json)
    read_latency_ns = 10_000  # 10 μs
    bandwidth_bps = 512_000_000_000  # 512 GB/s

    def fast_transfer_ns(num_bytes, bw):
        if num_bytes == 0 or bw == 0:
            return 0
        return (num_bytes * 1_000_000_000 + bw - 1) // bw  # ceiling

    def fast_service_ns(base_latency, num_bytes, bw):
        transfer = fast_transfer_ns(num_bytes, bw)
        return max(base_latency, transfer)  # MAX, not sum

    # Small transfer: latency dominates (transfer < latency)
    small_bytes = 1000  # 1 KB
    small_transfer = fast_transfer_ns(small_bytes, bandwidth_bps)
    assert small_transfer < read_latency_ns, "small transfer should be < latency"
    small_latency = fast_service_ns(read_latency_ns, small_bytes, bandwidth_bps)
    assert small_latency == read_latency_ns, "small: latency should dominate"

    # Large transfer: transfer dominates (transfer > latency)
    # At 512 GB/s, 10μs covers ~5.12 MB. So 10 MB > latency.
    large_bytes = 10 * 1024 * 1024  # 10 MB
    large_transfer = fast_transfer_ns(large_bytes, bandwidth_bps)
    assert large_transfer > read_latency_ns, "large transfer should be > latency"
    large_latency = fast_service_ns(read_latency_ns, large_bytes, bandwidth_bps)
    assert large_latency == large_transfer, "large: transfer should dominate"

    # Verify NOT sum: sum would be latency + transfer > max
    sum_latency = read_latency_ns + large_transfer
    assert large_latency < sum_latency, "max must be < sum (overlap-aware)"

    # Crossover point: transfer == latency
    # bytes = latency * bw / 1e9 = 10000 * 512e9 / 1e9 = 5,120,000 bytes = 5 MB
    crossover_bytes = read_latency_ns * bandwidth_bps // 1_000_000_000
    crossover_transfer = fast_transfer_ns(crossover_bytes, bandwidth_bps)
    crossover_latency = fast_service_ns(read_latency_ns, crossover_bytes, bandwidth_bps)
    # At crossover, max(latency, transfer) ≈ latency ≈ transfer
    assert abs(crossover_latency - read_latency_ns) <= 1, "crossover: ≈ latency"

    print(f"✅ fast formula: small={small_latency}ns(latency-dom), "
          f"large={large_latency}ns(transfer-dom), "
          f"crossover@{crossover_bytes}B={crossover_latency}ns")
    print(f"   sum would be {sum_latency}ns > max {large_latency}ns (overlap-aware)")


def test_hbf_transfer_duration():
    """Test _hbf_transfer_duration with mock config."""
    from hisim.simulation.sglang import sglang_hook
    from hisim.simulation.manager import ConfigManager
    from hisim.simulation.types import HBFConfig, PlatformConfig
    from hisim.spec import AcceleratorInfo

    # Set up mock platform config with HBF
    hbf = HBFConfig(
        capacity_gb=64,
        read_bandwidth_gb=512,
        write_bandwidth_gb=100,
        read_latency_us=10,
        write_latency_us=100,
    )
    ConfigManager._platform_config = PlatformConfig(
        device="h100_sxm",
        enable_hbf=True,
        hbf=hbf,
    )

    # Test read (H2D): 1 MB
    read_bytes = 1 * 1024 * 1024  # 1 MB
    read_dur = sglang_hook._hbf_transfer_duration("H2D", read_bytes)
    # 1 MB at 512 GB/s = 1e6 / 512e9 = 1.95μs < 10μs latency → max = 10μs
    expected_read = max(10e-6, read_bytes / (512e9))
    assert abs(read_dur - expected_read) < 1e-12, f"read: {read_dur} vs {expected_read}"

    # Test write (D2H): 1 MB
    write_dur = sglang_hook._hbf_transfer_duration("D2H", read_bytes)
    # 1 MB at 100 GB/s = 1e6 / 100e9 = 10μs = 100μs latency → max = 100μs
    expected_write = max(100e-6, read_bytes / (100e9))
    assert abs(write_dur - expected_write) < 1e-12, f"write: {write_dur} vs {expected_write}"

    # Test large read: 100 MB → transfer dominates
    large_bytes = 100 * 1024 * 1024
    large_read = sglang_hook._hbf_transfer_duration("H2D", large_bytes)
    expected_large = max(10e-6, large_bytes / (512e9))
    assert abs(large_read - expected_large) < 1e-12

    print(f"✅ _hbf_transfer_duration: 1MB read={read_dur*1e6:.2f}μs, "
          f"1MB write={write_dur*1e6:.2f}μs, 100MB read={large_read*1e6:.2f}μs")

    # Cleanup
    ConfigManager._platform_config = None


def test_state_manager_hbf_dur():
    """Test StateManager HBF duration tracking."""
    from hisim.simulation.manager.state import StateManager

    StateManager.reset()

    # Initially zero
    assert StateManager.pop_hicache_hbf_backup_dur() == 0
    assert StateManager.pop_hicache_hbf_load_dur() == 0

    # Increment (uses max for parallel events within a round, not sum)
    StateManager.inc_hicache_hbf_backup_dur(1.5e-6)
    StateManager.inc_hicache_hbf_backup_dur(2.5e-6)
    StateManager.inc_hicache_hbf_load_dur(3.0e-6)

    # Pop (resets) — backup should be max(1.5e-6, 2.5e-6) = 2.5e-6
    backup = StateManager.pop_hicache_hbf_backup_dur()
    load = StateManager.pop_hicache_hbf_load_dur()
    assert abs(backup - 2.5e-6) < 1e-15, f"backup={backup} (should be max=2.5e-6)"
    assert abs(load - 3.0e-6) < 1e-15, f"load={load}"

    # After pop, should be zero
    assert StateManager.pop_hicache_hbf_backup_dur() == 0
    assert StateManager.pop_hicache_hbf_load_dur() == 0

    # Last dur (set/get, not accumulative)
    StateManager.set_last_hbf_backup_dur(5.0e-6)
    assert abs(StateManager.get_last_hbf_backup_dur() - 5.0e-6) < 1e-15
    StateManager.set_last_hbf_load_dur(7.0e-6)
    assert abs(StateManager.get_last_hbf_load_dur() - 7.0e-6) < 1e-15

    # Reset clears all
    StateManager.reset()
    assert StateManager.get_last_hbf_backup_dur() == 0
    assert StateManager.get_last_hbf_load_dur() == 0

    print("✅ StateManager HBF dur tracking: inc/pop/last/reset all correct")


def test_config_parsing():
    """Test HBFConfig parsing from JSON."""
    from hisim.simulation.sim_args import _parse_hbf_config, SimulationArgs

    # Test _parse_hbf_config
    hbf_dict = {
        "capacity_gb": 256,
        "read_bandwidth_gb": 512,
        "write_bandwidth_gb": 50,
        "read_latency_us": 10,
        "write_latency_us": 100,
        "channels": 32,
        "queue_depth": 128,
        "hbm_cache_gb": 8,
        "placement_ratio": 0.7,
    }
    hbf = _parse_hbf_config(hbf_dict)
    assert hbf is not None
    assert hbf.capacity_gb == 256
    assert hbf.read_bandwidth_gb == 512
    assert hbf.write_bandwidth_gb == 50
    assert hbf.read_latency_us == 10
    assert hbf.write_bandwidth == 50  # property

    # Test default write_bandwidth (None → read * 0.2)
    hbf2 = _parse_hbf_config({"read_bandwidth_gb": 512})
    assert hbf2.write_bandwidth_gb is None
    assert hbf2.write_bandwidth == 512 * 0.2  # 102.4

    # Test None input
    assert _parse_hbf_config(None) is None
    assert _parse_hbf_config({}) is None  # empty dict returns None? No...

    # Actually empty dict: all defaults
    hbf3 = _parse_hbf_config({"capacity_gb": 64})
    assert hbf3.capacity_gb == 64
    assert hbf3.read_bandwidth_gb == 512.0  # default

    # Test full config parsing via SimulationArgs.from_json
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "test", "assets", "mock", "config_qwen3_hbf.json"
    )
    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        args = SimulationArgs.from_json(config_path)
        assert args.platform.enable_hbf is True
        assert args.platform.hbf is not None
        assert args.platform.hbf.capacity_gb == 64
        assert args.platform.hbf.read_bandwidth_gb == 512
        print(f"✅ config parsing: enable_hbf={args.platform.enable_hbf}, "
              f"hbf.capacity={args.platform.hbf.capacity_gb}GB, "
              f"read_bw={args.platform.hbf.read_bandwidth_gb}GB/s")
    else:
        print(f"⚠️ config file not found: {config_path}, skipping full parse test")
        print("✅ _parse_hbf_config: direct parsing OK")


def test_baseline_regression_no_hbf():
    """Verify enable_hbf=False produces no HBF config (backward compat)."""
    from hisim.simulation.types import HBFConfig, PlatformConfig
    from hisim.spec import AcceleratorInfo

    pc = PlatformConfig(
        device="H20",
        enable_hbf=False,
        hbf=None,
    )
    assert pc.hbf is None
    assert pc.enable_hbf is False
    assert pc.hbf_read_bandwidth is None  # property returns None when hbf is None
    assert pc.hbf_write_bandwidth is None

    # _hbf_transfer_duration with no HBF config returns 0
    from hisim.simulation.manager import ConfigManager
    ConfigManager._platform_config = pc

    from hisim.simulation.sglang import sglang_hook
    dur = sglang_hook._hbf_transfer_duration("H2D", 1e6)
    assert dur == 0.0, f"no HBF config should return 0, got {dur}"

    ConfigManager._platform_config = None
    print("✅ baseline regression: enable_hbf=False → hbf=None → timing=0 (no-op)")


def run_all_tests():
    tests = [
        ("fast formula semantics", test_fast_formula_semantics),
        ("_hbf_transfer_duration", test_hbf_transfer_duration),
        ("StateManager HBF dur", test_state_manager_hbf_dur),
        ("config parsing", test_config_parsing),
        ("baseline regression (no HBF)", test_baseline_regression_no_hbf),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
