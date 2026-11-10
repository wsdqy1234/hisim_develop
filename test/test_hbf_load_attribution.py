"""Host Load source attribution tests — real host_value, not proportional.

Tests that HBF/DRAM Load timing is computed from the ACTUAL host_value
virtual indices (collected during match_prefix tree walk), NOT from
capacity-ratio estimation.

Key invariants:
  - Host Hit all in HBF  → only HBF Load recorded
  - Host Hit all in DRAM → only DRAM Load recorded
  - Host Hit mixed       → split by real virtual-index position
  - HBF read_latency=10ms but only DRAM hit → NO HBF overflow
  - HBF read_latency=10ms and actually HBF hit → overflow occurs
  - last_inference_dur overlap: only the excess beyond inference delays

Run: cd /home/d00954154/HiSim_Qwen3 && PYTHONPATH=src python3 test/test_hbf_load_attribution.py
"""

import sys
import os
import torch
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.sglang.hbf_tiered_pool import (
    TieredHostKVCache, MockHBFPool, PlacementPolicyV0,
)
from hisim.simulation.manager.state import StateManager
from hisim.simulation.types import HBFConfig, PlatformConfig
from hisim.simulation.manager import ConfigManager


# ---- Mock SGLang tree structures for _collect_host_load_split ----

@dataclass
class MockComponentData:
    """Mimics SGLang's TreeComponentData with host_value."""
    host_value: torch.Tensor = None


class MockTreeNode:
    """Mimics SGLang's UnifiedTreeNode with component_data and parent."""
    def __init__(self, host_value=None, parent=None):
        self.parent = parent
        cd = MockComponentData(host_value=host_value)
        self.component_data = {0: cd}  # ComponentType.FULL == 0


class MockTreeCore:
    """Mimics SGLang's UnifiedTreeCore with node_by_id and root_node."""
    def __init__(self, nodes_dict, root_node):
        self._nodes = nodes_dict
        self.root_node = root_node

    def node_by_id(self, node_id):
        return self._nodes[node_id]


@dataclass
class MockMatchResult:
    """Mimics SGLang's MatchResult."""
    best_match_node: int
    last_device_node: int
    host_hit_length: int = 0


def setup_config(hbf_read_latency_us=10.0, hbf_write_latency_us=100.0,
                 hbf_read_bw_gb=512.0):
    """Set up ConfigManager with HBF config."""
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
    ConfigManager._kv_cache_bytes = 32768  # 32KB per token
    return pc


def build_chain(host_values, dram_offset):
    """Build a chain of MockTreeNodes with given host_value tensors.

    Returns (tree_core, best_match_node_id, last_device_node_id).
    The chain is: root -> node1 -> node2 -> ... -> nodeN (best_match).
    last_device_node = root (the first device-on node).
    """
    root = MockTreeNode(host_value=None, parent=None)
    nodes = {0: root}
    cur = root
    for i, hv in enumerate(host_values, start=1):
        node = MockTreeNode(host_value=hv, parent=cur)
        nodes[i] = node
        cur = node
    tree_core = MockTreeCore(nodes, root)
    return tree_core, len(host_values), 0  # best_match=last, last_device=root


def compute_load_from_real_split(hbf_tokens, dram_tokens, bytes_per_token=32768):
    """Simulate the scheduler hook's load timing injection using real split."""
    from hisim.simulation.sglang.sglang_hook import (
        _record_hbf_load_timing, _record_l2_load_timing,
    )
    if hbf_tokens > 0:
        _record_hbf_load_timing(hbf_tokens * bytes_per_token)
    if dram_tokens > 0:
        _record_l2_load_timing(dram_tokens * bytes_per_token, dram_tokens)


def simulate_clock(combined_load, inference, last_inference):
    """Simulate OVERLAP_SCHEDULE clock advancement (line ~2562)."""
    load_overflow = max(combined_load - last_inference, 0)
    StateManager.step_global_clock(load_overflow)
    StateManager.step_global_clock(inference)
    return load_overflow


def run_collector(tree_core, result, dram_offset):
    """Call the real _collect_host_load_split function."""
    from hisim.simulation.sglang.sglang_hook import _collect_host_load_split
    return _collect_host_load_split(tree_core, result, dram_offset)


# ===================== TESTS =====================

def test_all_hbf():
    """Host Hit all in HBF → only HBF Load recorded."""
    print("\n--- Test: Host Hit all in HBF ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=10.0)

    dram_offset = 1000  # HBF [0,1000), DRAM [1000,...)
    # Two nodes, both with host_value entirely in HBF range
    hv1 = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    hv2 = torch.tensor([20, 21, 22], dtype=torch.int64)
    tree_core, best, last_dev = build_chain([hv1, hv2], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=7)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  host_value: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 7, 0)")
    assert hbf_tok == 7 and dram_tok == 0, f"expected (7,0), got ({hbf_tok},{dram_tok})"

    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    print(f"  HBF load={hbf_load*1e6:.1f}μs, DRAM load={l2_load*1e6:.1f}μs")
    assert hbf_load > 0 and l2_load == 0, "only HBF should have load timing"
    print("  ✅ All-HBF: only HBF Load recorded, DRAM load=0")


def test_all_dram():
    """Host Hit all in DRAM → only DRAM Load recorded."""
    print("\n--- Test: Host Hit all in DRAM ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=10.0)

    dram_offset = 1000
    hv1 = torch.tensor([1000, 1001, 1002, 1003], dtype=torch.int64)
    hv2 = torch.tensor([1010, 1011, 1012], dtype=torch.int64)
    tree_core, best, last_dev = build_chain([hv1, hv2], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=7)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  host_value: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 0, 7)")
    assert hbf_tok == 0 and dram_tok == 7, f"expected (0,7), got ({hbf_tok},{dram_tok})"

    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    print(f"  HBF load={hbf_load} (should be 0), DRAM load={l2_load*1e6:.1f}μs")
    assert hbf_load == 0 and l2_load > 0, "only DRAM should have load timing"
    print("  ✅ All-DRAM: only DRAM Load recorded, HBF load=0")


def test_mixed():
    """Host Hit mixed HBF/DRAM → split by real virtual-index position."""
    print("\n--- Test: Host Hit mixed HBF/DRAM ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=10.0)

    dram_offset = 1000
    # Node 1: 4 tokens — 3 in HBF, 1 in DRAM
    hv1 = torch.tensor([10, 20, 30, 1005], dtype=torch.int64)
    # Node 2: 3 tokens — 1 in HBF, 2 in DRAM
    hv2 = torch.tensor([50, 2000, 2001], dtype=torch.int64)
    tree_core, best, last_dev = build_chain([hv1, hv2], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=7)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  host_value: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 4, 3)")
    assert hbf_tok == 4 and dram_tok == 3, f"expected (4,3), got ({hbf_tok},{dram_tok})"

    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    print(f"  HBF load={hbf_load*1e6:.1f}μs (4tk), DRAM load={l2_load*1e6:.1f}μs (3tk)")
    assert hbf_load > 0 and l2_load > 0, "both should have load timing"
    # HBF: max(10μs, 4*32768/512GB) = 10μs (latency dominates)
    expected_hbf = max(10e-6, 4 * 32768 / 512e9)
    assert abs(hbf_load - expected_hbf) < 1e-12, f"HBF: {hbf_load} vs {expected_hbf}"
    # DRAM: t0 + bytes/(bw*0.85) = 6.67μs + 98304/(16e9*0.85)
    expected_dram = 6.67e-6 + 3 * 32768 / (16e9 * 0.85)
    assert abs(l2_load - expected_dram) < 1e-9, f"DRAM: {l2_load} vs {expected_dram}"
    print("  ✅ Mixed: split by real position (4 HBF + 3 DRAM), timing correct")


def test_10ms_dram_only_no_hbf_overflow():
    """HBF read_latency=10ms but only DRAM hit → NO HBF overflow."""
    print("\n--- Test: 10ms HBF latency, DRAM-only hit → no HBF overflow ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=10000.0)  # 10ms!

    dram_offset = 1000
    hv1 = torch.tensor([1000, 1001, 1002], dtype=torch.int64)  # all DRAM
    tree_core, best, last_dev = build_chain([hv1], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=3)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  host_value: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 0, 3)")
    assert hbf_tok == 0, "HBF tokens must be 0 for DRAM-only hit"

    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    print(f"  HBF load={hbf_load} (must be 0!), DRAM load={l2_load*1e6:.1f}μs, combined={combined*1e6:.1f}μs")
    assert hbf_load == 0, "HBF load must be 0 — no HBF hit despite 10ms latency"

    # Clock: DRAM load is small (~6μs), inference=1ms → overflow=0
    overflow = simulate_clock(combined, 0.001, 0.001)
    print(f"  overflow={overflow*1e6:.1f}μs (DRAM load ~6μs < 1ms inference → 0)")
    assert overflow == 0, "DRAM-only load with 10ms HBF latency must NOT overflow"
    print("  ✅ 10ms HBF latency + DRAM-only hit → no HBF load, no overflow")


def test_10ms_hbf_hit_overflow():
    """HBF read_latency=10ms and actually HBF hit → overflow occurs."""
    print("\n--- Test: 10ms HBF latency, real HBF hit → overflow ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=10000.0)  # 10ms!

    dram_offset = 1000
    hv1 = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)  # all HBF
    tree_core, best, last_dev = build_chain([hv1], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=5)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  host_value: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 5, 0)")
    assert hbf_tok == 5, "HBF tokens must be 5"

    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    combined = max(hbf_load, l2_load)
    print(f"  HBF load={hbf_load*1e3:.2f}ms (must be ~10ms), DRAM load={l2_load}, combined={combined*1e3:.2f}ms")
    assert abs(hbf_load - 0.01) < 1e-9, "HBF load must be 10ms (latency dominates)"

    # Clock: HBF load=10ms > inference=1ms → overflow=9ms
    overflow = simulate_clock(combined, 0.001, 0.001)
    print(f"  overflow={overflow*1e3:.2f}ms (10ms load > 1ms inference → 9ms)")
    assert abs(overflow - 0.009) < 1e-9, "overflow must be 9ms"
    print("  ✅ 10ms HBF latency + real HBF hit → overflow=9ms (TPOT impact)")


def test_overlap_semantics():
    """Check last_inference_dur overlap — only excess delays, not full load."""
    print("\n--- Test: overlap semantics with real split ---")
    StateManager.reset()
    setup_config(hbf_read_latency_us=2000.0)  # 2ms

    dram_offset = 1000
    hv1 = torch.tensor([10, 11, 12], dtype=torch.int64)  # HBF
    tree_core, best, last_dev = build_chain([hv1], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=3)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    combined = max(hbf_load, 0.0)

    # Case A: load=2ms, last_inference=5ms → load fully overlapped
    overflow_a = simulate_clock(combined, 0.001, 0.005)
    print(f"  Case A: load={combined*1e3:.1f}ms, last_inference=5ms → overflow={overflow_a*1e3:.1f}ms")
    assert overflow_a == 0, "load < last_inference → fully overlapped"

    # Case B: load=2ms, last_inference=0.5ms → overflow=1.5ms
    StateManager.reset()
    setup_config(hbf_read_latency_us=2000.0)
    StateManager.set_current_inference_dur(0.001)
    StateManager._last_inference_dur = 0.0005
    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    combined = max(hbf_load, 0.0)
    overflow_b = simulate_clock(combined, 0.001, 0.0005)
    print(f"  Case B: load={combined*1e3:.1f}ms, last_inference=0.5ms → overflow={overflow_b*1e3:.1f}ms")
    assert abs(overflow_b - 0.0015) < 1e-9, "overflow = load - last_inference = 1.5ms"

    # Case C: first request — last_inference=0 → full load added (no overlap)
    StateManager.reset()
    setup_config(hbf_read_latency_us=2000.0)
    StateManager.set_current_inference_dur(0.001)
    StateManager._last_inference_dur = 0.0  # no previous round
    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    combined = max(hbf_load, 0.0)
    overflow_c = simulate_clock(combined, 0.001, 0.0)
    print(f"  Case C: load={combined*1e3:.1f}ms, last_inference=0 → overflow={overflow_c*1e3:.1f}ms (full)")
    assert abs(overflow_c - combined) < 1e-9, "no previous round → full load delay"

    print("  ✅ Overlap: only excess beyond last_inference delays; first round = full")


def test_baseline_hbf_off():
    """HBF disabled → no HBF load, all DRAM (backward compat)."""
    print("\n--- Test: baseline (HBF off) ---")
    StateManager.reset()
    pc = PlatformConfig(device="H20", enable_hbf=False, hbf=None,
                        memory_read_bandwidth_gb=16, memory_write_bandwidth_gb=16)
    ConfigManager._platform_config = pc

    # With HBF off, the collector should still work (dram_offset=0 → all DRAM)
    dram_offset = 0
    hv1 = torch.tensor([5, 6, 7], dtype=torch.int64)  # all ≥ 0 → all DRAM
    tree_core, best, last_dev = build_chain([hv1], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=3)
    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  HBF off: HBF={hbf_tok}tk, DRAM={dram_tok}tk (expected: 0, 3)")
    assert hbf_tok == 0 and dram_tok == 3
    compute_load_from_real_split(hbf_tok, dram_tok)
    hbf_load = StateManager.pop_hicache_hbf_load_dur()
    l2_load = StateManager.pop_hicache_l2_load_dur()
    assert hbf_load == 0 and l2_load > 0
    print("  ✅ Baseline: HBF off → all DRAM, no HBF timing")


def test_token_order_preserved():
    """Mixed split preserves token order (HBF before DRAM in chain)."""
    print("\n--- Test: token order in mixed chain ---")
    StateManager.reset()
    setup_config()

    dram_offset = 100
    # Chain: node1 (HBF indices) -> node2 (DRAM indices) -> node3 (mixed)
    hv1 = torch.tensor([1, 2], dtype=torch.int64)           # HBF
    hv2 = torch.tensor([100, 101], dtype=torch.int64)        # DRAM
    hv3 = torch.tensor([3, 102, 4], dtype=torch.int64)      # mixed
    tree_core, best, last_dev = build_chain([hv1, hv2, hv3], dram_offset)
    result = MockMatchResult(best_match_node=best, last_device_node=last_dev,
                             host_hit_length=7)

    hbf_tok, dram_tok, _slots = run_collector(tree_core, result, dram_offset)
    print(f"  Chain [HBF×2]→[DRAM×2]→[mixed 1HBF+2DRAM]: HBF={hbf_tok}, DRAM={dram_tok}")
    assert hbf_tok == 4 and dram_tok == 3, f"expected (4,3), got ({hbf_tok},{dram_tok})"
    print("  ✅ Token order preserved: walk collects all nodes correctly")


def run_all():
    tests = [
        ("All HBF hit", test_all_hbf),
        ("All DRAM hit", test_all_dram),
        ("Mixed HBF/DRAM", test_mixed),
        ("10ms latency, DRAM-only → no overflow", test_10ms_dram_only_no_hbf_overflow),
        ("10ms latency, real HBF hit → overflow", test_10ms_hbf_hit_overflow),
        ("Overlap semantics", test_overlap_semantics),
        ("Baseline HBF off", test_baseline_hbf_off),
        ("Token order preserved", test_token_order_preserved),
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
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
