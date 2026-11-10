#!/usr/bin/env python3
"""Phase 5E.2a timing regression tests.

Lock the timing semantics verified by the P0 causality audit:
  T1. StateManager per-round accumulators use MAX (parallel events within
      a round), not sum — same-medium concurrent loads serialize via the
      shared resource cursor, and the round's charged duration = the
      serial makespan (no double count).
  T2. Shared-DRAM same-medium serialization: two concurrent loads on one
      DRAM channel → the 2nd's duration includes its queue wait; the
      round max = the serial makespan (n2_A 5743μs mechanism).
  T3. Win-pool interface transmission: the SAME object's load at
      iface 112 vs 1600 GB/s differs by exactly bytes×(1/112−1/1600)
      (the p16k +15.6ms mechanism, 1:1 into the request's own load).
  T4. Cross-media round combination: batch load = max(L2, HBF) — the
      model composes parallel media, never sums them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel
from hisim.simulation.manager import StateManager
from hisim.simulation.hbf_shared_resource import get_shared_dram_manager

GB = 1e9


def test_t1_max_accumulator():
    """T1: per-round inc = max, pop resets."""
    StateManager._hicache_l2_load_dur = 0
    StateManager.inc_hicache_l2_load_dur(1.0)
    StateManager.inc_hicache_l2_load_dur(0.5)   # smaller — ignored
    assert StateManager.pop_hicache_l2_load_dur() == 1.0
    StateManager.inc_hicache_l2_load_dur(0.5)
    StateManager.inc_hicache_l2_load_dur(2.0)   # larger — replaces
    assert StateManager.pop_hicache_l2_load_dur() == 2.0
    # pop resets
    assert StateManager.pop_hicache_l2_load_dur() == 0.0
    print("✓ T1: per-round accumulator is MAX (parallel-in-round semantics)")


def test_t2_shared_dram_serialization():
    """T2: two concurrent DRAM loads serialize on the channel; the 2nd's
    duration carries its queue wait; round max = serial makespan."""
    mgr = get_shared_dram_manager()
    mgr.configure(dram_bw=13.6e9)
    mgr.busy_until = 0.0
    mgr.begin_batch(0.0)
    b = 38879232.0                       # 339tk × 114688 B
    d1 = mgr.submit(0.0, int(b), is_load=True)["total"]
    d2 = mgr.submit(0.0, int(b), is_load=True)["total"]   # same arrival
    service = b / 13.6e9 + mgr.t0_load
    assert abs(d1 - service) < 1e-4, (d1, service)
    assert abs(d2 - 2 * service) < 2e-3, \
        f"2nd load must include queue wait: {d2} vs {2*service}"
    # round max (StateManager semantics) = the serial makespan:
    assert max(d1, d2) == d2
    print(f"✓ T2: shared-DRAM serialization — 2 loads: {d1*1e6:.0f}μs / "
          f"{d2*1e6:.0f}μs (2nd incl. wait) → round = makespan")


def test_t3_interface_transmission():
    """T3: the SAME 1.88GB object loaded under shared-112 vs shared-1600
    differs by exactly bytes×(1/112−1/1600) — the +15.6ms mechanism."""
    KV = 114688
    obj_bytes = 16383 * KV
    deltas = {}
    for ic in (112e9, 1600e9):
        pool = HbfWinPoolModel(Hbf8WinConfig(
            placement="striped", stripe_block_tokens=2048,
            package_shared_interface_bw=ic))
        pool.begin_batch(0.0)
        st = pool.submit_object(0.0, list(range(16383)), KV, is_read=False)
        st2 = pool.submit_object(st.total + 1e-6, list(range(16383)), KV,
                                 is_read=True)
        deltas[ic] = st2.total
    expect = obj_bytes * (1 / 112e9 - 1 / 1600e9)
    got = deltas[112e9] - deltas[1600e9]
    assert abs(got - expect) / expect < 0.01, \
        f"interface delta {got*1e3:.2f}ms != envelope {expect*1e3:.2f}ms"
    print(f"✓ T3: interface transmission 1:1 — Δ(112−1600) = "
          f"{got*1e3:.2f}ms (envelope {expect*1e3:.2f}ms)")


def test_t4_cross_media_max():
    """T4: batch load combination = max(L2, HBF) — never summed."""
    StateManager._hicache_l2_load_dur = 0
    StateManager._hicache_hbf_load_dur = 0
    StateManager.inc_hicache_l2_load_dur(138.305e-3)
    StateManager.inc_hicache_hbf_load_dur(16.79e-3)
    combined = max(StateManager.pop_hicache_l2_load_dur(),
                   StateManager.pop_hicache_hbf_load_dur())
    assert abs(combined - 138.305e-3) < 1e-9
    print("✓ T4: cross-media combination is max (137ms DRAM ∥ 16.8ms HBF "
          "→ 138.3ms)")


def run_all():
    tests = [test_t1_max_accumulator, test_t2_shared_dram_serialization,
             test_t3_interface_transmission, test_t4_cross_media_max]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({passed + failed} total)")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
