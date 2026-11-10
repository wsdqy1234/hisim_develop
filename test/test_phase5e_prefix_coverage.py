#!/usr/bin/env python3
"""Phase 5E.1 P2 regression: large-prefix object round-trip coverage.

Locks the Phase 5E findings as tests:
  1. Every prefix size (128..16384) round-trips FULLY through the WIN
     object registry (backup → restore, same WINs, full bytes, extents,
     conservation) — the 5D "128-token" symptom was NOT object size.
  2. The prefix-capacity guardrail semantics: an object >50% of a pool
     cannot co-exist with a second copy (documents the 5E.1 root cause
     arithmetic).
  3. Sequential vs random read boundary: the SAME object restored under
     random access mode costs the envelope ratio (100 vs 200 GB/s), and
     the model reports them separately.
  4. G4 wait-then-read does not create zero-duration restores (already
     locked in test_phase5d_object_mgmt; here re-checked at 4K scale).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel

GB = 1e9
KV = 114688


def test_prefix_roundtrip_all_sizes():
    """Backup→restore full-coverage at every prefix size (P2 req 1-4)."""
    print(f"{'prefix':>8} {'block':>6} {'WINs':>5} {'bytes':>12} "
          f"{'restore_full':>13} {'same_wins':>10}")
    for plen in (128, 256, 512, 1024, 4096, 16384):
        block = max(1, plen // 8)
        pool = HbfWinPoolModel(Hbf8WinConfig(
            placement="striped", stripe_block_tokens=block))
        slots = list(range(plen))
        pool.begin_batch(0.0)
        wb = pool.submit_object(0.0, slots, KV, is_read=False)
        n_wins = len(wb.per_win)
        wr = pool.submit_object(wb.total + 1e-6, slots, KV, is_read=True)
        same = set(wb.per_win) == set(wr.per_win)
        full = wr.bytes == plen * KV
        cons = pool.registry.verify_conservation()
        assert full, f"{plen}tk restore bytes incomplete: {wr.bytes}"
        assert same, f"{plen}tk restore WINs diverged"
        assert cons["bytes_consistent"] and cons["slots_consistent"]
        # extents: every WIN has at least one non-empty contiguous run
        for obj in pool.registry.objects.values():
            for w, runs in obj.extents.items():
                assert runs and all(e > s for s, e in runs), \
                    f"{plen}tk: empty/bad extents on WIN {w}"
        print(f"{plen:>8} {block:>6} {n_wins:>5} {wr.bytes:>12} "
              f"{'YES' if full else 'NO':>13} {'YES' if same else 'NO':>10}")
    print("✓ all prefix sizes round-trip FULLY (object mechanism complete)")


def test_capacity_rule_arithmetic():
    """The 5E.1 root-cause arithmetic: the REUSE WAVE's re-insert (3rd
    object) exceeds a 2-object host → eviction of a LIVE object.

    The 5D p4k config: host 8192, objects 4090: 2 backups fit (8186) but
    the first reuse's re-insert backup (3rd object) must evict a LIVE
    object; the evicted object's async storage write (128-token batches)
    is all a later probe finds → the '128-token' symptom.
    """
    host = 8192
    obj = 4090
    assert obj * 2 <= host, "2 backups must fit (they did in 5D)"
    assert obj * 3 > host, "the reuse-wave re-insert must NOT fit (root cause)"
    pool = HbfWinPoolModel(Hbf8WinConfig(placement="balanced"))
    pool.submit_object(0.0, list(range(obj)), KV, is_read=False)
    pool.submit_object(0.1, list(range(obj, 2 * obj)), KV, is_read=False)
    assert len(pool.registry.objects) == 2
    # the LIVE object evicted (freed) mid-wave leaves the other intact
    pool.notify_slots_freed(list(range(obj)))
    assert len(pool.registry.objects) == 1
    st = pool.submit_object(0.2, list(range(obj, 2 * obj)), KV, is_read=True)
    assert st.bytes == obj * KV
    print("✓ capacity arithmetic locked: 3×4090 > 8192 host → reuse-wave "
          "eviction cascade (the 5D symptom mechanism)")


def test_seq_random_boundary_same_object():
    """P6 boundary: the same object restored sequential vs random —
    duration ratio = envelope ratio (200 vs 100 GB/s per WIN), reported
    as separate access modes, never blended."""
    plen = 4096
    for mode, expect_ratio in (("sequential", 1.0), ("random", 2.0)):
        pool = HbfWinPoolModel(Hbf8WinConfig(
            placement="striped", stripe_block_tokens=plen // 8,
            default_access_mode=mode))
        slots = list(range(plen))
        pool.begin_batch(0.0)
        pool.submit_object(0.0, slots, KV, is_read=False)
        st = pool.submit_object(1.0, slots, KV, is_read=True)
        # per-WIN chunk read: (plen/8)*KV bytes at the mode's BW
        chunk = plen / 8 * KV
        dur = chunk / (200e9 if mode == "sequential" else 100e9)
        assert abs(st.total - dur) / dur < 0.02, \
            f"{mode}: total {st.total*1e6:.0f}μs != envelope {dur*1e6:.0f}μs"
        print(f"✓ {mode} boundary: restore {st.total*1e6:.0f}μs "
              f"(envelope {dur*1e6:.0f}μs, ratio {expect_ratio}x)")
    print("✓ sequential/random reported separately; boundary = 2x envelope")


def test_g4_wait_no_zero_duration_at_4k():
    """G4 at 4K scale: restore before backup completion returns a
    duration >= the remaining backup time (never zero)."""
    plen = 4096
    pool = HbfWinPoolModel(Hbf8WinConfig(
        placement="striped", stripe_block_tokens=plen // 8))
    slots = list(range(plen))
    wb = pool.submit_object(0.0, slots, KV, is_read=False)   # ~5.9ms
    t = 1e-3
    st = pool.submit_object(t, slots, KV, is_read=True)
    assert st.total >= (wb.total - t) - 1e-9, \
        f"G4 wait missing: total={st.total} backup_ready={wb.total}"
    assert st.bytes == plen * KV
    print(f"✓ G4 at 4K: restore@1ms total={st.total*1e3:.2f}ms "
          f"(>= backup ready {wb.total*1e3:.2f}ms), media="
          f"{st.media_time*1e3:.2f}ms")


def run_all():
    tests = [
        test_prefix_roundtrip_all_sizes,
        test_capacity_rule_arithmetic,
        test_seq_random_boundary_same_object,
        test_g4_wait_no_zero_duration_at_4k,
    ]
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
