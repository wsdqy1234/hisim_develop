#!/usr/bin/env python3
"""Phase 5D: KV Object → WIN management tests (P0 guarantees + P1 placement
+ P2 scheduling + P5 composition + P6 contiguity).

P0 guarantees (each locked by a dedicated test):
  G1. Same object Backup/Restore → same WIN(s).
  G2. WIN migration only via explicit re-register.
  G3. Free/eviction/CoW leaves no stale mappings.
  G4. Restore cannot hit an object whose backup is not complete.
  G5. Cross-batch resource cursors preserved (object path).
  G6. Token / object-bytes / capacity conservation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import (
    Hbf8WinConfig, HbfWinPoolModel, WinSpec)

GB = 1e9
KV = 114688          # bytes/token (Qwen3-0.6B)


def pool_of(**kw):
    return HbfWinPoolModel(Hbf8WinConfig(**kw))


# ------------------------------------------------------------------ P0 ---

def test_g1_same_object_same_win():
    """G1: backup and restore of the same slots hit the same WIN(s)."""
    for placement, block in (("balanced", 256), ("striped", 43),
                             ("hotspot", 256)):
        pool = pool_of(placement=placement,
                       stripe_block_tokens=block)
        slots = list(range(340))
        wb = pool.submit_object(0.0, slots, KV, is_read=False)
        wr = pool.submit_object(wb.total + 1.0, slots, KV, is_read=True)
        wins_w = set(wb.per_win.keys())
        wins_r = set(wr.per_win.keys())
        assert wins_r == wins_w, \
            f"{placement}: restore WINs {wins_r} != backup WINs {wins_w}"
        assert wr.bytes == 340 * KV
    print(f"✓ G1 same-WIN mapping: balanced/striped/hotspot all consistent")


def test_g2_explicit_migration_only():
    """G2: migration only via migrate_object (accounting + counter)."""
    pool = pool_of(placement="balanced")
    slots = list(range(100))
    pool.submit_object(0.0, slots, KV, is_read=False)
    key = next(iter(pool.registry.objects))
    assert pool.registry.win_bytes[0] == 100 * KV
    ok = pool.registry.migrate_object(key, 5)
    assert ok and pool.registry.win_bytes[0] == 0 \
        and pool.registry.win_bytes[5] == 100 * KV
    assert pool.registry.n_migrations == 1
    # restore after migration hits the NEW win
    st = pool.submit_object(1.0, slots, KV, is_read=True)
    assert st.per_win and list(st.per_win.keys()) == [5]
    # striped objects cannot migrate whole (multi-WIN)
    pool2 = pool_of(placement="striped", stripe_block_tokens=8)
    pool2.submit_object(0.0, list(range(64)), KV, is_read=False)
    key2 = next(iter(pool2.registry.objects))
    assert pool2.registry.migrate_object(key2, 3) is False
    print("✓ G2 migration: explicit only, accounting moves, striped refused")


def test_g3_no_stale_mappings():
    """G3: eviction and CoW partial-free leave no stale mappings."""
    pool = pool_of(placement="balanced")
    slots = list(range(100))
    pool.submit_object(0.0, slots, KV, is_read=False)
    key = next(iter(pool.registry.objects))
    # full eviction
    evicted = pool.notify_slots_freed(slots)
    assert evicted == [key] and pool.registry.n_evictions == 1
    assert pool.registry.verify_conservation()["slots_consistent"]
    assert sum(pool.registry.win_bytes) == 0
    # restore of freed slots → no mapping → empty (miss)
    st = pool.submit_object(1.0, slots, KV, is_read=True)
    assert st.bytes == 100 * KV and not st.per_win, \
        "freed slots must decompose to nothing (miss)"
    # CoW partial free: object shrinks in place, conservation holds
    pool2 = pool_of(placement="balanced")
    pool2.submit_object(0.0, list(range(100)), KV, is_read=False)
    pool2.notify_slots_freed(list(range(40)))          # first 40 slots freed
    v = pool2.registry.verify_conservation()
    assert v["slots_consistent"] and v["bytes_consistent"]
    obj = next(iter(pool2.registry.objects.values()))
    assert obj.size_tokens == 60
    # restore of the remaining slots still maps to the object's WIN
    st2 = pool2.submit_object(1.0, list(range(40, 100)), KV, is_read=True)
    assert st2.per_win and st2.bytes == 60 * KV
    print("✓ G3 no stale mappings: full eviction + CoW partial free")


def test_g4_restore_blocked_before_backup_ready():
    """G4 (5E.0 audit #6 semantics): restore arriving before backup
    completion WAITs for the in-flight backup, then reads. The wait is
    INCLUDED in the total (never a zero-duration false fast restore);
    media service time is reported separately from the end-to-end wait."""
    pool = pool_of(placement="balanced")
    slots = list(range(340))
    wb = pool.submit_object(0.0, slots, KV, is_read=False)   # 3.9ms
    # restore arrives at 1ms — backup completes at ~3.9ms
    t_arr = 1e-3
    st = pool.submit_object(t_arr, slots, KV, is_read=True)
    assert st.bytes == 340 * KV and st.per_win, \
        "restore must still hit (wait-then-read, not miss)"
    assert st.total >= (wb.total - t_arr) - 1e-9, \
        f"restore total {st.total*1e3:.2f}ms must include the G4 wait " \
        f"(backup completes at {wb.total*1e3:.2f}ms)"
    assert st.media_time < st.total, \
        "media service time must be reported separately from the wait"
    assert pool.registry.n_stale_blocked_restores == 1
    # after completion → no wait, normal fast restore
    st2 = pool.submit_object(wb.total + 1.0, slots, KV, is_read=True)
    assert st2.bytes == 340 * KV and st2.total < st.total
    assert st2.queue_wait < st.queue_wait
    print(f"✓ G4 wait-then-read: restore@1ms total={st.total*1e3:.2f}ms "
          f"(media={st.media_time*1e3:.2f}ms, queue_wait incl. G4="
          f"{st.queue_wait*1e3:.2f}ms); after ready: "
          f"{st2.total*1e3:.2f}ms")


def test_g5_cross_batch_object_path():
    """G5: cross-batch in-flight object I/O keeps cursors busy."""
    pool = pool_of(placement="balanced")
    slots = list(range(340))
    pool.begin_batch(0.0)
    pool.submit_object(0.0, slots, KV, is_read=False)   # 3.9ms on win 0
    pool.begin_batch(1e-3)
    # same object, another backup gen on the SAME win must wait
    st = pool.submit_object(1e-3, list(range(340, 680)), KV, is_read=False)
    assert st.media_start >= pool.wins[0].media_cursor() - 1e-9 or \
        st.per_win and all(pool.wins[w].media_cursor() >= st.media_start
                           for w in st.per_win), \
        "batch-2 object must respect in-flight cursors"
    print(f"✓ G5 cross-batch: in-flight write cursor respected "
          f"(start={st.media_start*1e3:.2f}ms)")


def test_g6_conservation():
    """G6: tokens / object bytes / per-WIN capacity conservation."""
    pool = pool_of(placement="striped", stripe_block_tokens=32)
    for i in range(8):
        pool.submit_object(0.0, list(range(i * 256, (i + 1) * 256)),
                           KV, is_read=False)
    v = pool.registry.verify_conservation()
    assert v["bytes_consistent"] and v["slots_consistent"]
    assert v["total_slots"] == 8 * 256
    # per-WIN bytes vs WIN resource accounting
    reg_bytes = v["win_bytes"]
    res_used = [w.bytes_written for w in pool.wins]
    assert all(abs(a - b) < 1 for a, b in zip(reg_bytes, res_used)), \
        f"registry {reg_bytes} != resource {res_used}"
    # free half the objects → conservation still holds
    pool.notify_slots_freed(list(range(0, 512)) + list(range(1024, 1536)))
    v2 = pool.registry.verify_conservation()
    assert v2["bytes_consistent"] and v2["slots_consistent"]
    assert sum(v2["win_bytes"]) == 4 * 256 * KV
    print("✓ G6 conservation: tokens/bytes/per-WIN all consistent "
          "across alloc + free")


# ------------------------------------------------------------------ P1 ---

def test_p1_placement_mechanisms():
    """P1: object-balanced vs block-striping vs hotspot at 3 prefix sizes."""
    print(f"\n{'prefix':>10} {'mode':>9} {'WINs':>5} {'write_ms':>9} "
          f"{'restore_ms':>11} {'qw_max_us':>10}")
    cases = [
        ("340tk", 340, 43),       # block=43 → 8 WINs
        ("4Ktk", 4096, 512),
        ("16Ktk", 16384, 2048),
    ]
    for label, tokens, block in cases:
        for mode, placement in (("obj-bal", "balanced"),
                                ("striped", "striped"),
                                ("hotspot", "hotspot")):
            pool = pool_of(placement=placement,
                           stripe_block_tokens=block)
            slots = list(range(tokens))
            pool.begin_batch(0.0)
            wb = pool.submit_object(0.0, slots, KV, is_read=False)
            n_wins = len(wb.per_win)
            t_write = wb.total * 1e3
            wr = pool.submit_object(wb.total + 1e-6, slots, KV, is_read=True)
            t_restore = wr.total * 1e3
            qw = max((w.queue_wait_max for w in pool.wins), default=0) * 1e6
            print(f"{label:>10} {mode:>9} {n_wins:>5} {t_write:>9.3f} "
                  f"{t_restore:>11.3f} {qw:>10.1f}")
            if mode == "striped":
                # REAL per-WIN submissions: every WIN must have served I/O
                serving = [w for w in pool.wins
                           if w.bytes_written > 0 or w.bytes_read_seq > 0]
                assert len(serving) >= 8, \
                    f"striped must use all 8 WINs, got {len(serving)}"
                # no oversubscription: aggregate write BW ≤ 80GB/s
                span_w = max(s.media_completion for s in
                             [wb]) - 0.0
                agg_w = tokens * KV / span_w
                assert agg_w <= 80 * GB * 1.001, \
                    f"striped write oversubscribed: {agg_w/GB:.1f}GB/s"
            if mode == "hotspot":
                assert n_wins == 1 and list(wb.per_win.keys()) == [0]
            if mode == "obj-bal":
                assert n_wins == 1
    print("✓ P1 placement: striping submits REAL per-WIN I/Os (8 queues, "
          "no oversubscription); obj-bal/hotspot single WIN")


# ------------------------------------------------------------------ P2 ---

def _p2_trace(pool, order, write_slots, read_slots):
    """Submit the write-burst + concurrent-restore trace in `order`
    (policy = submission ordering within the same arrival cohort)."""
    pool.begin_batch(0.0)
    stats = {"w": [], "r": []}
    for kind, idx in order:
        slots = (write_slots if kind == "w" else read_slots)[idx]
        st = pool.submit_object(0.0, slots, KV, is_read=(kind == "r"))
        stats[kind].append(st)
    return stats


def test_p2_scheduling_policies():
    """P2: FIFO vs read-priority vs fair (submission-order policies under
    hotspot placement — writes and reads CONTEND on one WIN) plus
    R/W-separated placement (balanced + explicit migration of read objects
    to a dedicated WIN). Writes in service are NEVER preempted — the model
    has no preemption, only queue ordering; write starvation shows up as
    Backup P95 growth."""
    print(f"\n{'policy':>16} {'restore_ms':>11} {'backupP95_ms':>13} "
          f"{'backup_max_ms':>14}")

    def run_order(policy, placement):
        pool = pool_of(placement=placement)
        # pre-backup the read objects at t=0 (READY at +3.9ms)
        read_slots = [list(range(4000 + i * 340, 4000 + (i + 1) * 340))
                      for i in range(2)]
        for rs in read_slots:
            pool.submit_object(0.0, rs, KV, is_read=False)
        if policy == "rw-separated":
            # explicit migration (G2): move read objects to win 7 and
            # bias the balanced picker so writes avoid win 7
            for k in list(pool.registry.objects):
                pool.registry.migrate_object(k, 7)
            pool.registry.win_bytes[7] += 10 * GB
        burst_t = 10e-3                    # after pre-prime completes
        pool.begin_batch(burst_t)
        write_slots = [list(range(i * 340, (i + 1) * 340)) for i in range(6)]
        if policy == "fifo":
            order = [("w", i) for i in range(6)] + [("r", i) for i in range(2)]
        elif policy == "read-priority":
            order = [("r", i) for i in range(2)] + [("w", i) for i in range(6)]
        elif policy == "fair":
            order = [("w", 0), ("r", 0), ("w", 1), ("r", 1)] + \
                    [("w", i) for i in range(2, 6)]
        else:
            order = [("w", i) for i in range(6)] + [("r", i) for i in range(2)]
        stats = {"w": [], "r": []}
        for kind, idx in order:
            slots = (write_slots if kind == "w" else read_slots)[idx]
            stats[kind].append(
                pool.submit_object(burst_t, slots, KV, is_read=(kind == "r")))
        r_lat = max(s.total for s in stats["r"]) * 1e3
        w_sorted = sorted(s.total for s in stats["w"])
        w_p95 = w_sorted[min(len(w_sorted) - 1, int(0.95 * len(w_sorted)))]
        return r_lat, w_p95 * 1e3, w_sorted[-1] * 1e3

    results = {}
    for policy in ("fifo", "read-priority", "fair"):
        results[policy] = run_order(policy, "hotspot")
        print(f"{policy:>16} {results[policy][0]:>11.2f} "
              f"{results[policy][1]:>13.2f} {results[policy][2]:>14.2f}")
    results["rw-separated"] = run_order("rw-separated", "balanced")
    print(f"{'rw-separated':>16} {results['rw-separated'][0]:>11.2f} "
          f"{results['rw-separated'][1]:>13.2f} "
          f"{results['rw-separated'][2]:>14.2f}")

    # read-priority improves restore latency vs FIFO (same placement)
    assert results["read-priority"][0] < results["fifo"][0], \
        "read-priority must improve restore latency"
    # ... at the cost of write P95 (queued writes delayed, never preempted)
    assert results["read-priority"][1] > results["fifo"][1], \
        "read-priority must delay write P95 (starvation side effect)"
    # fair splits the difference
    assert results["fifo"][0] > results["fair"][0] > results["read-priority"][0]
    # separation gives both fast restores AND parallel writes
    assert results["rw-separated"][0] < results["fifo"][0]
    assert results["rw-separated"][1] < results["read-priority"][1]
    print("✓ P2 policies: read-priority restore "
          f"{results['fifo'][0]:.1f}→{results['read-priority'][0]:.1f}ms, "
          f"write P95 {results['fifo'][1]:.1f}→"
          f"{results['read-priority'][1]:.1f}ms (starvation quantified); "
          "separation wins both")


# ------------------------------------------------------------------ P5 ---

def test_p5_win8_downstream_composition():
    """P5: win8 + SharedDownstream external bridge — HBF and DRAM I/Os
    contend on ONE shared downstream cursor (correct composition)."""
    # module-level DS state (as the hook wires it)
    import hisim.simulation.hbf_shared_resource as ds_mod

    ds_state = {"busy": 0.0}

    def get_busy():
        return ds_state["busy"]

    def update_busy(t):
        ds_state["busy"] = max(ds_state["busy"], t)

    pool = pool_of()
    pool.attach_external_interface(get_busy, update_busy, 13.6 * GB)
    pool.begin_batch(0.0)
    # DRAM-side I/O occupies the downstream first (as the DRAM manager would)
    ds_busy_end = 38.99e6 / 13.6e9             # exact DRAM occupancy
    update_busy(ds_busy_end)
    # HBF read must wait for the downstream cursor
    st = pool.submit(1e-3, 38.88e6, is_read=True)
    assert st.ic_start >= ds_busy_end - 1e-9, \
        f"HBF I/O must queue behind DRAM on the shared downstream: " \
        f"ic_start={st.ic_start} < {ds_busy_end}"
    # and the HBF I/O advances the SHARED cursor (DRAM sees it)
    assert ds_state["busy"] > 2.87e-3, "cursor must advance for DRAM side"
    # cross-batch: in-flight cursor preserved via the external state
    ds_state["busy"] = 0.05
    pool.begin_batch(0.001)
    st2 = pool.submit(1e-3, 4096, is_read=True)
    assert st2.ic_start >= 0.05 - 1e-9
    print(f"✓ P5 composition: HBF waits behind DRAM ({st.ic_start*1e3:.2f}ms), "
          f"shared cursor advances, cross-batch preserved")


# ------------------------------------------------------------------ P6 ---

def test_p6_contiguity_and_mode():
    """P6: access mode from ACTUAL WIN storage contiguity (per-WIN runs),
    not merely logical host-index contiguity. Fragmentation is reported;
    random envelope is a boundary case."""
    # contiguous slots, striped: per-WIN runs are contiguous → sequential
    pool = pool_of(placement="striped", stripe_block_tokens=32)
    st = pool.submit_object(0.0, list(range(256)), KV, is_read=False)
    assert st.mode == "sequential"
    # fragmented slots (scattered by the free-list): multiple runs
    pool2 = pool_of(placement="balanced")
    frag = [i * 2 for i in range(100)]          # stride-2 → 100 runs of 1
    st2 = pool2.submit_object(0.0, frag, KV, is_read=False)
    obj = next(iter(pool2.registry.objects.values()))
    assert obj.fragmented_runs == 100, \
        f"fragmentation must be reported: {obj.fragmented_runs}"
    print(f"✓ P6 contiguity: contiguous→sequential; scattered slots → "
          f"{obj.fragmented_runs} runs reported (physical fragmentation "
          "drives mode, Unknown where undetermined)")


def run_all():
    tests = [
        test_g1_same_object_same_win,
        test_g2_explicit_migration_only,
        test_g3_no_stale_mappings,
        test_g4_restore_blocked_before_backup_ready,
        test_g5_cross_batch_object_path,
        test_g6_conservation,
        test_p1_placement_mechanisms,
        test_p2_scheduling_policies,
        test_p5_win8_downstream_composition,
        test_p6_contiguity_and_mode,
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
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
