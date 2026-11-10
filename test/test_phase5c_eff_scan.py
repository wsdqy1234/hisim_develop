#!/usr/bin/env python3
"""Phase 5C: 8-WIN pooled bandwidth efficiency scan + loss attribution.

P0 metric definitions (dual, always reported):
  - Fixed Pool Efficiency  = Useful BW / (8 × per-WIN rated BW)   [PRIMARY]
  - Active WIN Efficiency  = Useful BW / (active WIN count × rated)
  - WIN Utilization (media busy / span) + Load Balance (min/max per-WIN bytes)
  - Interface Delivered BW / Interface Rated BW
  Read/write reported SEPARATELY for mixed workloads (no artificial single
  denominator). Seq/rand/write denominators: 1600/800/80 GB/s.

Saturation model: closed-loop QD (exactly QD I/Os in flight; a new I/O is
submitted at the completion of a departed one) — sustained demand, all
completed bytes / simulation window.

Layers: A = expert Service Envelope (upper bound), B = assumed geometry
(HBFSim nominal placeholder — NOT calibrated, not tuned to 80%).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel, WinSpec

GB = 1e9
KV = 114688
OUT = "/tmp/p5c_results"
os.makedirs(OUT, exist_ok=True)

SEQ_CAP, RAND_CAP, WR_CAP = 1600 * GB, 800 * GB, 80 * GB


# ---------------------------------------------------------------------------
# P0: closed-loop saturation harness + dual metrics
# ---------------------------------------------------------------------------

def run_saturation(cfg, n_total, qd, ops, active=None, pin=None):
    """Closed-loop QD saturation. ops = list of (bytes, is_read, mode)
    cycled to n_total submissions. Returns (pool, stats, span)."""
    pool = HbfWinPoolModel(cfg)
    if active:
        pool.set_active_wins(active)
    if pin:
        for oid, win in pin.items():
            pool._objects[oid] = win
    pool.begin_batch(0.0)
    stats = []

    def submit_one(t, oid):
        b, rd, mode = ops[oid % len(ops)]
        st = pool.submit(t, b, is_read=rd, access_mode=mode, object_id=oid)
        stats.append(st)
        return (max(st.media_completion, st.ic_completion), st)

    inflight = []
    n = 0
    for _ in range(min(qd, n_total)):
        inflight.append(submit_one(0.0, n))
        n += 1
    while n < n_total:
        i = min(range(len(inflight)), key=lambda k: inflight[k][0])
        t = inflight[i][0]
        inflight[i] = submit_one(t, n)
        n += 1
    span = pool.get_batch_duration()
    return pool, stats, span


def p0_metrics(pool, stats, span, mode):
    """Dual efficiency metrics. mode: 'seq' | 'rand' | 'write' selects the
    rated denominator; mixed workloads call this per mode separately."""
    rated = {"seq": pool.config.win.seq_read_bw,
             "rand": pool.config.win.rand_read_bw,
             "write": pool.config.win.write_bw}[mode]
    is_read = mode != "write"
    io_mode = {"seq": "sequential", "rand": "random", "write": None}[mode]
    tot = sum(s.bytes for s in stats
              if s.is_read == is_read and
              (mode == "write" or s.mode == io_mode))
    bw = tot / span if span > 0 else 0.0
    n_wins = len(pool.wins)
    active = pool.active_win_count or 1
    fixed_eff = bw / (n_wins * rated)
    active_eff = bw / (active * rated)
    # per-WIN utilization (Layer A: service sum / span; Layer B: channel
    # data-path occupancy via PageSim) & load balance over ALL 8 WINs
    t0 = pool.batch_start
    utils = []
    for w in pool.wins:
        if w.pagesim is not None:
            utils.append(w.pagesim.device_utilization(t0, t0 + span)
                         if span > 0 else 0)
        else:
            utils.append(w.media_service_sum / span if span > 0 else 0)
    per_win_bytes = [w.bytes_read_seq + w.bytes_read_rand + w.bytes_written
                     for w in pool.wins]
    balance = (min(per_win_bytes) / max(per_win_bytes)
               if max(per_win_bytes) > 0 else 1.0)
    # interface metrics
    ic_stats = [s for s in stats if s.ic_time > 0]
    if ic_stats:
        ic_busy = (max(s.ic_completion for s in ic_stats)
                   - min(s.ic_start for s in ic_stats))
        ic_total = sum(s.bytes for s in ic_stats)
        ic_deliv = ic_total / ic_busy if ic_busy > 0 else 0.0
        ic_rated = (pool.config.package_shared_interface_bw
                    or pool.config.per_win_interface_bw or 0)
        ic_util = ic_busy / span if span > 0 else 0.0
    else:
        ic_deliv = ic_rated = ic_util = 0.0
    qw_max = max((s.queue_wait for s in stats), default=0.0)
    constrained = sum(1 for s in stats if s.interface_constrained)
    return {
        "bw_gbps": round(bw / GB, 2), "fixed_eff": round(fixed_eff, 4),
        "active_eff": round(active_eff, 4), "active_wins": active,
        "win_util_max": round(max(utils), 3) if utils else 0,
        "win_util_avg": round(sum(utils) / n_wins, 3),
        "load_balance(min/max)": round(balance, 3),
        "ic_delivered_gbps": round(ic_deliv / GB, 2),
        "ic_util": round(ic_util, 3), "ic_rated_gbps": ic_rated / GB
        if ic_rated else 0,
        "ic_constrained_pct": round(constrained / len(stats) * 100, 1)
        if stats else 0,
        "qw_max_us": round(qw_max * 1e6, 1),
        "span_ms": round(span * 1e3, 3),
    }


# ---------------------------------------------------------------------------
# P2: single-WIN I/O efficiency scan
# ---------------------------------------------------------------------------

P2_SIZES = [("4KB", 4096), ("16KB", 16384), ("64KB", 65536),
            ("256KB", 262144), ("1MB", 1048576), ("4MB", 4194304),
            ("16MB", 16777216), ("38.9MB", 339 * KV), ("64MB", 64 * 1024 * 1024)]
P2_QDS = [1, 4, 8, 16, 32, 64]
P2_MODES = [("seq", True, "sequential"), ("rand", True, "random"),
            ("write", False, None)]
P2_LATS = [5.0, 7.5, 10.0]


def p2_single_win_scan():
    print("\n" + "=" * 96)
    print("P2: SINGLE-WIN I/O efficiency (n_wins=1; fixed = active denominator)")
    print("=" * 96)
    rows = []
    for layer in ("envelope", "pagesim"):
        print(f"\n--- Layer {'A (envelope, expert spec)' if layer == 'envelope' else 'B (ASSUMED geometry — HBFSim nominal placeholder)'} ---")
        for mode_key, is_read, mode in P2_MODES:
            print(f"\n[{mode_key}]  eff vs size × QD (lat=7.5μs):")
            print(f"{'size':>7} " + " ".join(f"QD{q:<6}" for q in P2_QDS))
            for label, size in P2_SIZES:
                cells = []
                for qd in P2_QDS:
                    cfg = Hbf8WinConfig(
                        n_wins=1, layer=layer,
                        win=WinSpec(read_latency_s=7.5e-6))
                    pool, stats, span = run_saturation(
                        cfg, n_total=max(64, 2 * qd), qd=qd,
                        ops=[(size, is_read, mode)])
                    m = p0_metrics(pool, stats, span, mode_key)
                    cells.append(f"{m['fixed_eff']:.3f}")
                    rows.append({
                        "layer": layer, "mode": mode_key, "lat_us": 7.5,
                        "size": label, "qd": qd,
                        "bw_gbps": m["bw_gbps"], "eff": m["fixed_eff"],
                        "qw_max_us": m["qw_max_us"],
                    })
                print(f"{label:>7} " + " ".join(f"{c:<8}" for c in cells))

    # read-latency sweep (Layer A, seq read, QD=1)
    print("\n[latency sweep] seq-read single WIN, QD=1 (Layer A):")
    print(f"{'size':>7} {'lat5μs':>8} {'lat7.5μs':>9} {'lat10μs':>9}")
    for label, size in P2_SIZES:
        cells = []
        for lat in P2_LATS:
            cfg = Hbf8WinConfig(
                n_wins=1, win=WinSpec(read_latency_s=lat * 1e-6))
            pool, stats, span = run_saturation(
                cfg, n_total=16, qd=1, ops=[(size, True, "sequential")])
            m = p0_metrics(pool, stats, span, "seq")
            cells.append(f"{m['fixed_eff']:.3f}")
            rows.append({"layer": "envelope", "mode": "seq",
                         "lat_us": lat, "size": label, "qd": 1,
                         "bw_gbps": m["bw_gbps"], "eff": m["fixed_eff"],
                         "qw_max_us": m["qw_max_us"]})
        print(f"{label:>7} " + " ".join(f"{c:>9}" for c in cells))

    # 80% working points (Layer A)
    print("\n80% single-WIN working points (Layer A, QD=1):")
    for lat in P2_LATS:
        thr = 0.8 * lat * 1e-6 * 200e9  # bytes for 80% at this latency
        print(f"  lat={lat}μs: I/O ≥ {thr/1e6:.2f}MB (200GB/s×{lat}μs×0.8)")
    write_csv(f"{OUT}/p2_single_win.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# P3: 8-WIN pool scaling & load balance
# ---------------------------------------------------------------------------

def p3_pool_scaling():
    print("\n" + "=" * 96)
    print("P3: 8-WIN pool scaling (pool FIXED at 8 WINs; active routing varies)")
    print("=" * 96)
    rows = []
    io = 339 * KV
    print(f"\n--- seq read 38.88MB, closed-loop QD=64 ---")
    print(f"{'placement':>10} {'active':>7} {'readGB/s':>9} {'fixedEff':>9} "
          f"{'activeEff':>10} {'balance':>8} {'utilMax':>8}")
    for placement in ("balanced", "hotspot"):
        for active in (1, 2, 4, 8):
            cfg = Hbf8WinConfig(placement=placement)
            if placement == "hotspot":
                cfg.placement = "hotspot"
            pool, stats, span = run_saturation(
                cfg, n_total=64, qd=64,
                ops=[(io, True, "sequential")], active=active)
            m = p0_metrics(pool, stats, span, "seq")
            print(f"{placement:>10} {active:>7} {m['bw_gbps']:>9.1f} "
                  f"{m['fixed_eff']:>9.3f} {m['active_eff']:>10.3f} "
                  f"{m['load_balance(min/max)']:>8.3f} {m['win_util_max']:>8.3f}")
            rows.append({"placement": placement, "active_wins": active,
                         "read_gbps": m["bw_gbps"],
                         "fixed_eff": m["fixed_eff"],
                         "active_eff": m["active_eff"],
                         "balance": m["load_balance(min/max)"],
                         "qw_max_us": m["qw_max_us"]})

    # 80% aggregate targets check
    print("\n--- 80% aggregate targets (balanced, 8 active, QD=64) ---")
    for mode_key, ops, cap, target in (
            ("seq", [(339 * KV, True, "sequential")], 1600, 1280),
            ("rand", [(339 * KV, True, "random")], 800, 640),
            ("write", [(340 * KV, False, None)], 80, 64)):
        pool, stats, span = run_saturation(
            Hbf8WinConfig(), n_total=64, qd=64, ops=ops)
        m = p0_metrics(pool, stats, span, mode_key)
        ok = "✅" if m["bw_gbps"] >= target * 0.999 else "❌"
        print(f"  {mode_key:>6}: {m['bw_gbps']:>7.1f} GB/s "
              f"(cap {cap}, 80% target {target}) {ok} "
              f"fixedEff={m['fixed_eff']:.3f}")
        rows.append({"placement": "balanced", "active_wins": 8,
                     f"{mode_key}_gbps": m["bw_gbps"],
                     "fixed_eff": m["fixed_eff"]})

    # page striping analytic upper bound
    print("\n--- Page-striping upper bound (analytic, one logical I/O across 8 WINs) ---")
    for label, size in (("38.9MB", 339 * KV), ("16MB", 16 << 20),
                        ("4MB", 4 << 20)):
        t = max(7.5e-6, size / (8 * 200e9))
        print(f"  {label:>7} striped: {t*1e6:8.1f}μs → "
              f"{size/t/GB:7.1f} GB/s ({size/t/SEQ_CAP:.1%} fixed-pool eff)")
    print("  (single-WIN I/O of same size: "
          f"{(339*KV)/max(7.5e-6, 339*KV/200e9)/GB:.0f} GB/s — striping is the "
          "analytic upper bound; NOT simulated as a placement policy)")
    write_csv(f"{OUT}/p3_pool_scaling.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# P4: UB interface assumptions & read/write contention
# ---------------------------------------------------------------------------

IFACES = [
    ("none", {}),
    ("shared112", {"package_shared_interface_bw": 112 * GB}),
    ("perwin112", {"per_win_interface_bw": 112 * GB}),
    ("shared896", {"package_shared_interface_bw": 896 * GB}),
]


def p4_interface_scan():
    print("\n" + "=" * 96)
    print("P4: UB interface assumptions (ALL are sensitivity assumptions — "
          "real HW unresolved)")
    print("=" * 96)
    rows = []
    io_r, io_w = 339 * KV, 340 * KV
    workloads = [
        ("R100", [(io_r, True, "sequential")], SEQ_CAP),
        ("W100", [(io_w, False, None)], WR_CAP),
        ("R50W50", [(io_r, True, "sequential"), (io_w, False, None)], None),
    ]
    for wl_name, ops, cap in workloads:
        print(f"\n--- {wl_name} (8 WINs, balanced, QD=32) ---")
        print(f"{'interface':>10} {'boundGB/s':>10} {'readGB/s':>9} "
              f"{'writeGB/s':>9} {'ICdeliv':>8} {'ICutil':>7} "
              f"{'constr%':>8} {'qwMaxμs':>9}")
        for name, kw in IFACES:
            cfg = Hbf8WinConfig(**kw)
            pool, stats, span = run_saturation(cfg, n_total=64, qd=32, ops=ops)
            mr = p0_metrics(pool, stats, span, "seq")
            mw = p0_metrics(pool, stats, span, "write")
            ic_rated = (kw.get("package_shared_interface_bw")
                        or kw.get("per_win_interface_bw"))
            if ic_rated is None:
                bound = SEQ_CAP if wl_name == "R100" else WR_CAP
            else:
                agg_ic = (ic_rated * 8 if "per_win" in kw or name == "perwin112"
                          and False else ic_rated)
                # per-WIN interface caps each WIN's delivery but reads are
                # 200GB/s > 112 → per-WIN IC bound = 8×112 for the pool
                agg_ic = ic_rated * 8 if kw.get("per_win_interface_bw") \
                    else ic_rated
                media_cap = SEQ_CAP if wl_name == "R100" else WR_CAP
                bound = min(media_cap, agg_ic)
            print(f"{name:>10} {bound/GB:>10.0f} {mr['bw_gbps']:>9.1f} "
                  f"{mw['bw_gbps']:>9.1f} {mr['ic_delivered_gbps']:>8.1f} "
                  f"{mr['ic_util']:>7.3f} {mr['ic_constrained_pct']:>8.1f} "
                  f"{mr['qw_max_us']:>9.1f}")
            rows.append({"workload": wl_name, "interface": name,
                         "bound_gbps": round(bound / GB, 1),
                         "read_gbps": mr["bw_gbps"],
                         "write_gbps": mw["bw_gbps"],
                         "ic_delivered": mr["ic_delivered_gbps"],
                         "ic_util": mr["ic_util"],
                         "constrained_pct": mr["ic_constrained_pct"],
                         "qw_max_us": mr["qw_max_us"]})

    # write burst with concurrent restore (KV-realistic mixed scenario)
    print("\n--- Write burst + concurrent restore (6W + 2R at t=0, 8 WINs) ---")
    for name, kw in IFACES:
        cfg = Hbf8WinConfig(placement="balanced", **kw)
        pool = HbfWinPoolModel(cfg)
        pool.begin_batch(0.0)
        # interleave: R W W W R W W W W W
        order = [(io_r, True, "sequential"), (io_w, False, None)] * 1 + \
                [(io_w, False, None)] * 4 + [(io_r, True, "sequential")] + \
                [(io_w, False, None)] * 1
        st = []
        for i, (b, rd, mode) in enumerate(order[:8]):
            st.append(pool.submit(0.0, b, is_read=rd, access_mode=mode,
                                  object_id=i))
        span = pool.get_batch_duration()
        reads = [s for s in st if s.is_read]
        read_lat = max(s.total for s in reads) * 1e3
        mr = p0_metrics(pool, st, span, "seq")
        print(f"  {name:>10}: span={span*1e3:7.2f}ms  restore latency="
              f"{read_lat:7.2f}ms (no-write baseline 0.19ms)  "
              f"ICconstr={mr['ic_constrained_pct']:.0f}%  qw={mr['qw_max_us']:.0f}μs")
        rows.append({"workload": "burstW+R", "interface": name,
                     "span_ms": round(span * 1e3, 3),
                     "restore_latency_ms": round(read_lat, 3),
                     "constrained_pct": mr["ic_constrained_pct"]})

    print("\nContention levels distinguished:")
    print("  WIN-internal: R/W on same WIN → media serial (write 3.9ms blocks)")
    print("  Package interface: shared cursor across WINs (half-duplex)")
    print("  Downstream: OUT OF SCOPE here (DRAM-side, Phase 3E topology)")
    write_csv(f"{OUT}/p4_interface.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# P5: loss attribution (same trace, incremental chain)
# ---------------------------------------------------------------------------

def p5_waterfall():
    print("\n" + "=" * 96)
    print("P5: loss attribution — same trace, base chain + mechanism branches")
    print("=" * 96)
    ops_kv = [(339 * KV, True, "sequential")] * 8 + \
             [(340 * KV, False, None)] * 4
    ops_small = [(65536, True, "sequential")]

    def burst_run(cfg, ops, n_total):
        """Open-loop burst: all I/Os arrive at t=0 (saturated device)."""
        pool = HbfWinPoolModel(cfg)
        pool.begin_batch(0.0)
        stats = []
        for i in range(n_total):
            b, rd, mode = ops[i % len(ops)]
            stats.append(pool.submit(0.0, b, is_read=rd,
                                     access_mode=mode, object_id=i))
        return pool, stats, pool.get_batch_duration()

    def ideal_span(cfg, ops, n_total):
        """Analytic work-conserving lower bound (incl. latency floors)."""
        w = cfg.win
        work = 0.0
        for i in range(n_total):
            b, rd, mode = ops[i % len(ops)]
            if rd:
                bw = w.seq_read_bw if mode == "sequential" else w.rand_read_bw
                lat = w.read_latency_s
            else:
                bw, lat = w.write_bw, w.write_latency_s
            work += max(lat, b / bw)
        return work / len(pool_wins(cfg))

    def pool_wins(cfg):
        return list(range(cfg.n_wins))

    def report(trace_name, ops, n_total, qd_closed):
        cfg = Hbf8WinConfig()
        rows = []
        levels = []
        c0 = ideal_span(cfg, ops, n_total)
        levels.append(("C0 analytic ideal (work-conserving)", None, None, c0))
        p, s, sp = burst_run(cfg, ops, n_total)
        levels.append(("C1 burst @t=0, balanced (packing loss)", p, s, sp))
        p, s, sp = run_saturation(cfg, n_total=n_total, qd=qd_closed, ops=ops)
        levels.append((f"C2 closed-loop QD={qd_closed} (slot coupling)",
                       p, s, sp))
        base = levels[-1]
        # branches FROM C2 (alternatives, no double counting):
        p, s, sp = run_saturation(
            Hbf8WinConfig(placement="hotspot"), n_total=n_total,
            qd=qd_closed, ops=ops)
        levels.append(("C2a + hotspot@win0 (placement loss)", p, s, sp))
        p, s, sp = run_saturation(
            Hbf8WinConfig(package_shared_interface_bw=112 * GB),
            n_total=n_total, qd=qd_closed, ops=ops)
        levels.append(("C2b + shared interface 112 (interface loss)",
                       p, s, sp))

        print(f"\n--- {trace_name} ---")
        print(f"{'level':>48} {'span_ms':>9} {'readGB/s':>9} "
              f"{'writeGB/s':>9} {'fixedEffR':>10} {'fixedEffW':>10}")
        c2_eff = None
        for name, pool, stats, span in levels:
            if stats is None:
                print(f"{name:>48} {span*1e3:>9.3f} {'—':>9} {'—':>9} "
                      f"{'—':>10} {'—':>10}")
                rows.append({"trace": trace_name, "level": name,
                             "span_ms": round(span * 1e3, 3)})
                continue
            mr = p0_metrics(pool, stats, span, "seq")
            mw = p0_metrics(pool, stats, span, "write")
            print(f"{name:>48} {span*1e3:>9.3f} {mr['bw_gbps']:>9.1f} "
                  f"{mw['bw_gbps']:>9.1f} {mr['fixed_eff']:>10.3f} "
                  f"{mw['fixed_eff']:>10.3f}")
            rows.append({"trace": trace_name, "level": name,
                         "span_ms": round(span * 1e3, 3),
                         "read_gbps": mr["bw_gbps"],
                         "write_gbps": mw["bw_gbps"],
                         "fixed_eff_read": mr["fixed_eff"],
                         "fixed_eff_write": mw["fixed_eff"]})
        return rows

    rows = []
    rows += report("KV trace (48 I/Os = 32R+16W, 38.9MB)",
                   ops_kv, n_total=48, qd_closed=8)
    rows += report("Small trace (128×64KB R)",
                   ops_small, n_total=128, qd_closed=8)

    # R/W contention pairwise (head-of-line blocking, same window)
    print("\n--- R/W contention pairwise (1 write + 2 reads, 8 WINs, t=0) ---")
    io_r, io_w = 339 * KV, 340 * KV
    for label, pin in (("reads pinned to FREE WINs", {1: 5, 2: 6}),
                       ("reads pinned to the WRITE's WIN", {1: 0, 2: 0})):
        pool = HbfWinPoolModel(Hbf8WinConfig())
        pool._objects[0] = 0           # the write on win 0
        for oid, win in pin.items():
            pool._objects[oid] = win
        pool.begin_batch(0.0)
        pool.submit(0.0, io_w, is_read=False, object_id=0)
        r1 = pool.submit(0.0, io_r, is_read=True, object_id=1)
        r2 = pool.submit(0.0, io_r, is_read=True, object_id=2)
        lat = max(r1.total, r2.total) * 1e3
        print(f"  {label:>36}: restore latency = {lat:6.2f}ms")
        rows.append({"trace": "R/W contention pairwise", "level": label,
                     "restore_latency_ms": round(lat, 3)})

    # Layer B mechanism toggles (span deltas; ASSUMED geometry)
    print("\n--- Layer B mechanism toggles (32×38.88MB reads, ASSUMED "
          "geometry) ---")
    print(f"{'variant':>34} {'span_ms':>9} {'Δ vs full':>10}")
    base_span = None
    variants = [
        ("full pagesim (assumed geom)", {}),
        ("page latency OFF", {"enable_page_latency": False}),
        ("plane contention OFF", {"enable_plane_contention": False}),
        ("channel transfer OFF*", {"enable_channel_transfer": False}),
    ]
    for name, sw in variants:
        pool = HbfWinPoolModel(Hbf8WinConfig(layer="pagesim"))
        for w in pool.wins:
            for k, v in sw.items():
                setattr(w.pagesim, k, v)
        pool.begin_batch(0.0)
        stats = []
        for i in range(32):
            stats.append(pool.submit(0.0, 339 * KV, is_read=True,
                                     object_id=i))
        span = pool.get_batch_duration()
        if base_span is None:
            base_span = span
        m = p0_metrics(pool, stats, span, "seq")
        print(f"{name:>34} {span*1e3:>9.3f} "
              f"{(span-base_span)*1e3:>+10.3f}  ({m['bw_gbps']:.1f}GB/s)")
        rows.append({"trace": "KV-read layerB", "level": name,
                     "span_ms": round(span * 1e3, 3),
                     "read_gbps": m["bw_gbps"]})
    print("  *channel transfer OFF removes the data-path bound entirely —")
    print("   NOT a valid efficiency point; shown to identify the channel")
    print("   data path as the binding resource at this (assumed) geometry.")

    print("\nUNMODELED (Unknown — no numeric loss assigned): FTL, ECC, GC,")
    print("read disturb, write amplification, P/E endurance effects.")
    write_csv(f"{OUT}/p5_waterfall.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# P6: Qwen3 KV I/O feature replay
# ---------------------------------------------------------------------------

def p6_qwen3_features():
    print("\n" + "=" * 96)
    print("P6: Qwen3 KV I/O feature replay (standalone microbench — NOT "
          "end-to-end benefit)")
    print("=" * 96)
    prefixes = [
        ("340tk (agent prefix)", 340 * KV),
        ("4Ktk (long prompt)", 4096 * KV),
        ("16Ktk (long context)", 16384 * KV),
    ]
    rows = []
    print(f"\n{'prefix':>22} {'IOMB':>7} {'mode':>6} {'QD':>4} {'GB/s':>8} "
          f"{'fixedEff':>9} {'lat(ms)':>9}")
    for label, size in prefixes:
        for mode_key, is_read, mode in (("seq", True, "sequential"),
                                        ("rand", True, "random"),
                                        ("write", False, None)):
            for qd in (1, 8):
                pool, stats, span = run_saturation(
                    Hbf8WinConfig(), n_total=max(16, 2 * qd), qd=qd,
                    ops=[(size, is_read, mode)])
                m = p0_metrics(pool, stats, span, mode_key)
                per_io = max(s.total for s in stats) * 1e3
                print(f"{label:>22} {size/1e6:>7.1f} {mode_key:>6} {qd:>4} "
                      f"{m['bw_gbps']:>8.1f} {m['fixed_eff']:>9.3f} "
                      f"{per_io:>9.2f}")
                rows.append({"prefix": label, "io_mb": round(size / 1e6, 1),
                             "mode": mode_key, "qd": qd,
                             "bw_gbps": m["bw_gbps"],
                             "fixed_eff": m["fixed_eff"],
                             "io_latency_ms": round(per_io, 3)})
    write_csv(f"{OUT}/p6_qwen3.csv", rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


def main():
    print("Phase 5C scan — Layer A = expert envelope; Layer B = ASSUMED "
          "geometry (NOT calibrated, NOT tuned)")
    p2_single_win_scan()
    p3_pool_scaling()
    p4_interface_scan()
    p5_waterfall()
    p6_qwen3_features()
    print("\n" + "=" * 96)
    print(f"All CSVs → {OUT}/")


if __name__ == "__main__":
    main()
