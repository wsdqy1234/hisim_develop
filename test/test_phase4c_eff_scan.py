#!/usr/bin/env python3
"""Phase 4-Closure P0/P1: Corrected bandwidth-efficiency scan.

Distinguishes:
  - Media BW efficiency:       single-I/O useful BW / configured media BW
  - Pool useful BW efficiency: batch total bytes / makespan / media BW
                               (includes queue + concurrency effects)
  - Interconnect efficiency:   bytes / ic_time / configured IC BW
  - E2E efficiency:            bytes / max(media, ic) / min(media, ic)

Models:
  - "formula": closed-form max(transfer, page_latency) — Theory Bound
  - "pagesim": discrete-event page/plane/channel — Model Behavior

Efficiency configurations (P0 — three INDEPENDENT experiments):
  nominal : media=51.2, IC=16   (reference)
  EFF-M   : media=40.96, IC=16  (media 80% — the correct "media efficiency")
  EFF-I   : media=51.2, IC=12.8 (interconnect 80%)
  EFF-MI  : media=40.96, IC=12.8 (both 80%)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager,
    configure_shared_downstream,
)
import hisim.simulation.hbf_shared_resource as ds_mod

IO_SIZES = [
    ("4KB", 4096), ("16KB", 16384), ("64KB", 65536), ("256KB", 262144),
    ("1MB", 1048576), ("4MB", 4194304), ("16MB", 16777216),
    ("38.9MB", 339 * 114688),
]
KV = 114688

EFF_CONFIGS = [
    ("nominal", 51.2e9, 16e9),
    ("EFF-M", 40.96e9, 16e9),
    ("EFF-I", 51.2e9, 12.8e9),
    ("EFF-MI", 40.96e9, 12.8e9),
]


def run_batch(io_bytes, qd, read_frac, media_bw, ic_bw, media_model, n=1):
    """Submit n I/Os (concurrent burst, QD=n) through the production manager."""
    ds_mod._shared_downstream_bw = 0  # independent topology
    ds_mod._downstream_busy_until = 0
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=media_bw, interconnect_bw=ic_bw,
                  queue_depth=128, media_model=media_model)
    mgr.begin_batch(0.0)

    n_reads = int(n * read_frac)
    n_writes = n - n_reads
    for _ in range(n_reads):
        mgr.submit(0.0, io_bytes, is_read=True)
    for _ in range(n_writes):
        mgr.submit(0.0, io_bytes, is_read=False)

    total = n * io_bytes
    span = mgr.get_batch_duration()

    # Single-I/O media service time (no contention)
    single = mgr._media_service_time(io_bytes, True) \
        if media_model == "formula" else None
    if media_model == "pagesim":
        from hisim.simulation.hbf_page_sim import PageLevelMediaSim
        sim = PageLevelMediaSim(32, 32, 16384, 10e-6, 100e-6,
                                media_bw / 32)
        single = sim.submit(0.0, io_bytes, is_read=True).media_time

    media_eff = (io_bytes / single / media_bw) if single else 0
    ic_time = io_bytes / ic_bw
    ic_eff = (io_bytes / ic_time / ic_bw) if ic_time else 0
    pool_bw = total / span if span > 0 else 0
    pool_eff = pool_bw / media_bw
    e2e_time = max(single, ic_time)
    e2e_eff = (io_bytes / e2e_time) / min(media_bw, ic_bw) if e2e_time else 0
    bottleneck = "media" if single >= ic_time else "interconnect"
    max_qw = max(s.queue_wait for s in mgr.io_stats) if mgr.io_stats else 0

    return {
        "media_eff": media_eff, "ic_eff": ic_eff, "pool_eff": pool_eff,
        "e2e_eff": e2e_eff, "bottleneck": bottleneck,
        "span_ms": span * 1e3, "max_qw_us": max_qw * 1e6,
    }


def scan_model(media_model):
    print(f"\n{'='*100}")
    print(f"MODEL = {media_model} "
          f"({'Theory Bound (closed-form)' if media_model == 'formula' else 'Model Behavior (discrete-event)'})")
    print(f"{'='*100}")

    for cfg_name, media_bw, ic_bw in EFF_CONFIGS:
        ref = media_bw
        print(f"\n--- {cfg_name}: media={media_bw/1e9:.2f}GB/s, "
              f"IC={ic_bw/1e9:.2f}GB/s ---")
        print(f"{'I/O':>7} {'QD':>4} {'RW':>7} {'MediaEff':>9} "
              f"{'PoolEff':>9} {'E2EEff':>8} {'Bottleneck':>11} {'MaxQW':>8}")
        for size_label, io_bytes in IO_SIZES:
            for qd in (1, 16):
                for rw_label, rf in (("R100", 1.0), ("W100", 0.0), ("R50W50", 0.5)):
                    r = run_batch(io_bytes, qd, rf, media_bw, ic_bw,
                                  media_model, n=qd)
                    print(f"{size_label:>7} {qd:>4} {rw_label:>7} "
                          f"{r['media_eff']:>8.1%} {r['pool_eff']:>8.1%} "
                          f"{r['e2e_eff']:>7.1%} {r['bottleneck']:>11} "
                          f"{r['max_qw_us']:>7.0f}μs")


def threshold_analysis():
    print(f"\n{'='*100}")
    print("80% Efficiency Threshold (read, QD=1, pagesim vs formula)")
    print(f"{'='*100}")
    from hisim.simulation.hbf_page_sim import PageLevelMediaSim
    for cfg_name, media_bw, ic_bw in EFF_CONFIGS:
        print(f"\n--- {cfg_name} (media={media_bw/1e9:.2f}, IC={ic_bw/1e9:.2f}) ---")
        print(f"{'I/O':>7} {'formula':>9} {'pagesim':>9} {'≥80%?':>7}")
        for size_label, io_bytes in IO_SIZES:
            mgr = SharedHbfResourceManager()
            mgr.configure(media_bw=media_bw, interconnect_bw=ic_bw)
            f_eff = io_bytes / mgr._media_service_time(io_bytes, True) / media_bw
            sim = PageLevelMediaSim(32, 32, 16384, 10e-6, 100e-6,
                                    media_bw / 32)
            mt = sim.submit(0.0, io_bytes, is_read=True).media_time
            p_eff = io_bytes / mt / media_bw
            ok = "✅" if p_eff >= 0.8 else "❌"
            print(f"{size_label:>7} {f_eff:>8.1%} {p_eff:>8.1%} {ok:>7}")

    print("\nKey: formula (Theory Bound) overstates efficiency for I/Os")
    print("below ~1MB (hides initial tR exposure and partial-channel use).")
    print("pagesim thresholds: READ ~1MB@QD=1 (burst), WRITE ~4MB+ (trailing tW).")


if __name__ == "__main__":
    scan_model("formula")
    scan_model("pagesim")
    threshold_analysis()
