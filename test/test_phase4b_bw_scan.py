#!/usr/bin/env python3
"""Phase 4B: HBF Bandwidth Efficiency Scan.

Scans I/O Size × Queue Depth × Read/Write Ratio using the
page-level SharedHbfResourceManager model.

Definitions:
- Reference BW = 51.2 GB/s (simulated channel aggregate, NOT hardware measured)
- Media BW Efficiency = effective_media_bw / 51.2 GB/s
- Pooling Effective BW = total_bytes / total_time (includes queue wait)
- End-to-End Efficiency = effective_e2e_bw / min(media_bw, ic_bw)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager,
    configure_shared_downstream,
)

REF_BW = 51.2e9  # simulated reference (NOT hardware measured)

IO_SIZES = [
    ("4KB", 4096), ("16KB", 16384), ("64KB", 65536), ("256KB", 262144),
    ("1MB", 1048576), ("4MB", 4194304), ("16MB", 16777216),
]
QUEUE_DEPTHS = [1, 4, 16, 64]
RW_RATIOS = [("Read-only", 1.0), ("Balanced", 0.5), ("Write-heavy", 0.25)]


def run_single(io_bytes, qd, read_frac, n_io=32, ic_bw=51.2e9):
    """Run N I/Os with given parameters, return efficiency stats."""
    mgr = SharedHbfResourceManager()
    mgr.configure(
        media_bw=51.2e9, interconnect_bw=ic_bw, queue_depth=qd,
        page_bytes=16384, channels=32, planes_per_channel=32,
        read_latency=10e-6, write_latency=100e-6,
    )
    mgr.begin_batch(0.0)

    n_reads = int(n_io * read_frac)
    n_writes = n_io - n_reads

    # Submit reads first, then writes (same arrival = same batch)
    for i in range(n_reads):
        mgr.submit(0.0, io_bytes, is_read=True)
    for i in range(n_writes):
        mgr.submit(0.0, io_bytes, is_read=False)

    batch_dur = mgr.get_batch_duration()
    total_bytes = n_io * io_bytes

    # Pooling effective BW: total bytes / total time
    pool_bw = total_bytes / batch_dur if batch_dur > 0 else 0

    # Media efficiency: for a single I/O (no queue)
    single_media_time = mgr._media_service_time(io_bytes, is_read=True)
    single_media_bw = io_bytes / single_media_time if single_media_time > 0 else 0

    # End-to-end: max(media, ic) for single I/O
    single_ic_time = io_bytes / ic_bw if ic_bw > 0 else 0
    single_e2e = max(single_media_time, single_ic_time)
    e2e_bw = io_bytes / single_e2e if single_e2e > 0 else 0

    # Device utilization: how much of the batch was media busy
    total_media_busy = sum(s.media_time for s in mgr.io_stats)
    media_util = total_media_busy / batch_dur if batch_dur > 0 else 0

    # Queue wait stats
    max_qw = max(s.queue_wait for s in mgr.io_stats) if mgr.io_stats else 0

    return {
        "batch_dur": batch_dur,
        "pool_bw": pool_bw,
        "pool_eff": pool_bw / REF_BW,
        "single_media_bw": single_media_bw,
        "media_eff": single_media_bw / REF_BW,
        "e2e_bw": e2e_bw,
        "e2e_eff": e2e_bw / min(REF_BW, ic_bw),
        "media_util": media_util,
        "max_queue_wait": max_qw,
        "peak_outstanding": mgr.peak_outstanding,
    }


def run_scan():
    print("=" * 100)
    print("Phase 4B: HBF Bandwidth Efficiency Scan")
    print(f"Reference BW = {REF_BW/1e9:.1f} GB/s (SIMULATED, not hardware measured)")
    print("Page-level model: 16KB pages, 32 channels, 10μs read / 100μs write")
    print("=" * 100)

    for rw_label, read_frac in RW_RATIOS:
        print(f"\n--- {rw_label} (read_frac={read_frac}) ---")
        print(f"{'I/O Size':>8} {'QD':>4} {'Media Eff':>10} {'Pool Eff':>10} "
              f"{'E2E Eff':>10} {'Media Util':>11} {'Max QW':>10} {'Peak Out':>9}")
        print("-" * 90)

        for size_label, io_bytes in IO_SIZES:
            for qd in QUEUE_DEPTHS:
                r = run_single(io_bytes, qd, read_frac, n_io=32, ic_bw=51.2e9)
                print(f"{size_label:>8} {qd:>4} {r['media_eff']:>9.1%} {r['pool_eff']:>9.1%} "
                      f"{r['e2e_eff']:>9.1%} {r['media_util']:>10.1%} "
                      f"{r['max_queue_wait']*1e6:>8.0f}μs {r['peak_outstanding']:>9}")


def find_80pct_threshold():
    """Find I/O size where efficiency reaches 80%."""
    print("\n" + "=" * 100)
    print("80% Efficiency Threshold Analysis")
    print("=" * 100)

    print(f"\n{'I/O Size':>8} {'Media Eff (QD=1)':>16} {'Reaches 80%?':>12}")
    print("-" * 40)
    for size_label, io_bytes in IO_SIZES:
        r = run_single(io_bytes, 1, 1.0, n_io=1, ic_bw=51.2e9)
        reaches = "✅ YES" if r['media_eff'] >= 0.8 else "❌ NO"
        print(f"{size_label:>8} {r['media_eff']:>15.1%} {reaches:>12}")

    # Compute exact threshold
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=51.2e9, queue_depth=1)
    # For single page per channel: media_time = page_latency = 10μs
    # eff = bytes / 10μs = 0.8 × 51.2e9 → bytes = 409.6KB
    print(f"\nExact threshold: ~410KB (single page per channel, page latency bound)")
    print(f"  410KB / 10μs = 41 GB/s = 80.1% of 51.2 GB/s")
    print(f"  Below 512KB (32 pages × 32 channels): latency-bound")
    print(f"  Above 512KB: bandwidth-bound (100% efficiency)")

    # Write threshold
    print(f"\nWrite threshold (100μs page latency):")
    print(f"  ~4.1MB (40.96e9 × 100e-6 = 4.096MB)")
    r = run_single(4194304, 1, 0.0, n_io=1, ic_bw=51.2e9)  # 4MB write
    print(f"  4MB write efficiency: {r['media_eff']:.1%}")


def interconnect_bottleneck_analysis():
    """Analyze when interconnect becomes the bottleneck."""
    print("\n" + "=" * 100)
    print("Interconnect Bottleneck Analysis (Read, QD=1)")
    print("=" * 100)

    ic_bws = [8e9, 16e9, 32e9, 51.2e9]
    print(f"\n{'I/O Size':>8} {'Media BW':>10} ", end="")
    for ic in ic_bws:
        print(f"{'IC=' + str(int(ic/1e9)) + 'GB':>10}", end="")
    print()
    print("-" * 60)

    for size_label, io_bytes in IO_SIZES:
        mgr = SharedHbfResourceManager()
        mgr.configure(media_bw=51.2e9, interconnect_bw=51.2e9, queue_depth=1)
        mt = mgr._media_service_time(io_bytes, is_read=True)
        media_bw = io_bytes / mt if mt > 0 else 0
        print(f"{size_label:>8} {media_bw/1e9:>8.1f}GB/s ", end="")
        for ic in ic_bws:
            ic_time = io_bytes / ic
            e2e_time = max(mt, ic_time)
            e2e_bw = io_bytes / e2e_time if e2e_time > 0 else 0
            bottleneck = "M" if mt >= ic_time else "I"
            print(f"{e2e_bw/1e9:>8.1f}({bottleneck})", end="")
        print()
    print("\n(M = media bottleneck, I = interconnect bottleneck)")


if __name__ == "__main__":
    run_scan()
    find_80pct_threshold()
    interconnect_bottleneck_analysis()
