#!/usr/bin/env python3
"""Phase 4-Closure P2: Loss attribution waterfall.

Runs the SAME byte sequence / I/O pattern through the page-level
simulator with mechanisms toggled one at a time (additive waterfall),
measuring each mechanism's effect on Useful BW and latency. Every number
is MEASURED — no hand-assigned percentages.

Arrival semantics (fixed per trace, identical at every level):
  - "burst": all I/Os arrive at t=0 (concurrent submission, QD = N)
  - "closed": closed-loop QD=1 — I/O i arrives when I/O i-1 completes
    (arrival times depend on the level's own completions, which is the
    definition of closed-loop; byte sequence identical at all levels)

Levels (additive mechanisms):
  L0 Ideal          — analytic span = total_bytes / aggregate BW → 100%
  L1 +ChannelPath   — pagesim, page_latency=0, plane contention off:
                      per-channel data-path serialization; exposes
                      partial-channel use for small I/O
  L2 +PageLatency   — tR/tW enabled (pipelined with transfers; read tR
                      is exposed once per burst, write tW trails)
  L3 +PlaneContent  — finite planes (plane busy conflicts)

NOT MODELED (Unknown — no numeric loss assigned):
  FTL, ECC, GC, read disturb, write amplification, die-enable bus
  sharing beyond channel data-path serialization. These require
  MQSim/hardware calibration and are NOT counted in the waterfall.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_page_sim import PageLevelMediaSim

REF = 51.2e9   # simulated channel aggregate (NOT hardware measured)
PAGE = 16384
TR = 10e-6
TW = 100e-6
KV = 114688    # bytes/token, Qwen3-0.6B


def run_level(io_list, level, mode):
    """Run the trace at a waterfall level. Returns (useful_bw, span)."""
    total = sum(b for b, _ in io_list)
    if level == 0:
        span = total / REF
        return total / span, span

    sim = PageLevelMediaSim(channels=32, planes_per_channel=32,
                            page_bytes=PAGE, read_latency=TR,
                            write_latency=TW, channel_rate=REF / 32)
    sim.enable_page_latency = (level >= 2)
    sim.enable_plane_contention = (level >= 3)

    arrival = 0.0
    last = 0.0
    for b, rd in io_list:
        s = sim.submit(arrival, b, is_read=rd)
        last = max(last, s.media_completion)
        if mode == "closed":
            arrival = last
    span = last
    return (total / span if span > 0 else 0.0), span


def waterfall(name, io_list, mode):
    total = sum(b for b, _ in io_list)
    print(f"\n--- {name} [{mode}] ({len(io_list)} I/Os, {total/1e6:.1f}MB) ---")
    print(f"{'Level':<26}{'Useful GB/s':>12}{'Eff/REF':>9}"
          f"{'ΔLoss':>9}{'Span ms':>10}")
    prev_bw = None
    labels = [("L0 Ideal (aggregate)", 0),
              ("L1 + channel path", 1),
              ("L2 + page latency", 2),
              ("L3 + plane contention", 3)]
    for label, lvl in labels:
        bw, span = run_level(io_list, lvl, mode)
        eff = bw / REF
        loss = f"-{(prev_bw - bw) / prev_bw * 100:.1f}%" if prev_bw else "  —"
        print(f"{label:<26}{bw/1e9:>12.2f}{eff:>8.1%}{loss:>9}"
              f"{span*1e3:>10.3f}")
        if bw > 0:
            prev_bw = bw


def main():
    print("=" * 78)
    print("Phase 4-Closure P2: measured loss attribution waterfall")
    print(f"Reference BW = {REF/1e9:.1f} GB/s (SIMULATED aggregate, "
          "not hardware measured)")
    print("=" * 78)

    # 1. KV restore burst (N=2 Host Hit, concurrent)
    waterfall("KV restore burst N=2", [(339 * KV, True)] * 2, "burst")

    # 2. KV backup burst (N=2 concurrent write-through)
    waterfall("KV backup burst N=2", [(340 * KV, False)] * 2, "burst")

    # 3. Churn (9 sequential backups + 7 restores, closed-loop)
    churn = [(340 * KV, False)] * 9 + [(339 * KV, True)] * 7
    waterfall("Churn (9 backup + 7 restore)", churn, "closed")

    # 4. Small I/O, closed-loop QD=1 (latency-bound regime)
    waterfall("Small I/O 32×64KB", [(65536, True)] * 32, "closed")

    # 5. QD comparison for small I/O (burst vs closed, full model L3)
    small = [(65536, True)] * 32
    bw_b, _ = run_level(small, 3, "burst")
    bw_c, _ = run_level(small, 3, "closed")
    print(f"\n--- QD effect on small I/O (32×64KB, full model) ---")
    print(f"burst  (QD=32): {bw_b/1e9:6.2f} GB/s ({bw_b/REF:.1%})")
    print(f"closed (QD=1):  {bw_c/1e9:6.2f} GB/s ({bw_c/REF:.1%})")
    print("→ QD fills idle channels: small-I/O efficiency is a QD "
          "property, not fixed.")

    # 6. KV-size I/O single (the actual per-request restore size)
    waterfall("Single KV restore (339tk)", [(339 * KV, True)], "closed")
    waterfall("Single KV backup (340tk)", [(340 * KV, False)], "closed")

    print("\n" + "=" * 78)
    print("UNMODELED (Unknown — deliberately no numeric loss assigned):")
    print("  FTL mapping, ECC, GC, read disturb, write amplification,")
    print("  die-enable bus sharing. Requires MQSim/hardware calibration.")
    print("=" * 78)


if __name__ == "__main__":
    main()
