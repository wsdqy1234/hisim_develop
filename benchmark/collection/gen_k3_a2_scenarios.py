#!/usr/bin/env python3
"""Generate A2 four-scenario KDA checkpoint verification trace.

Four controlled scenarios with SEPARATE prefixes to avoid cross-scenario
cache state interference:

  1. L1-only:  Build P0 → Reuse P0 (0 intermediates, P0 still in L1)
  2. L2-only:  Build P1 → Build Q1 → Build R1 → Reuse P1
              (2 intermediates evict P1 from L1 to L2)
  3. L3-only:  Build P2 → Build Q2 → Build R2 → Build S2 → Reuse P2
              (3 intermediates: R2 evicts P2 from L1, S2 evicts P2 from L2)
  4. All-Miss: Build T (completely new prefix, no match anywhere)

L1=4 (HISIM_MAMBA_POOL_SIZE=4), L2=3 (HISIM_L2_MAMBA_SIZE=3).

Slot accounting per scenario (L1=4, each req needs 3 slots = 1 run + 2 pp):

L1-only:
  Build P0:  L1=[P0_ckpt, free, free, free] (1 used)
  Reuse P0:  L1 hit (P0_ckpt value set, no init_load_back restore needed)

L2-only:
  Build P1:  L1=[P1_ckpt]
  Build Q1:  L1=[Q1_ckpt, P1_ckpt] (2 used)
  Build R1:  alloc 3 > 2 free → evict P1_ckpt → L2. L1=[R1_ckpt, Q1_ckpt]
             L2=[R1, Q1, P1] (3, full)
  Reuse P1:  L2 hit (host_value set, value=None → init_load_back restores)

L3-only:
  Build P2:  L1=[P2_ckpt], L2=[P2]
  Build Q2:  L1=[Q2_ckpt, P2_ckpt], L2=[Q2, P2]
  Build R2:  evict P2 → L1=[R2_ckpt, Q2_ckpt], L2=[R2, Q2, P2] (full)
  Build S2:  evict Q2 → L1=[S2_ckpt, R2_ckpt], L2 full → evict P2 → L3
             L2=[S2, R2, Q2]
  Reuse P2:  L3 hit (only L3 has P2, L2 miss → L3 restore)

All-Miss:
  Build T:   new prefix, no match → full prefill

Usage: gen_k3_a2_scenarios.py [outdir]  (default /tmp/opencode)
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"

C = 1024          # track interval / build prefix length
OUT_LEN = 1       # minimal output


def prefix_tokens(base, n=C):
    """Disjoint token ranges: [base, base+1, ..., base+n-1]."""
    return [base + i for i in range(n + 10)]


def row(ids, length, ts, label=""):
    return {
        "input_ids": ids[:length],
        "input_length": length,
        "output_length": OUT_LEN,
        "timestamp": ts,
        "session_label": label,
    }


def gen_trace():
    rows = []
    ts = 0.0

    # Phase 1: L1-only (prefix P0)
    P0 = prefix_tokens(1)
    rows.append(row(P0, C, ts, "build_P0")); ts += 10.0
    rows.append(row(P0, C + 1, ts, "reuse_P0_L1")); ts += 10.0

    # Phase 2: L2-only (prefix P1, intermediates Q1, R1)
    P1 = prefix_tokens(100_000)
    Q1 = prefix_tokens(200_000)
    R1 = prefix_tokens(300_000)
    rows.append(row(P1, C, ts, "build_P1")); ts += 10.0
    rows.append(row(Q1, C, ts, "build_Q1")); ts += 10.0
    rows.append(row(R1, C, ts, "build_R1")); ts += 10.0
    rows.append(row(P1, C + 1, ts, "reuse_P1_L2")); ts += 10.0

    # Phase 3: L3-only (prefix P2, intermediates Q2, R2, S2)
    P2 = prefix_tokens(400_000)
    Q2 = prefix_tokens(500_000)
    R2 = prefix_tokens(600_000)
    S2 = prefix_tokens(700_000)
    rows.append(row(P2, C, ts, "build_P2")); ts += 10.0
    rows.append(row(Q2, C, ts, "build_Q2")); ts += 10.0
    rows.append(row(R2, C, ts, "build_R2")); ts += 10.0
    rows.append(row(S2, C, ts, "build_S2")); ts += 10.0
    rows.append(row(P2, C + 1, ts, "reuse_P2_L3")); ts += 10.0

    # Phase 4: All-Miss (new prefix T)
    T = prefix_tokens(800_000)
    rows.append(row(T, C, ts, "all_miss_T")); ts += 10.0

    return rows


if __name__ == "__main__":
    rows = gen_trace()
    path = os.path.join(OUTDIR, "k3_a2_scenarios.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    print(f"[gen_a2_scenarios] {path}: {len(rows)} reqs, total input {total} tokens")
    for r in rows:
        print(f"  ts={r['timestamp']:5.1f} len={r['input_length']:4d} label={r['session_label']}")
