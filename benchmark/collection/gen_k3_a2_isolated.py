#!/usr/bin/env python3
"""Generate 4 isolated A2 scenario traces (one file per scenario).

Each scenario runs in its own server instance with a clean cache state,
ensuring no cross-scenario interference.

Scenarios:
  1. L1-only:  Build P0 → Reuse P0 (P0 stays in L1, Device COW)
  2. L2-only:  Build P1 → Build Q1 → Build R1 → Reuse P1 (L2 Load-back)
  3. L3-only:  Build P2 → Build Q2 → Build R2 → Build S2 → Reuse P2 (L3 Restore)
  4. All-Miss: Build T (new prefix, full prefill, no match)

Usage: gen_k3_a2_isolated.py [outdir]
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
C = 1024          # track interval / build prefix length
OUT_LEN = 1       # minimal output


def prefix_tokens(base, n=C):
    return [base + i for i in range(n + 10)]


def row(ids, length, ts, label=""):
    return {
        "input_ids": ids[:length],
        "input_length": length,
        "output_length": OUT_LEN,
        "timestamp": ts,
        "session_label": label,
    }


def gen_l1_only():
    """L1-only: Build P0, then reuse P0 (extend by 1 token).
    P0 checkpoint stays in L1, reuse via Device COW."""
    rows = []
    ts = 0.0
    P0 = prefix_tokens(1)
    rows.append(row(P0, C, ts, "build_P0")); ts += 10.0
    rows.append(row(P0, C + 1, ts, "reuse_P0_L1")); ts += 10.0
    return rows


def gen_l2_only():
    """L2-only: Build P1, then 2 intermediates (Q1, R1) to evict P1 to L2.
    With L1=4: P1+Q1 use 2 ckpt slots, R1 needs 3 alloc but only 2 free
    → native eviction evicts P1 (LRU) to L2. Reuse P1 via L2 Load-back."""
    rows = []
    ts = 0.0
    P1 = prefix_tokens(100_000)
    Q1 = prefix_tokens(200_000)
    R1 = prefix_tokens(300_000)
    rows.append(row(P1, C, ts, "build_P1")); ts += 10.0
    rows.append(row(Q1, C, ts, "build_Q1")); ts += 10.0
    rows.append(row(R1, C, ts, "build_R1")); ts += 10.0
    rows.append(row(P1, C + 1, ts, "reuse_P1_L2")); ts += 10.0
    return rows


def gen_l3_only():
    """L3-only: Build P2, then 3 intermediates (Q2, R2, S2) to evict P2
    from L1→L2→L3. With L1=4, L2=3:
    - Build P2: L2=[P2], L3=[P2] (write-through)
    - Build Q2: L2=[P2,Q2], L3=[P2,Q2]
    - Build R2: L1 evicts P2, L2=[P2,Q2,R2] (full), L3=[P2,Q2,R2]
    - Build S2: L1 evicts Q2, L2 full→evict P2 to L3, L2=[Q2,R2,S2]
    - Reuse P2: L1 miss, L2 miss, L3 hit → L3 Restore
    """
    rows = []
    ts = 0.0
    P2 = prefix_tokens(400_000)
    Q2 = prefix_tokens(500_000)
    R2 = prefix_tokens(600_000)
    S2 = prefix_tokens(700_000)
    rows.append(row(P2, C, ts, "build_P2")); ts += 10.0
    rows.append(row(Q2, C, ts, "build_Q2")); ts += 10.0
    rows.append(row(R2, C, ts, "build_R2")); ts += 10.0
    rows.append(row(S2, C, ts, "build_S2")); ts += 10.0
    rows.append(row(P2, C + 1, ts, "reuse_P2_L3")); ts += 10.0
    return rows


def gen_all_miss():
    """All-Miss: Build T with completely new prefix, no match anywhere."""
    rows = []
    ts = 0.0
    T = prefix_tokens(800_000)
    rows.append(row(T, C, ts, "all_miss_T")); ts += 10.0
    return rows


SCENARIOS = [
    ("l1_only", gen_l1_only),
    ("l2_only", gen_l2_only),
    ("l3_only", gen_l3_only),
    ("all_miss", gen_all_miss),
]


if __name__ == "__main__":
    for name, gen_fn in SCENARIOS:
        rows = gen_fn()
        path = os.path.join(OUTDIR, f"k3_a2_{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        total = sum(r["input_length"] for r in rows)
        print(f"[{name}] {path}: {len(rows)} reqs, total {total} tokens")
        for r in rows:
            print(f"  ts={r['timestamp']:5.1f} len={r['input_length']:4d} label={r['session_label']}")
