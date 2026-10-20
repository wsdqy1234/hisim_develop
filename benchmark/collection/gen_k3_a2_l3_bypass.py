#!/usr/bin/env python3
"""Generate A2 L3-only + All-Miss trace (L2=0 bypass).

With L2 disabled (HISIM_L2_MAMBA_SIZE=0), every checkpoint goes
directly to L3. This bypasses the L2 post-restore deadlock.

Scenarios:
  1. L1-only:  Build P0 → Reuse P0 (P0 still in L1, COW hit)
  2. L3-only:  Build P2 → Build Q2 → Build R2 → Reuse P2
              (P2 evicted from L1, only in L3 since L2=0)
  3. All-Miss: Build T (new prefix, no match)

L1=4, L2=0, L3=file(80).
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
C = 1024
OUT_LEN = 1

def prefix_tokens(base, n=C):
    return [base + i for i in range(n + 10)]

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": OUT_LEN, "timestamp": ts, "session_label": label}

def gen_trace():
    rows = []
    ts = 0.0
    P0 = prefix_tokens(1)
    rows.append(row(P0, C, ts, "build_P0")); ts += 10.0
    rows.append(row(P0, C + 1, ts, "reuse_P0_L1")); ts += 10.0

    P2 = prefix_tokens(400_000)
    Q2 = prefix_tokens(500_000)
    R2 = prefix_tokens(600_000)
    rows.append(row(P2, C, ts, "build_P2")); ts += 10.0
    rows.append(row(Q2, C, ts, "build_Q2")); ts += 10.0
    rows.append(row(R2, C, ts, "build_R2")); ts += 10.0
    rows.append(row(P2, C + 1, ts, "reuse_P2_L3")); ts += 10.0

    T = prefix_tokens(800_000)
    rows.append(row(T, C, ts, "all_miss_T")); ts += 10.0
    return rows

if __name__ == "__main__":
    rows = gen_trace()
    path = os.path.join(OUTDIR, "k3_a2_l3_bypass.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    print(f"[gen_a2_l3_bypass] {path}: {len(rows)} reqs, total input {total} tokens")
    for r in rows:
        print(f"  ts={r['timestamp']:5.1f} len={r['input_length']:4d} label={r['session_label']}")
