#!/usr/bin/env python3
"""Generate Phase 2A minimal test trace with hotspot pattern.

8 hot sessions (S0-S7, accessed 3x each) + 8 cold sessions (S8-S15, 1x each)
+ 16 churn sessions + 16 reuse sessions (8 hot + 8 cold).

With KDA=8: 8+8+16=32 checkpoints, only 8 slots → heavy eviction.
Hot sessions should survive with HotnessAware but be evicted with Native LRU.
"""
import json, os, sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
C = 1024

def tokens(base, n=C):
    return [base + i for i in range(n + 10)]

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}

def gen():
    rows = []
    ts = 0.0
    hot = [tokens(100000 * (i+1)) for i in range(8)]   # S0-S7
    cold = [tokens(200000 * (i+1)) for i in range(8)]   # S8-S15
    churn = [tokens(300000 * (i+1)) for i in range(16)] # C0-C15

    # Build hot (8)
    for i, t in enumerate(hot):
        rows.append(row(t, C, ts, f"build_hot_S{i}")); ts += 10.0
    # Build cold (8)
    for i, t in enumerate(cold):
        rows.append(row(t, C, ts, f"build_cold_S{i}")); ts += 10.0
    # Churn (16) — create capacity pressure
    for i, t in enumerate(churn):
        rows.append(row(t, C, ts, f"churn_C{i}")); ts += 10.0
    # Re-access hot (round 2) — makes them "hotter"
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reaccess_hot_S{i}")); ts += 10.0
    # Churn again (8 more)
    for i in range(8):
        rows.append(row(churn[i], C, ts, f"churn2_C{i}")); ts += 10.0
    # Final reuse: hot should hit, cold should miss
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reuse_hot_S{i}")); ts += 10.0
    for i, t in enumerate(cold):
        rows.append(row(t, C+1, ts, f"reuse_cold_S{i}")); ts += 10.0
    return rows

if __name__ == "__main__":
    rows = gen()
    path = os.path.join(OUTDIR, "k3_p2a_trace.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    print(f"[P2A-trace] {path}: {len(rows)} reqs, {total} tokens")
    print(f"  8 hot (3x access) + 8 cold (1x) + 16 churn + 16 reuse")
