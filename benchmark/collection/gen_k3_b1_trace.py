#!/usr/bin/env python3
"""Generate B1 trace: L1-only capacity pressure.

96 independent 1024-token Build sessions → 32 Churn sessions →
24 Reuse sessions (8 early + 8 mid + 8 recent).

Each Build session creates a unique 1024-token prefix.
Churn sessions use different prefixes to create L1 capacity pressure.
Reuse sessions re-use early/mid/recent Build prefixes (with +1 token
extension) to observe hit/miss patterns under native LRU eviction.

KDA Slots sweep: 40/80/165 (separate experiments).
With 96+32=128 total checkpoints and KDA=40 → 88 evicted.
With KDA=80 → 48 evicted. With KDA=165 → 0 evicted (all survive).

Usage: gen_k3_b1_trace.py [outdir]
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
C = 1024          # track interval / build prefix length
OUT_LEN = 1       # minimal output
TOKEN_BASE = 100_000  # base for token generation

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

    # --- Phase 1: 96 Build sessions (S0..S95) ---
    build_prefixes = []
    for i in range(96):
        base = TOKEN_BASE * (i + 1)
        toks = prefix_tokens(base)
        build_prefixes.append(toks)
        rows.append(row(toks, C, ts, f"build_S{i:02d}"))
        ts += 10.0

    # --- Phase 2: 32 Churn sessions (C0..C31) ---
    for i in range(32):
        base = TOKEN_BASE * (100 + i)
        toks = prefix_tokens(base)
        rows.append(row(toks, C, ts, f"churn_C{i:02d}"))
        ts += 10.0

    # --- Phase 3: 24 Reuse sessions ---
    # Early: S0..S7 (oldest, most likely evicted)
    for i in range(8):
        toks = build_prefixes[i]
        rows.append(row(toks, C + 1, ts, f"reuse_early_S{i:02d}"))
        ts += 10.0

    # Mid: S44..S51
    for i in range(44, 52):
        toks = build_prefixes[i]
        rows.append(row(toks, C + 1, ts, f"reuse_mid_S{i:02d}"))
        ts += 10.0

    # Recent: S88..S95 (newest, most likely retained)
    for i in range(88, 96):
        toks = build_prefixes[i]
        rows.append(row(toks, C + 1, ts, f"reuse_recent_S{i:02d}"))
        ts += 10.0

    return rows

if __name__ == "__main__":
    rows = gen_trace()
    path = os.path.join(OUTDIR, "k3_b1_trace.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    # Summary by phase
    phases = {}
    for r in rows:
        phase = r["session_label"].split("_")[0]
        phases[phase] = phases.get(phase, 0) + 1
    print(f"[B1-trace] {path}: {len(rows)} reqs, total {total} tokens")
    for p, c in sorted(phases.items()):
        print(f"  {p:10s}= {c} reqs")
    print(f"  Phases: 96 build → 32 churn → 8 early + 8 mid + 8 recent reuse")
