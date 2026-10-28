#!/usr/bin/env python3
"""Phase 2D traces: Bidirectional KDA↔MLA pressure + Reuse.

Trace 1 (kda_to_mla): KDA-heavy → MLA-heavy → KDA Reuse
  Phase 1: 24 build × 8K tokens (creates 24 KDA checkpoints, 8 evicted to L2)
  Phase 2: 300 MLA × 1000 tokens (fills L1=200K, 100K overflow to L2 MLA)
  Phase 3: 4 reuse × 8K tokens (same prefix as Phase 1[0:4], tests KDA L2/L3 hit)

Trace 2 (mla_to_kda): MLA-heavy → KDA-heavy → MLA Reuse
  Phase 1: 300 MLA × 1000 tokens (fills L1=200K, 100K overflow to L2 MLA)
  Phase 2: 24 build × 8K tokens (creates 24 KDA checkpoints, 8 evicted to L2)
  Phase 3: 4 reuse × 1000 tokens (same prefix as Phase 1[0:4], tests MLA L2/L3 hit)

Design: 0.5GB L2 budget → MLA capacity ~67K tokens (Two Pools)
        100K overflow > 67K → L2 MLA fills → cross-evict pressure in One Pool
"""
import json, os, sys

def tokens(base, n):
    return [base + i for i in range(n + 10)]

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}

def gen_kda_to_mla(outdir):
    """KDA-heavy → MLA-heavy → KDA Reuse"""
    rows = []
    ts = 0.0
    # Phase 1: 24 KDA-heavy build (8K tokens each, distinct prefixes)
    kda_bases = [100000 * (i + 1) for i in range(24)]
    kda_tokens = [tokens(kda_bases[i], 8192) for i in range(24)]
    for i in range(24):
        rows.append(row(kda_tokens[i], 8192, ts, f"P1_kda_{i}"))
        ts += 10.0
    # Phase 2: 300 MLA-heavy (1000 tokens each, distinct prefixes, no KDA checkpoint)
    for i in range(300):
        base = 500000 * (i + 1)
        ids = tokens(base, 1000)
        rows.append(row(ids, 1000, ts, f"P2_mla_{i}"))
        ts += 10.0
    # Phase 3: 4 KDA reuse (same prefix as Phase 1[0:4] + 1 new token)
    for i in range(4):
        rows.append(row(kda_tokens[i], 8193, ts, f"P3_reuse_kda_{i}"))
        ts += 10.0
    path = os.path.join(outdir, "k3_p2d_kda_to_mla.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Generated {path}: {len(rows)} requests")

def gen_mla_to_kda(outdir):
    """MLA-heavy → KDA-heavy → MLA Reuse"""
    rows = []
    ts = 0.0
    # Phase 1: 300 MLA-heavy (1000 tokens each, distinct prefixes)
    mla_bases = [500000 * (i + 1) for i in range(300)]
    mla_tokens = [tokens(mla_bases[i], 1000) for i in range(300)]
    for i in range(300):
        rows.append(row(mla_tokens[i], 1000, ts, f"P1_mla_{i}"))
        ts += 10.0
    # Phase 2: 24 KDA-heavy build (8K tokens each, distinct prefixes)
    for i in range(24):
        base = 100000 * (i + 1)
        ids = tokens(base, 8192)
        rows.append(row(ids, 8192, ts, f"P2_kda_{i}"))
        ts += 10.0
    # Phase 3: 4 MLA reuse (same prefix as Phase 1[0:4] + 1 new token)
    for i in range(4):
        rows.append(row(mla_tokens[i], 1001, ts, f"P3_reuse_mla_{i}"))
        ts += 10.0
    path = os.path.join(outdir, "k3_p2d_mla_to_kda.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Generated {path}: {len(rows)} requests")

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
    gen_kda_to_mla(outdir)
    gen_mla_to_kda(outdir)
