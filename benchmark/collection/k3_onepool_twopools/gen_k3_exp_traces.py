#!/usr/bin/env python3
"""Generate the One Pool vs Two Pools experiment traces (P5.1 / P5.2 / P5.3).

Mirrors docs/k3_onepool_vs_twopools_experiment_report.md:
  - P5.1 KDA-heavy : 96 x 1024 build + 6 x 1025 reuse (C=1024)
  - P5.2 MLA-heavy : 8 x 32768 build + 3 x 33024 reuse (C=8192)
  - P5.3 dynamic   : A(96x1K + 6x1K+1) -> B(8x32K + 3x32K+256) -> C(revisit A+B)
  - P5.2 seq       : P5.2 with arrivals spaced 120s apart (concurrency=1
                     attribution variant: rules out arrival-pattern effects on
                     LRU eviction victim selection)

Trace design rule (report appendix): build input_len = C so that the tracked
boundary == C; reuse input_len = C + N (N > 0) so full prefix match holds; reuse
tokens = build tokens + N new tokens. Session token ids come from disjoint
ranges so cross-session prefixes never collide. Fake ids > vocab are fine: the
server accepts raw input_ids and the mock forward is never executed.

Usage: gen_k3_exp_traces.py [outdir]   (default /tmp/opencode)
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"

C_KDA = 1024          # P5.1 / P5.3 phase A track interval, build length
C_MLA = 8192          # P5.2 / P5.3 phase B track interval, per-checkpoint stride
OUT_LEN = 1

KDA_SESSIONS = 96
KDA_REUSE = [0, 16, 32, 48, 64, 80]
MLA_SESSIONS = 8
MLA_BUILD = 32768
MLA_REUSE_EXTRA = 256
MLA_REUSE = [0, 3, 7]


def kda_base(s):
    return [s * 200_000 + i for i in range(C_KDA + 1)]


def mla_base(s):
    return [500_000_000 + s * 200_000 + i for i in range(MLA_BUILD + MLA_REUSE_EXTRA)]


def row(ids, length, ts):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": OUT_LEN, "timestamp": ts}


def p51():
    rows = []
    for s in range(KDA_SESSIONS):
        rows.append(row(kda_base(s), C_KDA, s * 1.0))
    for i, s in enumerate(KDA_REUSE):
        rows.append(row(kda_base(s), C_KDA + 1, 200.0 + i))
    return rows


def p52():
    rows = []
    for s in range(MLA_SESSIONS):
        rows.append(row(mla_base(s), MLA_BUILD, s * 1.0))
    for i, s in enumerate(MLA_REUSE):
        rows.append(row(mla_base(s), MLA_BUILD + MLA_REUSE_EXTRA, 100.0 + i))
    return rows


def p52_seq():
    """P5.2 with builds strictly serialized (concurrency=1): space arrivals far
    apart relative to per-request service time (~15s virtual)."""
    rows = []
    for s in range(MLA_SESSIONS):
        rows.append(row(mla_base(s), MLA_BUILD, s * 120.0))
    for i, s in enumerate(MLA_REUSE):
        rows.append(row(mla_base(s), MLA_BUILD + MLA_REUSE_EXTRA, 1200.0 + i * 120.0))
    return rows


def p53():
    rows = []
    # Phase A: KDA-heavy build + reuse
    for s in range(KDA_SESSIONS):
        rows.append(row(kda_base(s), C_KDA, s * 1.0))
    for i, s in enumerate(KDA_REUSE):
        rows.append(row(kda_base(s), C_KDA + 1, 150.0 + i))
    # Phase B: MLA-heavy build + reuse (radix tree NOT cleared between phases)
    for s in range(MLA_SESSIONS):
        rows.append(row(mla_base(s), MLA_BUILD, 300.0 + s))
    for i, s in enumerate(MLA_REUSE):
        rows.append(row(mla_base(s), MLA_BUILD + MLA_REUSE_EXTRA, 500.0 + i))
    # Phase C: cross-phase revisit of A(0,32,80) and B(0,3,7)
    for i, s in enumerate([0, 32, 80]):
        rows.append(row(kda_base(s), C_KDA + 1, 700.0 + i))
    for i, s in enumerate(MLA_REUSE):
        rows.append(row(mla_base(s), MLA_BUILD + MLA_REUSE_EXTRA, 720.0 + i))
    return rows


def write(name, rows):
    rows.sort(key=lambda x: x["timestamp"])
    path = os.path.join(OUTDIR, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    print(f"[gen_k3_exp] {path}: {len(rows)} reqs, total input {total} tokens")


if __name__ == "__main__":
    write("k3_p51_kda_heavy.jsonl", p51())
    write("k3_p52_mla_heavy.jsonl", p52())
    write("k3_p52_mla_heavy_seq.jsonl", p52_seq())
    write("k3_p53_dynamic.jsonl", p53())
