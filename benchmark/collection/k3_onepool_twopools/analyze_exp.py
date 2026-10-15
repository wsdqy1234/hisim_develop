#!/usr/bin/env python3
"""Compile experiment-cell metrics snapshots into P5.1/P5.2/P5.3 tables.

Reads <workdir>/exp_results/*.json (written by run_all_exp.sh) and prints the
metrics that matter for the One Pool vs Two Pools comparison.

Usage: analyze_exp.py [results_dir]   (default /tmp/opencode/exp_results)
"""
import glob
import json
import os
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/exp_results"

KEYS = [
    "num_completed_requests", "completed",
    "kda_checkpoint_donate_count", "kda_checkpoint_hit_count", "kda_cow_exec_count",
    "kda_checkpoint_position", "kda_write_calls", "kda_l3_writes", "kda_load_calls", "kda_l3_hits",
    "kda_new_tokens", "kda_historical_replay_tokens",
    "mla_prefix_hit_tokens_agg", "mla_prefix_hit_l1_tokens_agg", "mla_prefill_new_tokens_agg",
    "peak_mamba_live_pages", "peak_mla_live_pages", "unified_flush_count", "alloc_fail_count",
    "mean_e2e_latency_ms", "median_e2e_latency_ms",
]


def show(label):
    p = os.path.join(RES, label + ".json")
    print(f"\n=== {label} ===")
    if not os.path.exists(p):
        print("  (missing)")
        return
    m = json.load(open(p))
    for k in KEYS:
        if k in m:
            print(f"  {k:34s}= {m[k]}")


def main():
    labels = sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RES, "*.json"))
    )
    print("labels:", labels)
    for lb in labels:
        show(lb)


if __name__ == "__main__":
    main()
