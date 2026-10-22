#!/usr/bin/env python3
"""Phase 2A.1 analysis: compare native vs hotness vs oracle.

Reports:
  - Completion (56/56)
  - COW Hits, Replay Tokens, Prefix Reuse Ratio
  - Policy decisions (EVICT/SKIP counts)
  - Victim node lists
  - Victim re-access rate
"""
import json, sys, os
from collections import defaultdict

HISIM_DIR = "/tmp/hisim"
WORKDIR = "/tmp/opencode"

def load_metrics(pol):
    path = f"{HISIM_DIR}/p2a_{pol}/metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_bench_log(pol):
    path = f"{WORKDIR}/p2a_{pol}_bench.log"
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()

def load_policy_events(pol):
    path = f"{HISIM_DIR}/p2a_{pol}/checkpoint_events.jsonl"
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("event") == "policy_evict":
                    events.append(e)
            except:
                pass
    return events

def load_policy_starts(pol):
    path = f"{HISIM_DIR}/p2a_{pol}/checkpoint_events.jsonl"
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("event") == "policy_start":
                    events.append(e)
            except:
                pass
    return events

def analyze(pol):
    print(f"\n{'='*60}")
    print(f"Policy: {pol}")
    print(f"{'='*60}")

    # Completion
    blog = load_bench_log(pol)
    done = "All prompts done" in blog
    print(f"  Completed 56/56: {'YES' if done else 'NO'}")

    # Metrics
    m = load_metrics(pol)
    if m:
        for key in ["prefix_cache_reused_ratio", "kda_checkpoint_donate_count",
                     "kda_checkpoint_hit_count", "kda_cow_exec_count",
                     "kda_historical_replay_tokens", "kda_l3_hits",
                     "mla_prefix_hit_tokens_agg"]:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                print(f"  {key:40s} = {val:.4f}")
            else:
                print(f"  {key:40s} = {val}")
    else:
        print("  Metrics: NOT FOUND")

    # Policy start events
    starts = load_policy_starts(pol)
    print(f"\n  Policy start events: {len(starts)}")
    for s in starts[:3]:
        extra = s
        print(f"    req={extra.get('request_cnt',-1)} cand={extra.get('candidates',-1)} q={extra.get('queue_size',-1)}")

    # Policy evict events
    evs = load_policy_events(pol)
    evicts = [e for e in evs if e.get("action") == "evict"]
    skips = [e for e in evs if e.get("action") == "skip"]
    print(f"\n  Policy decisions: total={len(evs)} evict={len(evicts)} skip={len(skips)}")

    # Victim nodes
    victim_nodes = [e.get("node_id", -1) for e in evicts]
    if victim_nodes:
        print(f"  Victim nodes (first 20): {victim_nodes[:20]}")
        # Check re-access: which victims were accessed again after eviction
        # (This requires timestamp analysis from checkpoint_events)
    if skips:
        skip_nodes = [e.get("node_id", -1) for e in skips]
        print(f"  Skipped nodes (first 10): {skip_nodes[:10]}")

    return m, evs

if __name__ == "__main__":
    policies = sys.argv[1:] if len(sys.argv) > 1 else ["native", "hotness", "oracle"]
    results = {}
    for p in policies:
        results[p] = analyze(p)

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':<40s}", end="")
    for p in policies:
        print(f" {p:>12s}", end="")
    print()

    metrics_to_compare = [
        "prefix_cache_reused_ratio",
        "kda_checkpoint_donate_count",
        "kda_checkpoint_hit_count",
        "kda_cow_exec_count",
        "kda_historical_replay_tokens",
    ]
    for key in metrics_to_compare:
        print(f"{key:<40s}", end="")
        for p in policies:
            m = results[p][0]
            val = m.get(key, "N/A") if m else "N/A"
            if isinstance(val, float):
                print(f" {val:>12.2f}", end="")
            elif isinstance(val, int):
                print(f" {val:>12d}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()

    # Completion
    print(f"{'Completed 56/56':<40s}", end="")
    for p in policies:
        blog = load_bench_log(p)
        done = "YES" if "All prompts done" in blog else "NO"
        print(f" {done:>12s}", end="")
    print()

    # Policy decisions
    print(f"{'Policy EVICT count':<40s}", end="")
    for p in policies:
        evs = results[p][1]
        n = len([e for e in evs if e.get("action") == "evict"])
        print(f" {n:>12d}", end="")
    print()
    print(f"{'Policy SKIP count':<40s}", end="")
    for p in policies:
        evs = results[p][1]
        n = len([e for e in evs if e.get("action") == "skip"])
        print(f" {n:>12d}", end="")
    print()
