"""Join validated E2E cases, logical references and actual Host I/O.

Counterfactual reference capacities, configured Host capacities and observed
residency/traffic are deliberately kept in distinct output columns.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(root, output, repo):
    sys.path.insert(0, str(Path(repo) / "test/research_hbf"))
    from analyze_host import analyze
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, requirements, excluded = [], [], []
    for case in sorted(root.iterdir()):
        manifest_file = case / "manifest.json"
        if not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("status") != "complete":
            excluded.append({"case": case.name, "status": manifest.get("status"),
                             "error": manifest.get("error")})
            continue
        result = analyze(case)
        if not result["data_quality"]["all_load_bytes_conserved"]:
            raise ValueError("Observed Host byte conservation failed: " + case.name)
        cfg = json.loads((case / "effective_config.json").read_text())
        refs = jsonl(case / "logical_prefix_accesses.jsonl")
        requests = {row["rid"]: row for row in jsonl(case / "request.jsonl")}
        trace = [row for row in jsonl(case / "io_trace.jsonl") if row.get("path") == "real"]
        reused, requested, recomputed = 0, 0, 0
        for ref in refs:
            if ref["phase"] != "reuse":
                continue
            req = requests[ref["request_id"]]
            prefix = ref["prefix_tokens"]
            cached = min(prefix, max(0, req["input_length"] - req["recomputed_tokens"]))
            requested += prefix
            reused += cached
            recomputed += prefix - cached
        spec = manifest["workload_spec"]
        geometry = cfg["geometry"]
        logical = result["logical_demand"]
        loads = [row for row in trace if row["op"] == "load"]
        hbf = [row for row in loads if row["pool"] == "hbf"]
        hbf_writes = [row for row in trace if row["pool"] == "hbf" and row["op"] == "backup"]
        l3_writes = [row for row in hbf_writes
                     if (row.get("details") or {}).get("metadata", {}).get("source") == "l3_restore"]
        children = [child for row in hbf for child in (row.get("details") or {}).get("sub_ios", [])]
        row = {"case": case.name, "scenario": manifest["args"]["scenario"],
               "prefix_tokens": spec["prefix_tokens"], "concurrency": spec["concurrency"],
               "reuse_waves": spec["reuse"], "churn_per_wave": spec["churn"],
               "completed_requests": len(requests),
               "workload_content_sha256": manifest["workload_content_sha256"],
               "l1_tokens": cfg["capacities"]["l1_tokens"],
               "configured_host_tokens": cfg["capacities"]["host_total_tokens"],
               "configured_host_gb": cfg["capacities"]["host_total_tokens"] * geometry["bytes_per_token"] / 1e9,
               "logical_unique_prefix_union_gb": logical["unique_reusable_prefix_bytes"] / 1e9,
               "offline_lru_sampled_c95_gb": (logical["sampled_capacity_for_repeat_hit_target_bytes"]["0.95"] or 0) / 1e9,
               "reuse_requested_prefix_tokens": requested, "reuse_cached_prefix_tokens": reused,
               "reuse_prefix_hit_ratio_all_tiers": reused / requested if requested else None,
               "reuse_recomputed_prefix_tokens": recomputed,
               "hbf_restore_ios": len(hbf), "hbf_restore_bytes": sum(e["bytes"] for e in hbf),
               "hbf_physical_write_bytes": sum(e["bytes"] for e in hbf_writes),
               "hbf_l3_materialization_write_bytes": sum(e["bytes"] for e in l3_writes),
               "hbf_l3_materialization_ios": len(l3_writes),
               "dram_restore_bytes": sum(e["bytes"] for e in loads if e["pool"] == "dram"),
               "hbf_restore_duration_max_ms": max((e["dur_s"] for e in hbf), default=0) * 1000,
               "hbf_child_size_max_mb": max((e["bytes"] for e in children), default=0) / 1e6,
               "hbf_parent_inflight_max": result["observed_host_io"]["groups"].get("hbf:load", {}).get("logical_inflight", {}).get("max", 0),
               "l3_consumed_tokens_all_requests": sum(r.get("l3_consumed_tokens", 0) for r in requests.values()),
               "source_frozen_during_case": manifest["git_state_unchanged_during_run"],
               "all_load_bytes_conserved": True}
        rows.append(row)
        for deadline in (1, 10, 20, 100):
            wave_bytes = spec["concurrency"] * spec["prefix_tokens"] * geometry["bytes_per_token"]
            bw = wave_bytes / (deadline / 1000)
            requirements.append({"case": case.name, "scope": "hypothesis: one reuse wave fully restored from Host",
                                 "deadline_ms": deadline, "wave_payload_bytes": wave_bytes,
                                 "required_effective_read_gbps": bw / 1e9,
                                 "logical_full_prefix_union_bytes": logical["unique_reusable_prefix_bytes"],
                                 "full_union_capacity_over_required_bw_s": logical["unique_reusable_prefix_bytes"] / bw,
                                 "capacity_basis": "retain every unique reusable-prefix block referenced in this finite workload; not minimum Host capacity",
                                 "saturated_seq_rated_fraction": bw / 1600e9})
    write_csv(output / "case_matrix.csv", rows)
    write_csv(output / "conditional_capacity_bandwidth.csv", requirements)
    (output / "excluded_cases.json").write_text(json.dumps(excluded, indent=2))
    (output / "definitions.json").write_text(json.dumps({
        "evidence": "Layer-A model + deterministic Qwen3 Host experiments; hardware uncalibrated",
        "logical_capacity": "Unique finite-reference union, or explicitly labeled counterfactual byte-LRU curve",
        "configured_host": "Actual adapter token capacity times real model KV bytes/token",
        "reuse_hit_ratio": "Reused prefix tokens over requested prefix tokens, reuse phase only, all cache tiers",
        "hbf_qd": "Outstanding logical HBF restores; each may fan out into multiple WIN I/Os",
        "l3": "L3 consumed counters may overlap the Host restore path; not an independent additive byte partition",
        "deadline_scenarios": "Illustrative full-Host-restore service deadlines, not an agreed production SLA",
        "not_hardware_proof": True}, indent=2))
    print(json.dumps({"validated_cases": len(rows), "excluded_cases": len(excluded), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    summarize(args.root, args.output, args.repo)
