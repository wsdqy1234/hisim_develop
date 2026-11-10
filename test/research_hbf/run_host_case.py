#!/usr/bin/env python3
"""Deterministic Qwen3 Host-prefix experiment, with case-owned processes/files.

Run inside the existing HiSim container. --prepare-only creates the frozen
workload and manifest without importing torch/SGLang or starting a server.
All network requests and shutdown signals target the single case-owned server.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

SCHEMA_VERSION = 1
DEFAULT_MODEL = "/home/d00832329/Qwen3-0.6B"
REPO = Path(__file__).resolve().parents[2]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def token_digest(ids):
    return hashlib.sha256(struct.pack("<%dI" % len(ids), *ids)).hexdigest()


def workload_content_digest(rows):
    # Paired scenarios have different case/request IDs, but identical content,
    # arrivals and output geometry must share this digest.
    return digest_json([{key: value for key, value in row.items() if key != "request_id"}
                        for row in rows])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class WorkloadSpec:
    case_id: str
    prefix_tokens: int = 340
    suffix_tokens: int = 8
    output_tokens: int = 16
    concurrency: int = 2
    reuse: int = 1
    churn: int = 0
    block_tokens: int = 128


def make_tokens(object_index, length, *, suffix=False):
    # Fixed content in the Qwen3 vocabulary; reserve token 1 for mock generation.
    # Prefix and unique suffix occupy disjoint numeric ranges.
    low, width = (130000, 15000) if suffix else (10000, 120000)
    return [low + ((object_index * 104729 + j * 8191) % width) for j in range(length)]


def make_workload(spec, bytes_per_token):
    """Return frozen HTTP requests and ALL logical reusable-prefix references.

    References describe logical content, including cache misses. They do not
    claim to be observed Host accesses: SGLang may serve a reference from HBM,
    DRAM, HBF, L3, or recomputation. Generated output tokens are excluded.
    """
    if min(spec.prefix_tokens, spec.output_tokens, spec.concurrency, spec.block_tokens) <= 0:
        raise ValueError("prefix/output/concurrency/block sizes must be positive")
    if min(spec.suffix_tokens, spec.reuse, spec.churn) < 0:
        raise ValueError("suffix/reuse/churn must be nonnegative")
    requests, references = [], []
    n_build = spec.concurrency + 2

    def add(obj, phase, arrival):
        i = len(requests)
        prefix = make_tokens(obj, spec.prefix_tokens)
        suffix = make_tokens(i, spec.suffix_tokens, suffix=True)
        ids = prefix + suffix
        rid = "%s-r%05d" % (spec.case_id, i)
        blocks = []
        for start in range(0, len(prefix), spec.block_tokens):
            end = min(start + spec.block_tokens, len(prefix))
            # Full ancestry is part of each radix-style content identity.
            blocks.append({"id": "sha256:" + token_digest(prefix[:end]),
                           "tokens": end - start, "bytes": (end - start) * bytes_per_token})
        requests.append({"request_id": rid, "arrival_s": round(arrival, 6),
                         "phase": phase, "input_ids": ids,
                         "input_length": len(ids), "output_length": spec.output_tokens})
        references.append({"schema_version": SCHEMA_VERSION,
                           "scope": "logical_reusable_prefix_references",
                           "request_id": rid, "arrival_s": round(arrival, 6),
                           "prefix_object_index": obj,
                           "prefix_id": "sha256:" + token_digest(prefix),
                           "prefix_tokens": len(prefix), "suffix_tokens": len(suffix),
                           "suffix_id": "sha256:" + token_digest(suffix),
                           "input_tokens": len(ids), "bytes_per_token": bytes_per_token,
                           "block_tokens": spec.block_tokens, "blocks": blocks,
                           "phase": phase, "requested_output_tokens": spec.output_tokens})

    for obj in range(n_build):
        add(obj, "build", 2.0 * obj)
    next_time = 2.0 * n_build + 4.0
    next_obj = n_build
    for wave in range(spec.reuse):
        for _ in range(spec.churn):
            add(next_obj, "churn", next_time)
            next_obj += 1
            next_time += 2.0
        for obj in range(spec.concurrency):
            # No tied arrival timestamps: tie order must not depend on HTTP race.
            add(obj, "reuse", next_time + obj * 0.001)
        next_time += 4.0
    return requests, references


def model_geometry(model_path):
    cfg = json.loads((model_path / "config.json").read_text())
    if cfg.get("kv_lora_rank"):
        raise ValueError("This first-round runner expects Qwen3 GQA; MLA requires an explicit geometry adapter")
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    dim = int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
    # BF16 explicitly set on server; TP=PP=1 in this runner.
    return {"layers": layers, "kv_heads": heads, "head_dim": dim,
            "dtype": "bfloat16", "dtype_bytes": 2, "tp": 1, "pp": 1,
            "bytes_per_token": 2 * layers * heads * dim * 2,
            "vocab_size": int(cfg["vocab_size"])}


def build_plan(args, spec, geometry, case_dir):
    unit = spec.prefix_tokens + spec.suffix_tokens + spec.output_tokens
    min_l1 = spec.concurrency * unit + 128
    max_tokens = args.max_total_tokens or max(1024, min_l1)
    if max_tokens < unit:
        raise ValueError("L1 capacity must fit at least one complete request")
    # Cover builds plus a complete reinsertion wave for the no-churn baseline.
    # Explicit smaller capacities and churn cases intentionally expose pressure.
    host_rule = (2 * spec.concurrency + 2) * unit + 128
    host_tokens = args.host_capacity_tokens or max(2 * max_tokens, host_rule)
    if host_tokens < 2:
        raise ValueError("Host capacity must be at least 2 tokens")
    if args.scenario == "dram":
        hbf_tokens, dram_tokens, placement_ratio = 0, host_tokens, 0.0
    elif args.scenario == "hbf":
        # Original SGLang Host pool must be nonempty. The one-token DRAM anchor
        # cannot hold a complete request; all normal object placements use HBF.
        hbf_tokens, dram_tokens, placement_ratio = host_tokens - 1, 1, 1.0
    else:
        hbf_tokens = int(host_tokens * 0.7)
        dram_tokens, placement_ratio = host_tokens - hbf_tokens, 0.7
    dram_ratio = (dram_tokens + 0.25) / max_tokens  # int() must recover exact slots
    win_cfg = {"placement": "striped", "stripe_block_tokens": args.stripe_block_tokens,
               "layer": "envelope", "default_access_mode": "sequential",
               "package_shared_interface_bw": None if args.interface_gbps == 0 else args.interface_gbps * 1e9,
               "win": {"capacity_bytes": 100000000000,
                       "seq_read_bw": 200000000000, "rand_read_bw": 100000000000,
                       "write_bw": 10000000000, "read_latency_s": 7.5e-6,
                       "write_latency_s": 100e-6}}
    server_env = {
        "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_TRANSFER_TO_NPU": "0", "ASCEND_RT_VISIBLE_DEVICES": args.visible_device,
        "FLASHINFER_DISABLE_VERSION_CHECK": "1", "HISIM_SIMULATION_MODE": "OFFLINE",
        "HISIM_OUTPUT_DIR": str(case_dir), "HISIM_CONFIG_PATH": str(case_dir / "platform_config.json"),
        "HISIM_SHARED_DRAM": "1", "HISIM_L3_WRITE_MODE": "barrier",
        "HISIM_NUM_WARMUP": "0", "HISIM_RESET_HICACHE_STORAGE": "1",
        "HISIM_PERSIST_L3_STORAGE": "0", "HISIM_HBF_IO_TRACE": "1",
        "HISIM_L3_STORAGE_PATH": str(case_dir / "l3_storage_keys.txt"),
        "TMPDIR": str(case_dir / "runtime_tmp"),
        "HISIM_HBF_ENABLE": "0" if args.scenario == "dram" else "1",
    }
    if hbf_tokens:
        server_env.update({"HISIM_HBF_LOGICAL_SIZE": str(hbf_tokens),
                           "HISIM_HBF_RATIO": str(placement_ratio),
                           "HISIM_HBF_TIMING_MODEL": "channel", "HISIM_HBF_MEDIA_MODEL": "win8",
                           "HISIM_HBF_WIN_CONFIG": str(case_dir / "win_config.json")})
    command = [sys.executable, "-B", "-m", "hisim.simulation.sglang.launch_server",
               "--model-path", str(args.model_path.resolve()), "--load-format", "dummy",
               "--device", "cpu", "--dtype", "bfloat16",
               "--max-total-tokens", str(max_tokens), "--enable-hierarchical-cache",
               "--hicache-storage-backend", "sim", "--hicache-storage-backend-extra-config",
               '{"prefetch_threshold":1}', "--hicache-ratio", "%.16g" % dram_ratio,
               "--host", "127.0.0.1", "--port", str(args.port), "--skip-server-warmup"]
    return {"command": command, "server_env": server_env, "win_config": win_cfg,
            "capacities": {"l1_tokens": max_tokens, "host_total_tokens": host_tokens,
                           "dram_tokens": dram_tokens, "hbf_tokens": hbf_tokens,
                           "l1_concurrent_capacity_rule_tokens": min_l1,
                           "concurrent_capacity_rule_satisfied": max_tokens >= min_l1,
                           "host_full_restore_capacity_rule_tokens": host_rule,
                           "host_full_restore_capacity_rule_satisfied": host_tokens >= host_rule,
                           "host_full_prefix_count": host_tokens // spec.prefix_tokens,
                           "hbf_physical_pool_bytes": 800000000000 if hbf_tokens else 0,
                           "note": "Logical active Host capacity is distinct from the 800GB physical supply pool."},
            "geometry": geometry}


def run_git(*argv):
    proc = subprocess.run(["git", "-C", str(REPO), *argv], capture_output=True, check=True)
    return proc.stdout


def git_state():
    tracked = run_git("ls-files", "-z").split(b"\0")
    untracked = run_git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    records = []
    for name in sorted(set(tracked + untracked)):
        if not name:
            continue
        path = REPO / name.decode()
        if path.is_file():
            records.append({"path": name.decode(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"head": run_git("rev-parse", "HEAD").decode().strip(),
            "branch": run_git("branch", "--show-current").decode().strip(),
            "status_porcelain": run_git("status", "--porcelain=v1").decode(),
            "tracked_diff_sha256": hashlib.sha256(run_git("diff", "HEAD", "--binary")).hexdigest(),
            "file_manifest_sha256": digest_json(records), "files": records}


def runtime_versions():
    result = {"python": sys.version, "executable": sys.executable}
    for package in ("sglang", "torch", "aiconfigurator", "simpyssd", "kvcm_py_optimizer"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "distribution_metadata_unavailable"
    return result


def clean_server_env(overrides):
    # No previously exported experiment knobs may silently alter this case.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("HISIM_") and k not in ("TORCH_DEVICE_BACKEND_AUTOLOAD",)}
    env.update(overrides)
    return env


def http_json(url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # These are always loopback requests to the case-owned server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read().decode()
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            # /health and profile endpoints may return plain-text HTTP 200.
            return {"raw_response": raw}


def check_port_unused(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def wait_ready(proc, base_url, log_path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Case server exited before readiness; see server.log")
        if "ready to roll" in log_path.read_text(errors="replace"):
            try:
                # In this SGLang version /health aliases /health_generate and
                # injects a token request. Probe the listener without HTTP.
                port = int(base_url.rsplit(":", 1)[1])
                with socket.create_connection(("127.0.0.1", port), timeout=5):
                    pass
                return
            except (OSError, TimeoutError):
                pass
        time.sleep(0.25)
    raise TimeoutError("Case server readiness timeout")


def stop_owned_server(proc):
    """Signal only the new process group created for this exact subprocess."""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {"pid": proc.pid, "process_group": pgid, "already_exited": True}
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The parent may exit before its worker children. Probe the SAME owned
    # process group, never process names or unrelated multiprocessing workers.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return {"pid": proc.pid, "process_group": pgid, "signal": "SIGTERM", "exit_code": proc.poll()}
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=10)
    return {"pid": proc.pid, "process_group": pgid, "signal": "SIGKILL", "exit_code": proc.poll()}


def send_request(base_url, row, total, timeout):
    started = time.monotonic()
    payload = {"rid": row["request_id"], "input_ids": row["input_ids"],
               "sampling_params": {"max_new_tokens": row["output_length"],
                                   "temperature": 0, "ignore_eos": True,
                                   "custom_params": {"simulation": {
                                       "created_time": row["arrival_s"],
                                       "total_request": total,
                                       "input_length": row["input_length"],
                                       "output_length": row["output_length"]}}}}
    try:
        result = http_json(base_url + "/generate", payload, timeout)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return {"request_id": row["request_id"], "success": True, "response": result,
                "wall_elapsed_s": time.monotonic() - started}
    except Exception as exc:
        return {"request_id": row["request_id"], "success": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "wall_elapsed_s": time.monotonic() - started}


def verify_outputs(case_dir, workload):
    required = ("metrics.json", "request.jsonl", "iteration.jsonl", "io_trace.jsonl")
    missing = [name for name in required if not (case_dir / name).is_file()]
    if missing:
        raise RuntimeError("Missing simulator outputs: " + ", ".join(missing))
    metrics = json.loads((case_dir / "metrics.json").read_text())
    requests = [json.loads(s) for s in (case_dir / "request.jsonl").read_text().splitlines() if s.strip()]
    expected = {row["request_id"] for row in workload}
    observed = {row["rid"] for row in requests}
    errors = []
    if metrics.get("completed") != len(workload) or len(requests) != len(workload):
        errors.append("Simulator completed/exported count differs from frozen workload")
    if expected != observed or len(observed) != len(requests):
        errors.append("Exported request IDs differ from frozen workload")
    plan = json.loads((case_dir / "effective_config.json").read_text())
    bpt = plan["geometry"]["bytes_per_token"]
    for row in requests:
        rid = row["rid"]
        latencies = row["gen_token_latencies"]
        if len(latencies) != row["output_length"]:
            errors.append("%s: generated token count mismatch" % rid)
        if any(latency < -1e-9 for latency in latencies):
            errors.append("%s: negative token latency" % rid)
        if row["last_event_time"] + 1e-9 < row["created_time"]:
            errors.append("%s: completion precedes arrival" % rid)
        cached, host = row["final_reused_tokens"], row["l2_hit_tokens"]
        l1 = cached - host if cached >= host else cached
        if l1 + host + row["recomputed_tokens"] != row["actual_input_tokens"]:
            errors.append("%s: L1+Host+recompute token conservation failed" % rid)
        if row["hbf_hit_tokens"] + row["dram_hit_tokens"] != host:
            errors.append("%s: HBF+DRAM Host attribution failed" % rid)
        for tier in ("hbf", "dram"):
            if row[tier + "_load_bytes"] != row[tier + "_hit_tokens"] * bpt:
                errors.append("%s: %s byte/token conservation failed" % (rid, tier))
    trace = [json.loads(s) for s in (case_dir / "io_trace.jsonl").read_text().splitlines() if s.strip()]
    for row in trace:
        if row.get("completion_s") is not None and row["completion_s"] + 1e-9 < row["arrival_s"]:
            errors.append("I/O completion precedes arrival")
    # Mock tensor-copy paths contain placeholder bytes. They are diagnostic
    # records, not actual model-KV I/O and must not enter physical accounting.
    hbf_rows = [row for row in trace if row["pool"] == "hbf" and row["path"] == "real"]
    diagnostic_rows = [row for row in trace if row["path"] != "real"]
    load_conservation = {}
    for tier in ("hbf", "dram"):
        request_bytes = sum(row[tier + "_load_bytes"] for row in requests)
        trace_bytes = sum(row["bytes"] for row in trace
                          if row["pool"] == tier and row["path"] == "real" and row["op"] == "load")
        load_conservation[tier] = {"request_bytes": request_bytes, "real_trace_bytes": trace_bytes,
                                   "match": request_bytes == trace_bytes}
        if request_bytes != trace_bytes:
            errors.append("%s request / real trace load bytes do not match" % tier)
    child_count = 0
    if plan["server_env"]["HISIM_HBF_ENABLE"] == "1":
        child_path = case_dir / "win_io_events.jsonl"
        if not child_path.is_file():
            errors.append("Missing win_io_events.jsonl")
            events = []
        else:
            events = [json.loads(s) for s in child_path.read_text().splitlines() if s.strip()]
        logical = {row["logical_io_id"]: row for row in events}
        detailed = [row for row in hbf_rows if row.get("details")]
        if len(events) != len(logical) or {row["details"]["logical_io_id"] for row in detailed} != set(logical):
            errors.append("Generic HBF trace / WIN logical I/O IDs do not match")
        if sum(row["bytes"] for row in hbf_rows) != sum(row["bytes"] for row in events):
            errors.append("Generic HBF trace / WIN total bytes do not match")
        stripe = plan["win_config"]["stripe_block_tokens"]
        for event in events:
            children = event["sub_ios"]
            child_count += len(children)
            if sum(child["bytes"] for child in children) != event["bytes"]:
                errors.append("WIN child byte conservation failed")
            for child in children:
                slots = child.get("slots", [])
                if child["bytes"] != len(slots) * bpt:
                    errors.append("WIN child slot/byte conservation failed")
                if any((slot // stripe) % 8 != child["win"] for slot in slots):
                    errors.append("WIN child route differs from immutable slot stripe")
                if child["completion_s"] + 1e-9 < child["arrival_s"]:
                    errors.append("WIN child completion precedes arrival")
        summary_path = case_dir / "win_pool_summary.json"
        if not summary_path.is_file():
            errors.append("Missing win_pool_summary.json")
        else:
            conservation = json.loads(summary_path.read_text())["conservation"]
            if not conservation["bytes_consistent"] or not conservation["slots_consistent"]:
                errors.append("WIN object registry conservation failed")
    result = {"completed": metrics.get("completed"), "expected_requests": len(workload),
              "request_ids_match": expected == observed, "errors": sorted(set(errors)),
              "hbf_load_bytes": sum(row["bytes"] for row in hbf_rows if row["op"] == "load"),
              "hbf_backup_bytes": sum(row["bytes"] for row in hbf_rows if row["op"] == "backup"),
              "hbf_child_events": child_count,
              "load_byte_conservation": load_conservation,
              "diagnostics": {"non_real_trace_records": len(diagnostic_rows),
                              "non_real_hbf_bytes": sum(row["bytes"] for row in diagnostic_rows if row["pool"] == "hbf"),
                              "non_real_paths": sorted({row["path"] for row in diagnostic_rows})},
              "wall_metrics_are_not_simulated_metrics": True,
              "valid": not errors}
    write_json(case_dir / "validation.json", result)
    if errors:
        raise RuntimeError("Case validation failed: " + "; ".join(sorted(set(errors))))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="case directory must be new; shared output folders are never cleared")
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--platform-config", type=Path, default=REPO / "test/assets/mock/config.json")
    parser.add_argument("--scenario", choices=("dram", "mixed", "hbf"), default="mixed")
    parser.add_argument("--prefix-tokens", type=int, choices=(340, 4096, 16384), default=340)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--reuse", type=int, default=1, help="reuse waves; each references concurrency build objects")
    parser.add_argument("--churn", type=int, default=0, help="new objects inserted before each reuse wave")
    parser.add_argument("--block-tokens", type=int, default=128, help="logical reference block size, independent of physical striping")
    parser.add_argument("--stripe-block-tokens", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--host-capacity-tokens", type=int, default=0)
    parser.add_argument("--interface-gbps", type=float, default=0, help="decimal GB/s; 0=unbounded model hypothesis")
    parser.add_argument("--visible-device", default="7")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.case_id):
        parser.error("case-id must be a short simple name without path separators")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be in [1024,65535]")
    if args.interface_gbps < 0 or args.stripe_block_tokens <= 0:
        parser.error("interface bandwidth must be nonnegative and stripe block positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    case_dir = args.output_root.resolve() / args.case_id
    # Read-only validation occurs before reserving any case output path.
    geometry = model_geometry(args.model_path)
    spec = WorkloadSpec(args.case_id, args.prefix_tokens, args.suffix_tokens,
                        args.output_tokens, args.concurrency, args.reuse, args.churn, args.block_tokens)
    workload, references = make_workload(spec, geometry["bytes_per_token"])
    if max(max(row["input_ids"]) for row in workload) >= geometry["vocab_size"]:
        raise ValueError("Frozen token IDs exceed the selected tokenizer vocabulary")
    plan = build_plan(args, spec, geometry, case_dir)
    platform = json.loads(args.platform_config.read_text())
    if platform.get("platform", {}).get("enable_hbf"):
        raise ValueError("Use an HBF-disabled baseline platform config; runner controls HBF explicitly")
    state = git_state()
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "runtime_tmp").mkdir()
    write_jsonl(case_dir / "workload.jsonl", workload)
    write_jsonl(case_dir / "logical_prefix_accesses.jsonl", references)
    write_json(case_dir / "platform_config.json", platform)
    write_json(case_dir / "win_config.json", plan["win_config"])
    write_json(case_dir / "git_state.json", state)
    write_json(case_dir / "effective_config.json", plan)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "prepared", "case_id": args.case_id,
                "workload_spec": asdict(spec), "workload_sha256": digest_json(workload),
                "workload_content_sha256": workload_content_digest(workload),
                "logical_prefix_accesses_sha256": digest_json(references),
                "effective_config_sha256": digest_json(plan), "platform_config_sha256": digest_json(platform),
                "git_state_digest": state["file_manifest_sha256"], "git_head": state["head"],
                "runtime_versions": runtime_versions(), "scope": "Host Prefix backup/restore experiment",
                "model_artifacts": {name: hashlib.sha256((args.model_path / name).read_bytes()).hexdigest()
                                    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
                                    if (args.model_path / name).is_file()},
                "logical_reference_scope": "All reusable-prefix references including misses; offline LRU analysis is a hypothesis, not observed Host demand.",
                "limitations": ["Qwen3 GQA, TP=PP=1, BF16; Mock forward with the existing predictor",
                                "8-WIN envelope is a configured hypothesis, not physical hardware measurement",
                                "Read latency 7.5us is the midpoint of the expert 5-10us range; write latency 100us is an uncalibrated assumption, matching WinSpec defaults",
                                "L3 barrier remains enabled; no TTFT tuning",
                                "HBF-only retains one DRAM anchor token for SGLang compatibility"],
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(case_dir / "manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"case_dir": str(case_dir), "status": "prepared", "requests": len(workload)}), flush=True)
        return 0
    def terminate(_signum, _frame):
        raise SystemExit("Runner interrupted; cleaning its owned server group")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    proc = None
    base_url = "http://127.0.0.1:%d" % args.port
    started = time.time()
    try:
        check_port_unused(args.port)
        with (case_dir / "server.log").open("w") as log:
            proc = subprocess.Popen(plan["command"], cwd=REPO, env=clean_server_env(plan["server_env"]),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            manifest.update({"status": "running", "server_pid": proc.pid, "server_process_group": proc.pid,
                             "wall_started_unix_s": started})
            write_json(case_dir / "manifest.json", manifest)
            wait_ready(proc, base_url, case_dir / "server.log", args.startup_timeout)
            # ALL requests must be in-flight: OFFLINE waits for total_request.
            # A bounded client worker queue smaller than that count deadlocks.
            responses = []
            with ThreadPoolExecutor(max_workers=len(workload)) as executor:
                futures = [executor.submit(send_request, base_url, row, len(workload), args.request_timeout)
                           for row in workload]
                for future in as_completed(futures):
                    responses.append(future.result())
            responses.sort(key=lambda row: row["request_id"])
            write_jsonl(case_dir / "responses.jsonl", responses)
            failures = [row for row in responses if not row["success"]]
            if failures:
                raise RuntimeError("%d request(s) failed; see responses.jsonl" % len(failures))
            http_json(base_url + "/stop_profile", {}, timeout=30)
            validation = verify_outputs(case_dir, workload)
            write_json(case_dir / "validation.json", validation)
            manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        if proc is not None:
            manifest["cleanup"] = stop_owned_server(proc)
        manifest["wall_elapsed_s"] = time.time() - started
        # Detect a concurrent edit to the code/config snapshot used for the run.
        final_state = git_state()
        manifest["git_state_unchanged_during_run"] = final_state["file_manifest_sha256"] == state["file_manifest_sha256"]
        if manifest["status"] == "complete" and not manifest["git_state_unchanged_during_run"]:
            manifest["status"] = "failed"
            manifest["error"] = "Repository state changed during this case; result is not frozen"
        write_json(case_dir / "manifest.json", manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(manifest.get("error", "Case did not complete"))
    print(json.dumps({"case_dir": str(case_dir), "status": manifest["status"], "requests": len(workload)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
