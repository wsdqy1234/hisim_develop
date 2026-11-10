# Reproducible Qwen3 Host-prefix cases

Run this tool **inside** the existing `dqy-hisim-qwen3` container. It uses only
Python's standard library as a client; the case server uses the existing HiSim
and SGLang environment. It neither installs packages nor changes the predictor.

```bash
docker exec -e PYTHONDONTWRITEBYTECODE=1 dqy-hisim-qwen3 \
  python3 -B /home/d00954154/HiSim_Qwen3/test/research_hbf/run_host_case.py \
  --case-id short_mixed --port 30231 \
  --output-root /home/d00954154/hbf_research_results \
  --scenario mixed --prefix-tokens 340 --concurrency 2 --reuse 1
```

Add `--prepare-only` to freeze requests, logical references, configurations,
versions, and the Git working-state digest without starting a server. Every
case requires a **new** directory: reruns must choose a new case ID. Use an
output root outside the repository so generated results do not change the
code-state digest. No shared temporary/output directory is deleted. The server
has a new process group; cleanup signals only that group, including on errors.
Readiness uses the startup log plus a TCP listener probe. It never calls the
generation-backed `/health` endpoint, which would contaminate the workload.

`--scenario` is `dram`, `mixed` (70% HBF logical Host slots), or `hbf` (all but
one compatibility anchor token in HBF). `--prefix-tokens` is 340, 4096, or 16384.
The default unique suffix is 8 tokens and generated output is 16 tokens.
`--reuse` counts reuse waves; each wave requests `--concurrency` distinct build
objects. There are `concurrency+2` initial build objects. `--churn N` inserts N
new distinct objects before each reuse wave. Arrivals are distinct frozen
virtual timestamps, not the HTTP client send order.

Default L1 capacity fits one complete concurrency wave plus margin. Default
Host capacity fits builds plus one reinsertion wave. Explicit
`--max-total-tokens` / `--host-capacity-tokens` can introduce pressure; capacity
rule results are written in `effective_config.json`. Configured logical Host
capacity is distinct from the 800GB physical 8-WIN supply model. The 8-WIN
envelope keeps media parameters at 200/100/10 decimal GB/s per WIN.
Read latency is 7.5 microseconds (midpoint of the expert 5–10 microsecond range),
and write latency is an uncalibrated 100 microsecond assumption, matching the
current `WinSpec` defaults. These latencies are recorded in every manifest.
`--interface-gbps` means **decimal GB/s**, with 0 representing an unbounded
interface hypothesis. `--stripe-block-tokens` controls physical WIN striping;
`--block-tokens` controls logical analysis blocks independently.

Every case keeps `HISIM_SHARED_DRAM=1`, L3 write barrier, BF16, TP=PP=1, and the
same baseline platform/predictor JSON. Previously exported HISIM parameters are
removed from the server environment before explicit case parameters are set.
`--platform-config` is available for an explicit baseline snapshot; an already
HBF-enabled platform config is rejected. This runner does not optimize TTFT.

## Files and scope

- `workload.jsonl`: every request's stable ID, input token IDs, arrival, phase,
  and generated output length. These exact IDs are submitted via `/generate`.
- `logical_prefix_accesses.jsonl`: **all logical reusable-prefix references,
  including misses**. These are not observed Host accesses: the simulator may
  serve content from HBM, DRAM, HBF, L3, or recomputation.
- `platform_config.json`, `win_config.json`, `effective_config.json`: frozen
  inputs, command, environment, geometry, and capacity accounting.
- `git_state.json`, `manifest.json`: HEAD/branch, tracked diff, every tracked and
  untracked file's SHA256, aggregate digest, versions, parameters, completion,
  errors, PID/process-group cleanup, and concurrent-code-change detection.
- `server.log`, `responses.jsonl`: complete server log and all client responses.
  Response wall time is diagnostic, not simulated TTFT/TPOT.
- `metrics.json`, `request.jsonl`, `iteration.jsonl`, `io_trace.jsonl`,
  `win_pool_summary.json` (when HBF enabled), and other simulator exports remain
  in the same case directory. `validation.json` checks completed request count,
  exact request IDs, nonnegative token time, request token/byte conservation,
  generic-vs-WIN I/O byte reconciliation, child byte/slot conservation, and
  immutable slot-to-WIN routes. A repository edit during the run also marks
  the manifest failed. Failed cases are retained and never silently repaired.
  Byte accounting uses `path=real` only. Placeholder tensor-copy paths such as
  `mock_lbk` remain separately counted diagnostics and are not actual KV bytes.

`logical_prefix_accesses.jsonl` schema version 1 has one row per request:

```json
{
  "schema_version": 1,
  "scope": "logical_reusable_prefix_references",
  "request_id": "short_mixed-r00004",
  "arrival_s": 12.0,
  "phase": "reuse",
  "prefix_object_index": 0,
  "prefix_id": "sha256:<full-prefix-content-hash>",
  "prefix_tokens": 340,
  "suffix_tokens": 8,
  "suffix_id": "sha256:<unique-suffix-content-hash>",
  "input_tokens": 348,
  "requested_output_tokens": 16,
  "bytes_per_token": 114688,
  "block_tokens": 128,
  "blocks": [
    {"id": "sha256:<ancestor-inclusive-content-hash>", "tokens": 128, "bytes": 14680064}
  ]
}
```

Actual block arrays cover the complete reusable prefix, including the partial
last block. IDs hash little-endian uint32 token IDs; a block ID hashes the full
prefix through that block, including ancestors. Same content has identical IDs
across requests and scenarios. Suffix and generated output are excluded from
the reusable-prefix blocks and their bytes. Offline LRU capacity/reuse curves
derived from this stream are explicit policy hypotheses, not measurements of
the real Host tier. Actual Host traffic is in `io_trace.jsonl`.

Lightweight checks (no server):

```bash
docker exec -e PYTHONDONTWRITEBYTECODE=1 dqy-hisim-qwen3 \
  python3 -B -m pytest -p no:cacheprovider -q \
  /home/d00954154/HiSim_Qwen3/test/research_hbf/test_run_host_case.py
```
