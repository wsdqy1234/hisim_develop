#!/usr/bin/env bash
# HiSparse validation driver: baseline vs sparse replay + analytic sweep.
#
#   BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
#   PYTHONPATH=src bash benchmark/hisparse/run_validation.sh
#
# The two replay runs execute in separate processes on purpose: hisim keeps
# process-global pool/state caches and the hook install is idempotent only
# within one process.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
OUT_DIR="${HISIM_VALIDATION_OUT_DIR:-$SCRIPT_DIR/results}"
CONFIG="${HISIM_CONFIG_PATH:-$ROOT/test/assets/mock/config_hisparse.json}"
MODEL_PATH="${BENCHMARK_TEST_MODEL_PATH:-Qwen/Qwen3-0.6B}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HISIM_CONFIG_PATH="$CONFIG"
export BENCHMARK_TEST_MODEL_PATH="$MODEL_PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_SIMULATION_MODE=OFFLINE

run_one() {
    local name="$1" physical="$2" enable="$3"
    local args=(bench --output "$OUT_DIR/$name.json" --max-total-tokens "$physical")
    if [[ "$enable" == "1" ]]; then
        args+=(--enable)
    fi
    echo "===== $name (physical=$physical, hisparse=$enable) ====="
    python "$SCRIPT_DIR/hisparse_bench.py" "${args[@]}"
}

echo "===== analytic sweep (DeepSeek-V3.2 geometry) ====="
python "$SCRIPT_DIR/hisparse_bench.py" sweep --output "$OUT_DIR/sweep.json"

# Capacity-pressure scenario: 20 requests x (1024 prompt + 64 decode) need
# 21760 logical tokens; the baseline HBM pool only holds 20000.
run_one "baseline_capacity" 20000 0
run_one "sparse_capacity" 20000 1

# Roomy scenario: both fit, isolating the per-step swap-in overhead.
run_one "baseline_roomy" 40000 0
run_one "sparse_roomy" 40000 1

# Same but assuming shared-index prefetch (DSA index_topk_freq=4): only the
# anchor layer's swap-in is on the critical path.
(
    export HISIM_HISPARSE_SHARED_INDEX_FREQ=4
    run_one "sparse_roomy_freq4" 40000 1
)

python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for name in (
    "baseline_capacity",
    "sparse_capacity",
    "baseline_roomy",
    "sparse_roomy",
    "sparse_roomy_freq4",
):
    path = out / f"{name}.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    rows.append(
        {
            "run": name,
            "completed": m.get("completed"),
            "ttft_ms": round(m.get("mean_ttft_ms", 0.0), 3),
            "tpot_ms": round(m.get("mean_tpot_ms", 0.0), 3),
            "req_per_s": round(m.get("request_throughput", 0.0), 2),
            "hit_rate": round(m.get("hisparse_hit_rate", 0.0), 4),
            "h2d_gb": round(m.get("hisparse_h2d_bytes", 0) / 1e9, 3),
            "d2h_gb": round(m.get("hisparse_d2h_bytes", 0) / 1e9, 3),
            "staging_gb": round(m.get("hisparse_staging_bytes", 0) / 1e9, 3),
            "mean_load_us": round(m.get("hisparse_mean_load_us", 0.0), 2),
            "hbm_kv_gb": round(m.get("hisparse_hbm_kv_bytes", 0) / 1e9, 2),
            "host_kv_gb": round(m.get("hisparse_host_kv_bytes", 0) / 1e9, 2),
        }
    )
summary = {"runs": rows, "sweep": json.loads((out / "sweep.json").read_text())}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "Results written to $OUT_DIR"
