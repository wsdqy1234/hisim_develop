#!/usr/bin/env bash
# Run one K3 One Pool vs Two Pools experiment cell on local sglang 0.5.19.
# Usage: run_exp.sh <mode> <KDA> <MLA> <C> <chunk> <trace> <n> <label> [port]
#   mode : twopools | onepool
#   KDA  : mamba pool size (slots)            -> HISIM_MAMBA_POOL_SIZE
#   MLA  : --max-total-tokens                 (0 = leave default)
#   C    : SGLANG_MAMBA_TRACK_INTERVAL        (0 = leave default)
#   chunk: --chunked-prefill-size             (0 = unset)
# Env overrides:
#   WORKDIR   scratch dir for logs / traces / results (default /tmp/opencode)
#   MODEL_DIR mock K3 model dir (default $WORKDIR/k3-model)
#   SIM_CONF  sim config json  (default $WORKDIR/config_k3_0519.json)
#   NOL3=1    keep hierarchical cache (L2 host pool) but drop the L3 storage
#             backend (attribution runs). Without a backend sglang defaults
#             hicache_ratio to None, which crashes the mock host pool sizing;
#             an explicit --hicache-ratio is passed instead.
# NOTE: never invoke this from a shell whose command line contains the text
# "launch_server" (e.g. a py_compile path) — the pkill below would match and
# kill the calling shell.
set -u
MODE=$1; KDA=$2; MLA=$3; C=$4; CHUNK=$5; TRACE=$6; N=$7; LABEL=$8; PORT=${9:-30010}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HISIM_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
WORKDIR="${WORKDIR:-/tmp/opencode}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/k3-model}"
SIM_CONF="${SIM_CONF:-$WORKDIR/config_k3_0519.json}"

cd "$HISIM_ROOT"
OUT=/tmp/hisim/$LABEL
LOG=$WORKDIR/${LABEL}_server.log
BLOG=$WORKDIR/${LABEL}_bench.log
rm -rf "$OUT" /tmp/hisim/hicache
rm -f "$LOG" "$BLOG"

# --- kill any previous server/bench (this script's cmdline does NOT match) ---
pkill -9 -f 'hisim.simulation.sglang.launch[_]serv' 2>/dev/null
pkill -9 -f 'bench[_]serving' 2>/dev/null
sleep 3

ENVV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
ENVV="$ENVV HISIM_SIMULATION_MODE=OFFLINE HISIM_OUTPUT_DIR=$OUT HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
ENVV="$ENVV HISIM_MAMBA_POOL_SIZE=$KDA"
[ "$C" != "0" ] && ENVV="$ENVV SGLANG_MAMBA_TRACK_INTERVAL=$C"
ENVV="$ENVV HISIM_DISABLE_PROACTIVE_EVICT=1 PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

MLAARG=""; [ "$MLA" != "0" ] && MLAARG="--max-total-tokens $MLA"
CHUNKARG=""; [ "$CHUNK" != "0" ] && CHUNKARG="--chunked-prefill-size $CHUNK"
MODEARG=""; [ "$MODE" = "onepool" ] && MODEARG="--enable-unified-memory"
L3ARG="--hicache-storage-backend file --hicache-size 80"
[ "${NOL3:-0}" = "1" ] && L3ARG="--hicache-ratio 2"

setsid nohup env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
  --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
  --skip-server-warmup --port $PORT --enable-hierarchical-cache \
  $L3ARG \
  $MLAARG --trust-remote-code --device cpu --load-format dummy $CHUNKARG $MODEARG \
  > "$LOG" 2>&1 &
disown

HEALTHY=0
for _ in $(seq 1 24); do
  sleep 10
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
done
if [ "$HEALTHY" != "1" ]; then
  echo "SERVER-FAIL $LABEL"; grep -iE "error|traceback|exception|assert" "$LOG" | tail -15; exit 2
fi

setsid nohup env $ENVV python3 -u -m hisim.simulation.bench_serving \
  --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
  --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts "$N" \
  --host 127.0.0.1 --port $PORT > "$BLOG" 2>&1 &
disown

# wait for bench to finish (metrics.json gets rewritten by the final profile)
for _ in $(seq 1 30); do
  sleep 10
  grep -q "All prompts done\|RuntimeError\|requests failed\|Throughput" "$BLOG" 2>/dev/null && break
done

echo "=== $LABEL done ==="
python3 - "$OUT" "$LABEL" <<'PY'
import json, os, sys
out, label = sys.argv[1], sys.argv[2]
p = os.path.join(out, "metrics.json")
if not os.path.exists(p):
    print(f"{label}: NO metrics.json"); raise SystemExit
m = json.load(open(p))
keys = ["num_completed_requests","mla_prefix_hit_tokens_agg","mla_prefix_hit_l1_tokens_agg",
        "mla_prefill_new_tokens_agg","kda_write_calls","kda_load_calls","kda_l3_writes","kda_l3_hits",
        "kda_checkpoint_donate_count","kda_checkpoint_hit_count","kda_cow_exec_count",
        "mean_e2e_latency_ms","peak_mamba_live_pages","peak_mla_live_pages"]
print(label + ":")
for k in keys:
    if k in m: print(f"  {k:32s}= {m[k]}")
PY
