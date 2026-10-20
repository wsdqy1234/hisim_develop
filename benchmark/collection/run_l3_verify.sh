#!/usr/bin/env bash
# KDA L3 Reusability Verification Experiment
# Tests: Build → Evict L1→L2→L3 → Reuse (find in L3)
# Usage: bash run_l3_verify.sh
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HISIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKDIR=/tmp/opencode
MODEL_DIR=$WORKDIR/k3-model
SIM_CONF=$WORKDIR/config_k3_0519.json
TRACE=$WORKDIR/k3_l3_verify.jsonl
LABEL=l3_verify
PORT=30010

cd "$HISIM_ROOT"
OUT=/tmp/hisim/$LABEL
LOG=$WORKDIR/${LABEL}_server.log
BLOG=$WORKDIR/${LABEL}_bench.log
rm -rf "$OUT" /tmp/hisim/hicache
rm -f "$LOG" "$BLOG"
mkdir -p "$OUT"

pkill -9 -f 'hisim.simulation.sglang.launch[_]serv' 2>/dev/null
pkill -9 -f 'bench[_]serving' 2>/dev/null
sleep 3

ENVV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
ENVV="$ENVV HISIM_SIMULATION_MODE=OFFLINE HISIM_OUTPUT_DIR=$OUT HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
ENVV="$ENVV HISIM_MAMBA_POOL_SIZE=10"           # small L1 (10 slots → ~3 sessions)
ENVV="$ENVV HISIM_L2_MAMBA_SIZE=3"              # small L2 (3 slots → force L3 overflow)
ENVV="$ENVV SGLANG_MAMBA_TRACK_INTERVAL=1024"
ENVV="$ENVV HISIM_DISABLE_PROACTIVE_EVICT=1"    # clean baseline
ENVV="$ENVV HISIM_KDA_OBS=1"                     # enable checkpoint event logging
ENVV="$ENVV PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

echo "[l3_verify] Starting server (KDA=10, L2=3, C=1024, proactive=OFF, obs=ON)..."
setsid nohup env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
  --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
  --skip-server-warmup --port $PORT --enable-hierarchical-cache \
  --hicache-storage-backend file --hicache-size 80 \
  --max-total-tokens 200000 --trust-remote-code --device cpu --load-format dummy \
  > "$LOG" 2>&1 &
disown

echo "[l3_verify] Waiting for server..."
HEALTHY=0
for _ in $(seq 1 30); do
  sleep 10
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
done
if [ "$HEALTHY" != "1" ]; then
  echo "SERVER-FAIL"; grep -iE "error|traceback|exception|assert" "$LOG" | tail -20; exit 2
fi
echo "[l3_verify] Server ready. Sending trace (24 reqs)..."

setsid nohup env $ENVV python3 -u -m hisim.simulation.bench_serving \
  --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
  --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts 24 \
  --host 127.0.0.1 --port $PORT > "$BLOG" 2>&1 &
disown

echo "[l3_verify] Waiting for bench to complete..."
for _ in $(seq 1 30); do
  sleep 10
  grep -q "All prompts done\|RuntimeError\|requests failed\|Throughput" "$BLOG" 2>/dev/null && break
done

echo "=== l3_verify done ==="
echo "==== metrics.json ===="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
p = os.path.join(out, "metrics.json")
if not os.path.exists(p):
    print("NO metrics.json"); raise SystemExit
m = json.load(open(p))
keys = ["num_completed_requests","mla_prefix_hit_tokens_agg","mla_prefill_new_tokens_agg",
        "kda_write_calls","kda_load_calls","kda_l3_writes","kda_l3_hits",
        "kda_checkpoint_donate_count","kda_checkpoint_hit_count","kda_cow_exec_count",
        "kda_historical_replay_tokens","kda_active_slots","kda_pool_capacity",
        "kda_l2_cached_slots","kda_l2_capacity",
        "mean_e2e_latency_ms","mean_ttft_ms"]
for k in keys:
    if k in m: print(f"  {k:32s}= {m[k]}")
print()
# Check checkpoint events
ev = os.path.join(out, "checkpoint_events.jsonl")
if os.path.exists(ev):
    events = [json.loads(l) for l in open(ev)]
    print(f"==== checkpoint_events.jsonl: {len(events)} events ====")
    from collections import Counter
    types = Counter(e["event"] for e in events)
    for t, c in types.most_common():
        print(f"  {t:20s}= {c}")
else:
    print("==== checkpoint_events.jsonl: NOT FOUND ====")
PY
echo "==== server log KDA lines ===="
grep -E "\[KDA-|COW-EXEC|REPLAY-FALLBACK|P2 match|l3_" "$LOG" | head -30
echo "==== bench log ===="
tail -5 "$BLOG"
