#!/usr/bin/env bash
# A2 scenario runner — runs inside Docker container lfz-sglang
# Usage: bash /tmp/run_a2_scenario.sh <scenario_name>
#   scenario_name = l1_only|l2_only|l3_only|all_miss
set -u

SCEN="$1"
HISIM_ROOT=/home/d00954154/hisim
WORKDIR=/tmp/opencode
MODEL_DIR=$WORKDIR/k3-model
SIM_CONF=$WORKDIR/config_k3_0519.json
PORT=30020
TRACE=$WORKDIR/k3_a2_${SCEN}.jsonl
OUT=/tmp/hisim/a2_${SCEN}
LOG=$WORKDIR/a2_${SCEN}_server.log
BLOG=$WORKDIR/a2_${SCEN}_bench.log

pkill -9 -f 'launch_server' 2>/dev/null
sleep 2
rm -rf "$OUT" /tmp/hisim/hicache
rm -f "$LOG" "$BLOG"
mkdir -p "$OUT"

ENVV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
ENVV="$ENVV HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
ENVV="$ENVV HISIM_MAMBA_POOL_SIZE=4"
ENVV="$ENVV HISIM_L2_MAMBA_SIZE=3"
ENVV="$ENVV SGLANG_MAMBA_TRACK_INTERVAL=1024"
ENVV="$ENVV HISIM_DISABLE_PROACTIVE_EVICT=1"
ENVV="$ENVV HISIM_KDA_OBS=1"
ENVV="$ENVV HISIM_OUTPUT_DIR=$OUT"
ENVV="$ENVV PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

echo "[A2-${SCEN}] Starting server (L1=4, L2=3, proactive=OFF, obs=ON)..."
env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
  --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
  --skip-server-warmup --port $PORT --enable-hierarchical-cache \
  --hicache-storage-backend file --hicache-size 80 \
  --max-total-tokens 200000 --trust-remote-code --device cpu --load-format dummy \
  > "$LOG" 2>&1 &
SRV_PID=$!
echo "[A2-${SCEN}] Server PID=$SRV_PID"

echo "[A2-${SCEN}] Waiting for server..."
HEALTHY=0
for _ in $(seq 1 30); do
    sleep 10
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
done
if [ "$HEALTHY" != "1" ]; then
    echo "[A2-${SCEN}] SERVER-FAIL"
    grep -iE "error|traceback|exception|assert" "$LOG" | tail -20
    exit 2
fi
echo "[A2-${SCEN}] Server ready."

NREQ=$(python3 -c "import json; print(sum(1 for _ in open('$TRACE')))")
echo "[A2-${SCEN}] Sending ${NREQ} reqs..."

env $ENVV python3 -u -m hisim.simulation.bench_serving \
  --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
  --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts $NREQ \
  --host 127.0.0.1 --port $PORT > "$BLOG" 2>&1 &

echo "[A2-${SCEN}] Waiting for bench..."
for _ in $(seq 1 30); do
    sleep 10
    grep -q "All prompts done\|RuntimeError\|requests failed\|Throughput" "$BLOG" 2>/dev/null && break
done

# Stop server
kill -9 $SRV_PID 2>/dev/null
pkill -9 -f launch_server 2>/dev/null
sleep 2

echo "[A2-${SCEN}] Done."
