#!/usr/bin/env bash
# B1: Two Pools L1-only Capacity Pressure
# KDA Slots sweep: 40/80/165
# No hierarchical cache (pure L1, native LRU eviction)
#
# Usage: bash run_b1.sh [kda_slots]
#   kda_slots = 40|80|165|all (default: all)
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HISIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKDIR=/tmp/opencode
MODEL_DIR=/tmp/simple-k3-model
SIM_CONF=$WORKDIR/config_k3_0519.json
PORT=30030

KDA_VAL="${1:-all}"
if [ "$KDA_VAL" = "all" ]; then
    KDA_VALUES="40 80 165"
else
    KDA_VALUES="$KDA_VAL"
fi

# Generate trace
python3 "$SCRIPT_DIR/gen_k3_b1_trace.py" "$WORKDIR"
TRACE="$WORKDIR/k3_b1_trace.jsonl"
NREQ=$(python3 -c "import json; print(sum(1 for _ in open('$TRACE')))")

# Shared env — HiCache enabled but L2=0 (no KDA L2 backup) and no storage
# backend (no L3). This gives L1-only behavior while keeping the unified
# radix cache properly initialized.
BASE_ENV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
BASE_ENV="$BASE_ENV HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
BASE_ENV="$BASE_ENV SGLANG_MAMBA_TRACK_INTERVAL=1024"
BASE_ENV="$BASE_ENV HISIM_DISABLE_PROACTIVE_EVICT=1"
BASE_ENV="$BASE_ENV HISIM_KDA_OBS=1"
BASE_ENV="$BASE_ENV HISIM_L2_MAMBA_SIZE=0"
BASE_ENV="$BASE_ENV PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

run_b1() {
    local kda=$1
    local out="/tmp/hisim/b1_kda${kda}"
    local log="$WORKDIR/b1_kda${kda}_server.log"
    local blog="$WORKDIR/b1_kda${kda}_bench.log"

    echo ""
    echo "============================================================"
    echo "[B1] KDA=$kda, L1-only (no HiCache), max_running=4, track=1024"
    echo "============================================================"

    pkill -9 -f 'launch_server' 2>/dev/null
    sleep 2
    rm -rf "$out" /tmp/hisim/hicache
    rm -f "$log" "$blog"
    mkdir -p "$out"

    ENVV="$BASE_ENV"
    ENVV="$ENVV HISIM_MAMBA_POOL_SIZE=$kda"
    ENVV="$ENVV HISIM_OUTPUT_DIR=$out"

    echo "[B1-kda${kda}] Starting server (KDA=$kda, HiCache ON, L2=0, no storage, proactive=OFF)..."
    env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
      --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
      --skip-server-warmup --port $PORT --enable-hierarchical-cache \
      --hicache-size 80 \
      --max-total-tokens 200000 --trust-remote-code --device cpu --load-format dummy \
      --max-running-requests 4 --mamba-track-interval 1024 \
      > "$log" 2>&1 &
    SRV_PID=$!
    echo "[B1-kda${kda}] Server PID=$SRV_PID"

    echo "[B1-kda${kda}] Waiting for server..."
    HEALTHY=0
    for _ in $(seq 1 30); do
        sleep 10
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
    done
    if [ "$HEALTHY" != "1" ]; then
        echo "[B1-kda${kda}] SERVER-FAIL"
        grep -iE "error|traceback|exception|assert" "$log" | tail -20
        return 1
    fi
    echo "[B1-kda${kda}] Server ready."

    echo "[B1-kda${kda}] Sending ${NREQ} reqs..."
    env $ENVV python3 -u -m hisim.simulation.bench_serving \
      --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
      --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts $NREQ \
      --host 127.0.0.1 --port $PORT > "$blog" 2>&1 &

    echo "[B1-kda${kda}] Waiting for bench..."
    for _ in $(seq 1 60); do
        sleep 10
        grep -q "All prompts done\|RuntimeError\|requests failed\|Throughput" "$blog" 2>/dev/null && break
    done

    kill -9 $SRV_PID 2>/dev/null
    pkill -9 -f launch_server 2>/dev/null
    sleep 2
    echo "[B1-kda${kda}] Done."
}

for kda in $KDA_VALUES; do
    run_b1 "$kda"
done

echo ""
echo "============================================================"
echo "[B1] All experiments complete. Results in /tmp/hisim/b1_kda*"
echo "============================================================"
