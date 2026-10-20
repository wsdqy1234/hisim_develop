#!/usr/bin/env bash
# B2: Two Pools HiCache Tiered Capacity Pressure
# Fixed L1=40, sweep L2 KDA Host Slots: 3/32/128
# Enable L2/L3 (with file storage backend)
#
# Usage: bash run_b2.sh [l2_slots]
#   l2_slots = 3|32|128|all (default: all)
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HISIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKDIR=/tmp/opencode
MODEL_DIR=/home/l00886679/models/Kimi-K3-w4a8
SIM_CONF=$WORKDIR/config_k3_0519.json
PORT=30040
TRACE=$WORKDIR/k3_b1_trace.jsonl
NREQ=152

L2_VAL="${1:-all}"
if [ "$L2_VAL" = "all" ]; then
    L2_VALUES="3 32 128"
else
    L2_VALUES="$L2_VAL"
fi

# Shared env — HiCache ON with L2/L3 enabled
BASE_ENV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
BASE_ENV="$BASE_ENV HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
BASE_ENV="$BASE_ENV SGLANG_MAMBA_TRACK_INTERVAL=1024"
BASE_ENV="$BASE_ENV HISIM_DISABLE_PROACTIVE_EVICT=1"
BASE_ENV="$BASE_ENV HISIM_KDA_OBS=1"
BASE_ENV="$BASE_ENV HISIM_MAMBA_POOL_SIZE=40"
BASE_ENV="$BASE_ENV PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

run_b2() {
    local l2=$1
    local out="/tmp/hisim/b2_l2_${l2}"
    local log="$WORKDIR/b2_l2_${l2}_server.log"
    local blog="$WORKDIR/b2_l2_${l2}_bench.log"

    echo ""
    echo "============================================================"
    echo "[B2] L1=40, L2=$l2, HiCache ON (file storage), proactive=OFF"
    echo "============================================================"

    pkill -9 -f 'launch_server' 2>/dev/null
    sleep 2
    rm -rf "$out" /tmp/hisim/hicache
    rm -f "$log" "$blog"
    mkdir -p "$out"

    ENVV="$BASE_ENV"
    ENVV="$ENVV HISIM_L2_MAMBA_SIZE=$l2"
    ENVV="$ENVV HISIM_OUTPUT_DIR=$out"

    echo "[B2-l2_${l2}] Starting server (L1=40, L2=$l2, storage=file)..."
    env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
      --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
      --skip-server-warmup --port $PORT --enable-hierarchical-cache \
      --hicache-storage-backend file --hicache-size 80 \
      --max-total-tokens 200000 --trust-remote-code --device cpu --load-format dummy \
      --max-running-requests 4 --mamba-track-interval 1024 \
      > "$log" 2>&1 &
    SRV_PID=$!

    echo "[B2-l2_${l2}] Waiting for server..."
    HEALTHY=0
    for _ in $(seq 1 30); do
        sleep 10
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
    done
    if [ "$HEALTHY" != "1" ]; then
        echo "[B2-l2_${l2}] SERVER-FAIL"
        grep -iE "error|traceback|exception|assert" "$log" | tail -20
        return 1
    fi
    echo "[B2-l2_${l2}] Server ready."

    echo "[B2-l2_${l2}] Sending ${NREQ} reqs..."
    env $ENVV python3 -u -m hisim.simulation.bench_serving \
      --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
      --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts $NREQ \
      --host 127.0.0.1 --port $PORT > "$blog" 2>&1 &

    echo "[B2-l2_${l2}] Waiting for bench..."
    for _ in $(seq 1 60); do
        sleep 10
        grep -q "All prompts done\|RuntimeError\|requests failed\|Throughput" "$blog" 2>/dev/null && break
    done

    kill -9 $SRV_PID 2>/dev/null
    pkill -9 -f launch_server 2>/dev/null
    sleep 2
    echo "[B2-l2_${l2}] Done."
}

for l2 in $L2_VALUES; do
    run_b2 "$l2"
done

echo ""
echo "============================================================"
echo "[B2] All experiments complete. Results in /tmp/hisim/b2_l2_*"
echo "============================================================"
