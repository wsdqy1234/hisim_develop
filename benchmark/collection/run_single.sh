#!/bin/bash
# Single experiment runner for Phase 2D.2
# Usage: bash run_single.sh <config_name> <trace_file> <extra_env_exports>
set -e

NAME=$1
TRACE=$2
EXTRA_ENV=$3
PORT=30060
MODEL=/home/l00886679/models/Kimi-K3-w4a8
CONFIG=/tmp/opencode/config_k3_0519.json
RESULTS_DIR=/tmp/opencode/p2d2_results
mkdir -p $RESULTS_DIR

export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7 PYTHONHASHSEED=42
export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1
export HISIM_DISABLE_L3=0 HISIM_MAMBA_POOL_SIZE=16 HISIM_KDA_POLICY=native
export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd

# Unset previous config
unset HISIM_L2_KV_SIZE_GB HISIM_L2_TOTAL_GB HISIM_L2_ONE_POOL HISIM_L2_BORROW_POLICY HISIM_L2_MAMBA_SIZE

# Apply extra env
eval "$EXTRA_ENV"
echo "  ENV: HISIM_L2_KV_SIZE_GB=$HISIM_L2_KV_SIZE_GB HISIM_L2_TOTAL_GB=$HISIM_L2_TOTAL_GB HISIM_L2_MAMBA_SIZE=$HISIM_L2_MAMBA_SIZE HISIM_L2_ONE_POOL=$HISIM_L2_ONE_POOL"

logfile=$RESULTS_DIR/${NAME}_srv.log
benchlog=$RESULTS_DIR/${NAME}_bench.log

echo "[$(date +%H:%M:%S)] Starting: $NAME"

# Kill existing
pkill -9 -f "launch_server" 2>/dev/null || true
sleep 5
# Also kill anything on the port directly
for i in $(seq 1 10); do
    ss -tlnp 2>/dev/null | grep -q $PORT || break
    sleep 2
done

# Clear pycache
cd /home/d00954154/hisim
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/hisim/hicache 2>/dev/null || true

# Start server
python3 -u -m hisim.simulation.sglang.launch_server \
    --model-path $MODEL \
    --sim-config-path $CONFIG \
    --skip-server-warmup --port $PORT \
    --enable-hierarchical-cache --hicache-size 80 \
    --max-total-tokens 200000 --trust-remote-code \
    --device cpu --load-format dummy \
    --max-running-requests 4 --mamba-track-interval 1024 \
    --random-seed 42 > $logfile 2>&1 &
SRV_PID=$!

# Wait for ready
ready=0
for i in $(seq 1 36); do
    sleep 10
    if grep -q "HiSim hicache attached" $logfile 2>/dev/null; then
        sleep 10
        curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    fi
    if grep -q "OSError\|HFValidationError" $logfile 2>/dev/null; then
        echo "  FAILED: model path error"
        break
    fi
done

if [ $ready -eq 0 ]; then
    echo "  ERROR: Server not ready"
    kill -TERM $SRV_PID 2>/dev/null || true; sleep 2; kill -9 $SRV_PID 2>/dev/null || true
    exit 1
fi

echo "  Server ready, running benchmark..."

# Run benchmark
python3 -u -m hisim.simulation.bench_serving \
    --warmup-requests 0 --model $MODEL \
    --bench-mode simulation --dataset-name hisim-collection \
    --dataset-path "$TRACE" --num-prompts 56 \
    --host 127.0.0.1 --port $PORT > $benchlog 2>&1 || true

# Kill server
kill -TERM $SRV_PID 2>/dev/null || true; sleep 3; kill -9 $SRV_PID 2>/dev/null || true

echo "[$(date +%H:%M:%S)] Done: $NAME"
