#!/bin/bash
# Phase 2C.2: Deterministic experiment runner
# Fixes: --random-seed=42, PYTHONHASHSEED=42, identity fix, ancestor access recording
#
# Usage: bash run_p2c2.sh [trace] [policy] [tag] [base_prob]
#   trace:   path to trace file
#   policy:  native|neutral|hotness|depth|cost_aware|oracle
#   tag:     experiment tag (e.g. b2_fork)
#   base_prob: for cost_aware, 0.0/0.01/0.05/0.10 (default 0.0)

set -e

TRACE="${1:?trace required}"
POLICY="${2:?policy required}"
TAG="${3:?tag required}"
BASE_PROB="${4:-0.0}"

OUTDIR="/tmp/hisim/p2c2_${TAG}_${POLICY}"
if [ "$BASE_PROB" != "0.0" ]; then
    OUTDIR="${OUTDIR}_bp${BASE_PROB}"
fi

# Clean up
fuser -k 30050/tcp 2>/dev/null; sleep 5
rm -rf "$OUTDIR" /tmp/hisim/hicache; mkdir -p "$OUTDIR"

# Environment — DETERMINISTIC
export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONHASHSEED=42  # Deterministic dict hash ordering
export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1
export HISIM_L2_MAMBA_SIZE=0 HISIM_DISABLE_L3=1 HISIM_MAMBA_POOL_SIZE=16
export HISIM_KDA_POLICY=$POLICY HISIM_OUTPUT_DIR=$OUTDIR HISIM_HOTNESS_THRESHOLD=0
export HISIM_COST_AWARE_BASE_PROB=$BASE_PROB
export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd

# Clear pycache for code changes to take effect
find /home/d00954154/hisim/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

[ "$POLICY" = "oracle" ] && export HISIM_ORACLE_TRACE=$TRACE

LOGTAG="${TAG}_${POLICY}"
[ "$BASE_PROB" != "0.0" ] && LOGTAG="${LOGTAG}_bp${BASE_PROB}"

echo "[$LOGTAG] Starting server..."
setsid python3 -u -m hisim.simulation.sglang.launch_server \
    --model-path /home/l00886679/models/Kimi-K3-w4a8 \
    --sim-config-path /tmp/opencode/config_k3_0519.json \
    --skip-server-warmup --port 30050 \
    --enable-hierarchical-cache --hicache-size 80 \
    --max-total-tokens 200000 --trust-remote-code \
    --device cpu --load-format dummy \
    --max-running-requests 4 --mamba-track-interval 1024 \
    --random-seed 42 \
    > /tmp/opencode/p2c2_${LOGTAG}_srv.log 2>&1 &
SRV=$!

# Wait for server (up to 5 minutes)
for i in $(seq 1 30); do
    sleep 10
    grep -q "HiSim hicache attached" /tmp/opencode/p2c2_${LOGTAG}_srv.log 2>/dev/null && {
        sleep 15
        curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1 && break
    }
done

if ! curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1; then
    echo "[$LOGTAG] NOT READY"
    kill -9 $SRV 2>/dev/null; fuser -k 30050/tcp 2>/dev/null; sleep 5
    exit 1
fi

echo "[$LOGTAG] Server healthy, running bench..."
python3 -u -m hisim.simulation.bench_serving \
    --warmup-requests 0 --model /home/l00886679/models/Kimi-K3-w4a8 \
    --bench-mode simulation --dataset-name hisim-collection \
    --dataset-path "$TRACE" --num-prompts 56 \
    --host 127.0.0.1 --port 30050 \
    > /tmp/opencode/p2c2_${LOGTAG}_bench.log 2>&1 || true

kill -9 $SRV 2>/dev/null; fuser -k 30050/tcp 2>/dev/null; sleep 8

echo -n "[$LOGTAG] RESULT: "
if [ -f "$OUTDIR/metrics.json" ]; then
    python3 /tmp/check_b1_metrics.py "$OUTDIR/metrics.json" 2>/dev/null \
        | grep -E "cow_exec|replay_tokens|double_free" | tr '\n' ' '
    echo ""
else
    echo "NO METRICS"
fi
