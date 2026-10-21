#!/usr/bin/env bash
# Phase 2A minimal strategy verification
# KDA=8, L1-only, three policies: native / hotness / oracle
set -u
HISIM_ROOT=/home/d00954154/hisim
WORKDIR=/tmp/opencode
MODEL_DIR=/home/l00886679/models/Kimi-K3-w4a8
SIM_CONF=$WORKDIR/config_k3_0519.json
PORT=30050
TRACE=$WORKDIR/k3_p2a_trace.jsonl
NREQ=$(python3 -c "import json; print(sum(1 for _ in open('$TRACE')))" 2>/dev/null || echo 64)

BASE_ENV="SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
BASE_ENV="$BASE_ENV HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1"
BASE_ENV="$BASE_ENV SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1"
BASE_ENV="$BASE_ENV HISIM_L2_MAMBA_SIZE=0 HISIM_DISABLE_L3=1"
BASE_ENV="$BASE_ENV HISIM_MAMBA_POOL_SIZE=16"
BASE_ENV="$BASE_ENV PYTHONPATH=$HISIM_ROOT/src:$HISIM_ROOT/3rdparty/kvcm_l3:$HISIM_ROOT/3rdparty/simpyssd"

run() {
    local pol=$1
    local out="/tmp/hisim/p2a_${pol}"
    local log="$WORKDIR/p2a_${pol}_server.log"
    local blog="$WORKDIR/p2a_${pol}_bench.log"
    echo "[P2A] policy=$pol KDA=8"
    pkill -9 -f launch_server 2>/dev/null; sleep 2
    rm -rf "$out" /tmp/hisim/hicache "$log" "$blog"; mkdir -p "$out"
    ENVV="$BASE_ENV HISIM_KDA_POLICY=$pol HISIM_OUTPUT_DIR=$out"
    env $ENVV python3 -u -m hisim.simulation.sglang.launch_server \
      --model-path "$MODEL_DIR" --sim-config-path "$SIM_CONF" \
      --skip-server-warmup --port $PORT --enable-hierarchical-cache \
      --hicache-size 80 --max-total-tokens 200000 --trust-remote-code \
      --device cpu --load-format dummy --max-running-requests 4 \
      --mamba-track-interval 1024 > "$log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 30); do sleep 10; curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; done
    env $ENVV python3 -u -m hisim.simulation.bench_serving \
      --warmup-requests 0 --model "$MODEL_DIR" --bench-mode simulation \
      --dataset-name hisim-collection --dataset-path "$TRACE" --num-prompts $NREQ \
      --host 127.0.0.1 --port $PORT > "$blog" 2>&1 &
    for _ in $(seq 1 60); do sleep 10; grep -q "All prompts done\|RuntimeError\|Throughput" "$blog" 2>/dev/null && break; done
    kill -9 $SRV 2>/dev/null; pkill -9 -f launch_server 2>/dev/null; sleep 2
    echo "[P2A] $pol done"
}

for p in native hotness; do
    run "$p"
done
echo "[P2A] All done"
