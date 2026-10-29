#!/bin/bash
# Phase 2D.2: Robust experiment runner
# Uses PID tracking and proper port cleanup
cd /home/d00954154/hisim

export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7 PYTHONHASHSEED=42
export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1
export HISIM_DISABLE_L3=0 HISIM_MAMBA_POOL_SIZE=16 HISIM_KDA_POLICY=native
export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd

MODEL=/home/l00886679/models/Kimi-K3-w4a8
CONFIG=/tmp/opencode/config_k3_0519.json
RESULTS_DIR=/tmp/opencode/p2d2_results
mkdir -p $RESULTS_DIR

run_one() {
    local name=$1 trace=$2 port=$3
    shift 3
    local logfile=$RESULTS_DIR/${name}_srv.log
    local benchlog=$RESULTS_DIR/${name}_bench.log

    echo "[$(date +%H:%M:%S)] === $name (port=$port) ==="

    # Clear pycache
    find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    rm -rf /tmp/hisim/hicache 2>/dev/null

    # Unset all L2 env vars, then apply this config's env
    unset HISIM_L2_KV_SIZE_GB HISIM_L2_TOTAL_GB HISIM_L2_ONE_POOL HISIM_L2_BORROW_POLICY HISIM_L2_MAMBA_SIZE
    "$@"

    # Start server
    python3 -u -m hisim.simulation.sglang.launch_server \
        --model-path $MODEL --sim-config-path $CONFIG \
        --skip-server-warmup --port $port \
        --enable-hierarchical-cache --hicache-size 80 \
        --max-total-tokens 200000 --trust-remote-code \
        --device cpu --load-format dummy \
        --max-running-requests 4 --mamba-track-interval 1024 \
        --random-seed 42 > $logfile 2>&1 &
    local pid=$!
    echo "  Server PID=$pid"

    # Wait for ready
    local ready=0
    for i in $(seq 1 30); do
        sleep 10
        grep -q "HiSim hicache attached" $logfile 2>/dev/null || true
        if grep -q "Application startup complete" $logfile 2>/dev/null; then
            sleep 5
            curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ready=1; break; }
        fi
        if grep -q "address already in use" $logfile 2>/dev/null; then
            echo "  PORT CONFLICT — killing and retrying"
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            sleep 5
            # Retry
            python3 -u -m hisim.simulation.sglang.launch_server \
                --model-path $MODEL --sim-config-path $CONFIG \
                --skip-server-warmup --port $port \
                --enable-hierarchical-cache --hicache-size 80 \
                --max-total-tokens 200000 --trust-remote-code \
                --device cpu --load-format dummy \
                --max-running-requests 4 --mamba-track-interval 1024 \
                --random-seed 42 > $logfile 2>&1 &
            pid=$!
            echo "  Retry PID=$pid"
            for j in $(seq 1 20); do
                sleep 10
                grep -q "Application startup complete" $logfile 2>/dev/null && {
                    sleep 5
                    curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ready=1; break; }
                }
            done
            break
        fi
    done

    if [ $ready -eq 0 ]; then
        echo "  FAILED to start"
        kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
        return 1
    fi
    echo "  Server ready"

    # Run benchmark
    python3 -u -m hisim.simulation.bench_serving \
        --warmup-requests 0 --model $MODEL \
        --bench-mode simulation --dataset-name hisim-collection \
        --dataset-path "$trace" --num-prompts 56 \
        --host 127.0.0.1 --port $port > $benchlog 2>&1 || true
    echo "  Benchmark done"

    # Kill server by PID and wait
    kill -TERM $pid 2>/dev/null
    sleep 5
    kill -9 $pid 2>/dev/null
    wait $pid 2>/dev/null

    # Verify port freed
    for i in $(seq 1 10); do
        ss -tlnp 2>/dev/null | grep -q ":$port " || break
        sleep 2
    done
    echo "[$(date +%H:%M:%S)] === $name DONE ==="
}

# Config helper functions
cfg_leg()   { export HISIM_L2_KV_SIZE_GB=0.383 HISIM_L2_MAMBA_SIZE=8; }
cfg_2p8()   { export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=8; }
cfg_2p16()  { export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=16; }
cfg_1p_sf() { export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=self_first; }
cfg_1p_of() { export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=other_first; }

# Run all experiments with unique ports
PORT=30060
for trace_file in /tmp/opencode/k3_p2d_kda_to_mla.jsonl /tmp/opencode/k3_p2d_mla_to_kda.jsonl; do
    tname=$(basename "$trace_file" .jsonl | sed 's/k3_p2d_//')
    for cfg in leg:cfg_leg 2p8:cfg_2p8 2p16:cfg_2p16 1p_sf:cfg_1p_sf 1p_of:cfg_1p_of; do
        cname=${cfg%%:*}
        cfn=${cfg##*:}
        exp="p2d2_${cname}_${tname}"
        run_one "$exp" "$trace_file" $PORT "$cfn"
        PORT=$((PORT + 1))
    done
done

echo "=== ALL DONE ==="
