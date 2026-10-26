#!/bin/bash
# Phase 2C.3: L2 Placement comparison experiments
# L1 fixed to Native LRU, L2 policy varies, L2 size varies
LOGFILE=/tmp/opencode/p2c3_l2_results.log

run_one() {
    local trace=$1 l2_policy=$2 l2_size=$3 tag=$4 base_prob=${5:-0.0}
    local outdir="/tmp/hisim/p2c3_${tag}"
    local logtag="${tag}"

    pkill -f "launch_server" 2>/dev/null || true; sleep 5
    rm -rf "$outdir" /tmp/hisim/hicache; mkdir -p "$outdir"

    export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export PYTHONHASHSEED=42
    export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
    export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1
    export HISIM_L2_MAMBA_SIZE=$l2_size HISIM_DISABLE_L3=0
    export HISIM_MAMBA_POOL_SIZE=16
    export HISIM_KDA_POLICY=native HISIM_OUTPUT_DIR=$outdir
    export HISIM_L2_POLICY=$l2_policy
    export HISIM_L2_COST_BASE_PROB=$base_prob
    export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd
    find /home/d00954154/hisim/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    echo "[$(date +%H:%M:%S)] [$logtag] L2=$l2_size policy=$l2_policy bp=$base_prob Starting..." >> $LOGFILE
    setsid python3 -u -m hisim.simulation.sglang.launch_server \
        --model-path /home/l00886679/models/Kimi-K3-w4a8 \
        --sim-config-path /tmp/opencode/config_k3_0519.json \
        --skip-server-warmup --port 30050 \
        --enable-hierarchical-cache --hicache-size 80 \
        --max-total-tokens 200000 --trust-remote-code \
        --device cpu --load-format dummy \
        --max-running-requests 4 --mamba-track-interval 1024 \
        --random-seed 42 > /tmp/opencode/p2c3_${logtag}_srv.log 2>&1 &
    local SRV=$!

    for i in $(seq 1 36); do
        sleep 10
        grep -q "HiSim hicache attached" /tmp/opencode/p2c3_${logtag}_srv.log 2>/dev/null && {
            sleep 15
            curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1 && break
        }
    done

    if ! curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] [$logtag] NOT READY (NPU crash?)" >> $LOGFILE
        pkill -f "launch_server" 2>/dev/null || true; sleep 5
        return 1
    fi

    echo "[$(date +%H:%M:%S)] [$logtag] Running bench..." >> $LOGFILE
    python3 -u -m hisim.simulation.bench_serving \
        --warmup-requests 0 --model /home/l00886679/models/Kimi-K3-w4a8 \
        --bench-mode simulation --dataset-name hisim-collection \
        --dataset-path "$trace" --num-prompts 56 \
        --host 127.0.0.1 --port 30050 > /tmp/opencode/p2c3_${logtag}_bench.log 2>&1 || true

    pkill -f "launch_server" 2>/dev/null || true; sleep 8

    if [ -f "$outdir/metrics.json" ]; then
        local result=$(python3 -c "
import json
m = json.load(open('$outdir/metrics.json'))
print(f\"COW={m.get('kda_cow_exec_count',-1)} replay={m.get('kda_historical_replay_tokens',-1)} l2_backup={m.get('kda_l2_backup_count',0)} l2_hit={m.get('kda_l2_load_back_count',0)} l3_restore={m.get('kda_l3_load_back_count',0)} l2_pol={m.get('kda_l2_policy','?')} l2_cap={m.get('kda_l2_capacity',-1)}\")
" 2>/dev/null)
        echo "[$(date +%H:%M:%S)] [$logtag] RESULT: $result" >> $LOGFILE
    else
        echo "[$(date +%H:%M:%S)] [$logtag] NO METRICS" >> $LOGFILE
    fi
}

> $LOGFILE
echo "[$(date +%H:%M:%S)] Phase 2C.3 L2 Placement experiments starting..." >> $LOGFILE

# B2-Fork: L2=32 (moderate) — 3 policies
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "native"     32 "b2f_l2_32_native"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "hotness"    32 "b2f_l2_32_hotness"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "cost_aware" 32 "b2f_l2_32_ca" "0.05"

# B2-Fork: L2=3 (tiny) — compare native vs cost_aware
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "native"     3  "b2f_l2_3_native"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "cost_aware" 3  "b2f_l2_3_ca" "0.05"

# B2-Fork: L2=128 (large) — compare native vs cost_aware
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "native"     128 "b2f_l2_128_native"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "cost_aware" 128 "b2f_l2_128_ca" "0.05"

# Mixed: L2=32 — 3 policies
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "native"     32 "mixed_l2_32_native"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "hotness"    32 "mixed_l2_32_hotness"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "cost_aware" 32 "mixed_l2_32_ca" "0.05"

echo "[$(date +%H:%M:%S)] ALL DONE" >> $LOGFILE
