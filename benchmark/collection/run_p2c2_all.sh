#!/bin/bash
# Phase 2C.2: Full deterministic experiment suite
# All runs with --random-seed=42 and PYTHONHASHSEED=42

# Note: no set -e — run_one handles errors internally

# Helper function
run_one() {
    local trace=$1 pol=$2 tag=$3 base_prob=${4:-0.0}
    local outdir="/tmp/hisim/p2c2_${tag}_${pol}"
    local logtag="${tag}_${pol}"
    [ "$base_prob" != "0.0" ] && { outdir="${outdir}_bp${base_prob}"; logtag="${logtag}_bp${base_prob}"; }

    # Kill any process on port 30050 (fuser may not be available)
    pkill -f "launch_server.*30050" 2>/dev/null || true
    sleep 5
    rm -rf "$outdir" /tmp/hisim/hicache; mkdir -p "$outdir"

    export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export PYTHONHASHSEED=42
    export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
    export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1
    export HISIM_KDA_OBS=1
    export HISIM_L2_MAMBA_SIZE=0 HISIM_DISABLE_L3=1 HISIM_MAMBA_POOL_SIZE=16
    export HISIM_KDA_POLICY=$pol HISIM_OUTPUT_DIR=$outdir HISIM_HOTNESS_THRESHOLD=0
    export HISIM_COST_AWARE_BASE_PROB=$base_prob
    export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd
    find /home/d00954154/hisim/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    [ "$pol" = "oracle" ] && export HISIM_ORACLE_TRACE=$trace

    echo "[$logtag] Starting..."
    setsid python3 -u -m hisim.simulation.sglang.launch_server \
        --model-path /home/l00886679/models/Kimi-K3-w4a8 \
        --sim-config-path /tmp/opencode/config_k3_0519.json \
        --skip-server-warmup --port 30050 \
        --enable-hierarchical-cache --hicache-size 80 \
        --max-total-tokens 200000 --trust-remote-code \
        --device cpu --load-format dummy \
        --max-running-requests 4 --mamba-track-interval 1024 \
        --random-seed 42 \
        > /tmp/opencode/p2c2_${logtag}_srv.log 2>&1 &
    local SRV=$!

    for i in $(seq 1 36); do
        sleep 10
        grep -q "HiSim hicache attached" /tmp/opencode/p2c2_${logtag}_srv.log 2>/dev/null && {
            sleep 15
            curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1 && break
        }
    done

    if ! curl -s -m 2 "http://127.0.0.1:30050/health" >/dev/null 2>&1; then
        echo "[$logtag] NOT READY"
        pkill -f "launch_server.*30050" 2>/dev/null || true; sleep 5
        return 1
    fi

    echo "[$logtag] Running bench..."
    python3 -u -m hisim.simulation.bench_serving \
        --warmup-requests 0 --model /home/l00886679/models/Kimi-K3-w4a8 \
        --bench-mode simulation --dataset-name hisim-collection \
        --dataset-path "$trace" --num-prompts 56 \
        --host 127.0.0.1 --port 30050 \
        > /tmp/opencode/p2c2_${logtag}_bench.log 2>&1 || true

    pkill -f "launch_server.*30050" 2>/dev/null || true; sleep 8

    echo -n "[$logtag] RESULT: "
    if [ -f "$outdir/metrics.json" ]; then
        python3 /tmp/check_b1_metrics.py "$outdir/metrics.json" 2>/dev/null \
            | grep -E "cow_exec|replay_tokens|double_free" | tr '\n' ' '
        echo ""
    else
        echo "NO METRICS"
    fi
}

# ============================================================
# PRIORITY 1: B2-Fork (the 820K degradation test)
# ============================================================
echo "============================================================"
echo "=== B2-Fork: 6 strategies + Cost-Aware ablation ==="
echo "============================================================"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "native"    "b2_fork"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "neutral"   "b2_fork"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "hotness"   "b2_fork"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "depth"    "b2_fork"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "cost_aware" "b2_fork" "0.0"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "cost_aware" "b2_fork" "0.05"
run_one "/tmp/opencode/k3_p2b_b2_fork.jsonl" "oracle"   "b2_fork"

# ============================================================
# PRIORITY 2: B1-8K (verify Cost-Aware = Hotness for same depth)
# ============================================================
echo "============================================================"
echo "=== B1-8K: 5 strategies ==="
echo "============================================================"
run_one "/tmp/opencode/k3_p2b_b1_8192.jsonl" "native"     "b1_8k"
run_one "/tmp/opencode/k3_p2b_b1_8192.jsonl" "hotness"   "b1_8k"
run_one "/tmp/opencode/k3_p2b_b1_8192.jsonl" "cost_aware"  "b1_8k" "0.0"
run_one "/tmp/opencode/k3_p2b_b1_8192.jsonl" "cost_aware"  "b1_8k" "0.05"
run_one "/tmp/opencode/k3_p2b_b1_8192.jsonl" "oracle"    "b1_8k"

# ============================================================
# PRIORITY 3: Mixed (different depths — marginal cost matters)
# ============================================================
echo "============================================================"
echo "=== Mixed: 5 strategies + Cost-Aware ablation ==="
echo "============================================================"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "native"     "mixed"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "hotness"   "mixed"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "depth"    "mixed"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "cost_aware"  "mixed" "0.0"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "cost_aware"  "mixed" "0.05"
run_one "/tmp/opencode/k3_p2c_mixed.jsonl" "oracle"    "mixed"

# ============================================================
# PRIORITY 4: B1-32K (thrashing analysis)
# ============================================================
echo "============================================================"
echo "=== B1-32K: 4 strategies ==="
echo "============================================================"
run_one "/tmp/opencode/k3_p2b_b1_32768.jsonl" "native"     "b1_32k"
run_one "/tmp/opencode/k3_p2b_b1_32768.jsonl" "hotness"   "b1_32k"
run_one "/tmp/opencode/k3_p2b_b1_32768.jsonl" "cost_aware"  "b1_32k" "0.05"
run_one "/tmp/opencode/k3_p2b_b1_32768.jsonl" "oracle"    "b1_32k"

echo "============================================================"
echo "=== ALL EXPERIMENTS DONE ==="
echo "============================================================"

# Summary
echo ""
echo "=== SUMMARY ==="
for tag in b2_fork b1_8k mixed b1_32k; do
    for f in /tmp/hisim/p2c2_${tag}_*/metrics.json; do
        [ -f "$f" ] || continue
        pol=$(echo "$f" | sed "s|.*/p2c2_${tag}_||;s|/metrics.json||")
        echo "[$tag/$pol] $(python3 /tmp/check_b1_metrics.py "$f" 2>/dev/null | grep -E 'cow_exec|replay_tokens|double_free' | tr '\n' ' ')"
    done
done
