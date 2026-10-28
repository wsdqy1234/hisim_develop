#!/bin/bash
# Phase 2D.2: Post-fix minimal fair comparison
# 2 traces (K2M, M2K) × 5 configs = 10 experiments
set -e

PORT=30060
MODEL=/home/l00886679/models/Kimi-K3-w4a8
CONFIG=/tmp/opencode/config_k3_0519.json
RESULTS_DIR=/tmp/opencode/p2d2_results
mkdir -p $RESULTS_DIR

# Common env (from run_p2c3_l2.sh)
export SGLANG_USE_CPU_ENGINE=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONHASHSEED=42
export HISIM_SIMULATION_MODE=OFFLINE HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
export SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_DISABLE_PROACTIVE_EVICT=1 HISIM_KDA_OBS=1
export HISIM_DISABLE_L3=0
export HISIM_MAMBA_POOL_SIZE=16
export HISIM_KDA_POLICY=native
export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd

run_experiment() {
    local name=$1
    local trace=$2
    local extra_env=$3
    local logfile=$RESULTS_DIR/${name}_srv.log
    local benchlog=$RESULTS_DIR/${name}_bench.log

    echo "========================================"
    echo "[$(date +%H:%M:%S)] Running: $name (trace=$trace)"
    echo "========================================"

    # Kill existing server
    pkill -f "launch_server" 2>/dev/null || true
    sleep 3
    while ss -tlnp 2>/dev/null | grep -q $PORT; do sleep 1; done

    # Clear pycache and hicache storage
    cd /home/d00954154/hisim
    find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf /tmp/hisim/hicache 2>/dev/null || true

    # Set extra env vars for this config
    eval "$extra_env"

    # Start server
    setsid python3 -u -m hisim.simulation.sglang.launch_server \
        --model-path $MODEL \
        --sim-config-path $CONFIG \
        --skip-server-warmup --port $PORT \
        --enable-hierarchical-cache --hicache-size 80 \
        --max-total-tokens 200000 --trust-remote-code \
        --device cpu --load-format dummy \
        --max-running-requests 4 --mamba-track-interval 1024 \
        --random-seed 42 > $logfile 2>&1 &
    local srv_pid=$!

    # Wait for server ready
    echo "  Waiting for server (pid=$srv_pid)..."
    local ready=0
    for i in $(seq 1 36); do
        sleep 10
        grep -q "HiSim hicache attached" $logfile 2>/dev/null && {
            sleep 15
            curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
        }
    done

    if [ $ready -eq 0 ]; then
        echo "  ERROR: Server failed to start"
        kill -TERM $srv_pid 2>/dev/null || true
        sleep 2
        kill -9 $srv_pid 2>/dev/null || true
        return 1
    fi
    echo "  Server ready, running benchmark..."

    # Run benchmark
    python3 -u -m hisim.simulation.bench_serving \
        --warmup-requests 0 --model $MODEL \
        --bench-mode simulation --dataset-name hisim-collection \
        --dataset-path "$trace" --num-prompts 56 \
        --host 127.0.0.1 --port $PORT > $benchlog 2>&1 || true

    # Kill server gracefully
    kill -TERM $srv_pid 2>/dev/null || true
    sleep 3
    kill -9 $srv_pid 2>/dev/null || true

    echo "  [$(date +%H:%M:%S)] Done: $name"
    echo ""
}

# Extract metrics from server log
extract_metrics() {
    local name=$1
    local logfile=$RESULTS_DIR/${name}_srv.log

    echo "=== $name ==="

    # Group size and capacity check
    grep "\[P2D2\] Capacity check\|Legacy HostPoolGroup normalized\|L2 unified DRAM pool bound" $logfile 2>/dev/null | tail -1

    # MLA evictor registration
    grep "\[P2D2\] MLA evictor" $logfile 2>/dev/null | tail -1

    # ILB and L3 check counts
    echo -n "  ILB: "; grep -c "\[ILB\]" $logfile 2>/dev/null || echo 0
    echo -n "  LOAD-L3CHECK: "; grep -c "LOAD-L3CHECK" $logfile 2>/dev/null || echo 0
    echo -n "  LOAD-L2CHECK: "; grep -c "LOAD-L2CHECK" $logfile 2>/dev/null || echo 0

    # Failure boundary
    echo -n "  P2D2-L2-LOAD-FAIL: "; grep -c "P2D2-L2-LOAD-FAIL" $logfile 2>/dev/null || echo 0
    echo -n "  P2D2-L3-STAGE-FAIL: "; grep -c "P2D2-L3-STAGE-FAIL" $logfile 2>/dev/null || echo 0
    echo -n "  D1.2-L3-REJECT: "; grep -c "D1.2-L3-REJECT" $logfile 2>/dev/null || echo 0

    # Extract key metrics using python
    python3 -c "
import re, sys
log = open('$logfile').read()
def last_int(pat):
    m = re.findall(pat, log)
    return int(m[-1]) if m else 0
def last_float(pat):
    m = re.findall(pat, log)
    return float(m[-1]) if m else 0.0

kda_l3_wr = last_int(r'\"kda_l3_mamba_writes\":\s*(\d+)')
kda_l3_hits = last_int(r'\"kda_l3_mamba_hits\":\s*(\d+)')
cross_mla = last_int(r'\"l2_cross_evict_mla_tokens\":\s*(\d+)')
cross_kda = last_int(r'\"l2_cross_evict_kda_slots\":\s*(\d+)')
mla_for_kda = last_int(r'\"l2_evict_mla_for_kda_tokens\":\s*(\d+)')
kda_for_mla = last_int(r'\"l2_evict_kda_for_mla_slots\":\s*(\d+)')
kda_for_kda = last_int(r'\"l2_evict_kda_for_kda_slots\":\s*(\d+)')
mla_for_mla = last_int(r'\"l2_evict_mla_for_mla_tokens\":\s*(\d+)')
slack = last_int(r'\"l2_mla_pd_slack_bytes\":\s*(\d+)')
partial_ext = last_int(r'\"l2_mla_pd_partial_extents\":\s*(\d+)')
live_bytes = last_int(r'\"l2_mla_pd_live_token_bytes\":\s*(\d+)')
alloc_bytes = last_int(r'\"l2_mla_pd_allocated_bytes\":\s*(\d+)')
kda_slots = last_int(r'\"l2_kda_cached_slots\":\s*(\d+)')
mla_tokens = last_int(r'\"l2_mla_live_tokens\":\s*(\d+)')
used = last_int(r'\"l2_used_bytes\":\s*(\d+)')
l2_fail = last_int(r'\"kda_l2_load_fail\":\s*(\d+)')
l3_fail = last_int(r'\"kda_l3_stage_fail\":\s*(\d+)')

print(f'  KDA_L3_Wr={kda_l3_wr} KDA_L3_Hits={kda_l3_hits}')
print(f'  KDA_slots={kda_slots} MLA_tokens={mla_tokens} Used={used/1e6:.1f}MB')
print(f'  cross_mla={cross_mla} cross_kda={cross_kda}')
print(f'  DIR: mla_for_kda={mla_for_kda} kda_for_mla={kda_for_mla} kda_for_kda={kda_for_kda} mla_for_mla={mla_for_mla}')
print(f'  FAIL: l2_load={l2_fail} l3_stage={l3_fail}')
if alloc_bytes > 0:
    print(f'  MLA_PD: slack={slack/1e6:.1f}MB partial_ext={partial_ext} live={live_bytes/1e6:.1f}MB alloc={alloc_bytes/1e6:.1f}MB')
" 2>/dev/null || echo "  (metrics extraction failed)"

    echo ""
}

# Experiment configurations
# Format: "name|extra_env_export"
CONFIGS=(
    "leg|export HISIM_L2_KV_SIZE_GB=0.383 HISIM_L2_MAMBA_SIZE=8"
    "2p8|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=8"
    "2p16|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=16"
    "1p_sf|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=self_first"
    "1p_of|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=other_first"
)

# Run all experiments
for trace_file in /tmp/opencode/k3_p2d_kda_to_mla.jsonl /tmp/opencode/k3_p2d_mla_to_kda.jsonl; do
    if [[ "$trace_file" == *kda_to_mla* ]]; then
        tname="K2M"
    else
        tname="M2K"
    fi
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r cname cenv <<< "$cfg"
        exp_name="p2d2_${cname}_${tname,,}"
        # Unset previous config env vars
        unset HISIM_L2_KV_SIZE_GB HISIM_L2_TOTAL_GB HISIM_L2_ONE_POOL HISIM_L2_BORROW_POLICY HISIM_L2_MAMBA_SIZE
        run_experiment "$exp_name" "$trace_file" "$cenv"
    done
done

# Extract metrics
echo ""
echo "============================================"
echo "METRICS SUMMARY"
echo "============================================"
for trace_name in K2M M2K; do
    echo "--- ${trace_name} ---"
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r cname cenv <<< "$cfg"
        exp_name="p2d2_${cname}_${trace_name,,}"
        extract_metrics "$exp_name"
    done
done

echo "Done. Results in $RESULTS_DIR/"
