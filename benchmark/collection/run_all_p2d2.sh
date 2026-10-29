#!/bin/bash
# Phase 2D.2: Run all 10 experiments sequentially
# 2 traces × 5 configs
cd /home/d00954154/hisim

TRACES=(
    "/tmp/opencode/k3_p2d_kda_to_mla.jsonl|K2M"
    "/tmp/opencode/k3_p2d_mla_to_kda.jsonl|M2K"
)

CONFIGS=(
    "leg|export HISIM_L2_KV_SIZE_GB=0.383 HISIM_L2_MAMBA_SIZE=8"
    "2p8|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=8"
    "2p16|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_MAMBA_SIZE=16"
    "1p_sf|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=self_first"
    "1p_of|export HISIM_L2_TOTAL_GB=0.5 HISIM_L2_ONE_POOL=1 HISIM_L2_BORROW_POLICY=other_first"
)

for trace_entry in "${TRACES[@]}"; do
    IFS='|' read -r trace_file tname <<< "$trace_entry"
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r cname cenv <<< "$cfg"
        exp_name="p2d2_${cname}_${tname,,}"
        echo ">>> Running $exp_name"
        bash benchmark/collection/run_single.sh "$exp_name" "$trace_file" "$cenv"
        echo ">>> Completed $exp_name"
        echo ""
    done
done

echo "=== ALL EXPERIMENTS DONE ==="
