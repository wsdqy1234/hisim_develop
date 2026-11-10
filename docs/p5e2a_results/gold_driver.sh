#!/usr/bin/env bash
# Phase 5E.2a golden experiment driver
# - Unique Case ID + config hash (sha256 of effective env + win cfg + args + workload)
# - HISIM_SHARED_DRAM=1 for ALL scenarios (fixes the p3e A-baseline omission)
# - Deterministic reuse ordering (workload staggered timestamps)
# - Archives the full /tmp/hisim/output per case
run_gold() {
  local CASE_ID=$1 PORT=$2
  shift 2
  # remaining args: KEY=VALUE pairs
  local -A KV=()
  for kv in "$@"; do KV["${kv%%=*}"]="${kv#*=}"; done
  source /tmp/qwen3_npu_env.sh
  cd /home/d00954154/HiSim_Qwen3

  local WL=${KV[wl]:-p16k} SCEN=${KV[scen]:-A} IFACE=${KV[iface]:-none}
  local AMODE=${KV[amode]:-seq} MEDIA=${KV[media]:-200} WWRITE=${KV[wwrite]:-10}
  local MAXT_OVR=${KV[maxt]:-0}

  # --- workload geometry (Prefix Capacity Rule, co-batch margin) ---
  local MAXT SCRIPT SARGS PLEN=16384 BLOCK=2048
  SCRIPT=${KV[script]:-}
  case "$WL" in
    n2)    MAXT=1024; SCRIPT="/tmp/p3d_concurrent_unique.py"; SARGS="340 2"; PLEN=340; BLOCK=43;;
    n4)    MAXT=1700; SCRIPT="/tmp/p3d_concurrent_unique.py"; SARGS="340 4"; PLEN=340; BLOCK=43;;
    agent) MAXT=1024; SCRIPT="/tmp/p5d_agent_test.py"; SARGS="agent"; PLEN=340; BLOCK=43;;
    p4k)   MAXT=8300; SCRIPT="/tmp/p5d_prefix_test.py"; SARGS="4096 3 2 8300"; PLEN=4096; BLOCK=512;;
    p16k)  MAXT=33100; SCRIPT="/tmp/p5d_prefix_test.py"; SARGS="16384 3 2 33100"; PLEN=16384; BLOCK=2048;;
  esac
  if [ "$MAXT_OVR" != "0" ]; then MAXT=$MAXT_OVR; fi
  # serial-batch variant: MAXT just below 2*(plen+9) forces separate reuse batches
  local HOSTT=$((MAXT * 2))
  local RATIO=2.0

  # --- env (uniform: SHARED_DRAM for ALL scenarios incl. A) ---
  export HISIM_SHARED_DRAM=1
  export HISIM_HBF_ENABLE=0
  export HISIM_L3_WRITE_MODE=barrier
  unset HISIM_HBF_MEDIA_MODEL HISIM_HBF_WIN_CONFIG HISIM_HBF_LOGICAL_SIZE HISIM_HBF_RATIO
  local WCFG=""
  if [ "$SCEN" != "A" ]; then
    RATIO=0.6
    local HBF=$((HOSTT - MAXT * 6 / 10))
    if [ "$SCEN" = "HBFO" ]; then RATIO=0.05; HBF=$((HOSTT - MAXT / 20)); fi
    local SEQ_R=$MEDIA RND_R=100 W_BW=$WWRITE
    if [ "$AMODE" = "rand" ]; then SEQ_R=200; RND_R=$MEDIA; fi
    local AM="sequential"; [ "$AMODE" = "rand" ] && AM="random"
    local IC="null"
    case "$IFACE" in
      s112)  IC="112000000000.0";;
      s896)  IC="896000000000.0";;
      s1280) IC="1280000000000.0";;
      s1600) IC="1600000000000.0";;
    esac
    WCFG=/tmp/gold_${CASE_ID}.json
    cat > $WCFG << EOF
{
  "placement": "striped", "stripe_block_tokens": $BLOCK, "layer": "envelope",
  "default_access_mode": "$AM",
  "package_shared_interface_bw": $IC,
  "win": {"seq_read_bw": $((SEQ_R * 1000000000)), "rand_read_bw": $((RND_R * 1000000000)), "write_bw": $((W_BW * 1000000000)), "read_latency_s": 5e-06, "write_latency_s": 5e-06, "capacity_bytes": 100000000000}
}
EOF
    export HISIM_HBF_ENABLE=1 HISIM_HBF_LOGICAL_SIZE=$HBF
    export HISIM_HBF_RATIO=$([ "$SCEN" = "HBFO" ] && echo 1.0 || echo 0.7)
    export HISIM_HBF_TIMING_MODEL=channel HISIM_HBF_MEDIA_MODEL=win8
    export HISIM_HBF_WIN_CONFIG=$WCFG
  fi
  export HISIM_CONFIG_PATH=test/assets/mock/config.json
  export HISIM_HBF_IO_TRACE=1

  # --- config hash ---
  local HASH
  HASH=$(python3 - << PYEOF
import hashlib, json, os
eff = {
  "case": "$CASE_ID", "wl": "$WL", "scen": "$SCEN", "iface": "$IFACE",
  "amode": "$AMODE", "media_gb": "$MEDIA", "wwrite_gb": "$WWRITE",
  "maxt": $MAXT, "ratio": $RATIO, "plen": $PLEN, "block": $BLOCK,
  "shared_dram": 1, "hbf_logical": ${HBF:-0},
  "win_cfg": json.load(open("$WCFG")) if "$WCFG" and os.path.exists("$WCFG") else None,
  "model": "Qwen3-0.6B", "sglang": "0.5.19",
}
print(hashlib.sha256(json.dumps(eff, sort_keys=True).encode()).hexdigest()[:16])
PYEOF
)
  local LOG=/tmp/gold_${CASE_ID}.log
  > $LOG
  nohup python3 -m hisim.simulation.sglang.launch_server \
    --model-path $BENCHMARK_TEST_MODEL_PATH --device cpu \
    --max-total-tokens $MAXT \
    --enable-hierarchical-cache --hicache-storage-backend sim \
    --hicache-storage-backend-extra-config '{"prefetch_threshold": 1}' \
    --hicache-ratio $RATIO --allow-auto-truncate --port $PORT >> $LOG 2>&1 &
  local PID=$!
  local i
  for i in $(seq 1 400); do grep -q "ready to roll" $LOG 2>/dev/null && break; sleep 1; done
  if ! grep -q "ready to roll" $LOG; then echo "[$CASE_ID] SERVER FAIL" >> /tmp/gold_matrix.out; kill $PID 2>/dev/null; return 1; fi
  rm -rf /tmp/hisim/output/*
  curl -s -X POST http://127.0.0.1:${PORT}/start_profile > /dev/null 2>&1
  sleep 0.3
  timeout 400 python3 $SCRIPT $PORT $CASE_ID $SARGS > /tmp/gold_wl_${CASE_ID}.log 2>&1
  curl -s -X POST http://127.0.0.1:${PORT}/stop_profile > /dev/null 2>&1
  sleep 4
  # archive: parse + manifest + full output dir
  python3 /tmp/p5d_parse_run.py "$CASE_ID" "$LOG" > /tmp/gold_parse_${CASE_ID}.log 2>&1
  mkdir -p /tmp/gold_cases/$CASE_ID
  cp -r /tmp/hisim/output/* /tmp/gold_cases/$CASE_ID/ 2>/dev/null
  cp $LOG /tmp/gold_wl_${CASE_ID}.log /tmp/gold_cases/$CASE_ID/ 2>/dev/null
  cp /tmp/p5d_runs/${CASE_ID}.json /tmp/gold_cases/$CASE_ID/parse.json 2>/dev/null
  [ -f "$WCFG" ] && cp $WCFG /tmp/gold_cases/$CASE_ID/win_config.json 2>/dev/null
  python3 - "$CASE_ID" "$HASH" "$WL" "$SCEN" "$IFACE" "$AMODE" "$MEDIA" "$WWRITE" "$MAXT" "$RATIO" << PYEOF >> /tmp/gold_cases/$CASE_ID/manifest.json.tmp 2>&1
import json, sys, hashlib, os
case_id, h, wl, scen, iface, amode, media, wwrite, maxt, ratio = sys.argv[1:11]
m = {"case_id": case_id, "config_hash": h, "wl": wl, "scen": scen,
     "iface_hypothesis": iface, "access_mode": amode, "media_read_gb": int(media),
     "media_write_gb": int(wwrite), "max_total": int(maxt), "hicache_ratio": float(ratio),
     "shared_dram": True, "hbf_layer": "A-envelope", "notes":
     "iface values are HYPOTHESES (UB unconfirmed); media 160/80 = assumed 0.8 efficiency"}
json.dump(m, open("/tmp/gold_cases/%s/manifest.json" % case_id, "w"), indent=1)
PYEOF
  rm -f /tmp/gold_cases/$CASE_ID/manifest.json.tmp
  kill $PID 2>/dev/null; sleep 2
  ps aux | grep -E "launch_server|multiprocessing" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
  echo "[$CASE_ID] hash=$HASH done" >> /tmp/gold_matrix.out
}
