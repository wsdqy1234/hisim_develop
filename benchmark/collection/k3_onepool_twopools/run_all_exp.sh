#!/usr/bin/env bash
# Sequentially run the P5.1/P5.2/P5.3 Two Pools cells on the local 0.5.19 target.
# Configs mirror docs/k3_onepool_vs_twopools_experiment_report.md.
#   P5.1 KDA-heavy : KDA in {40,60,80,100,120}, MLA=200000, C=1024, chunk=0
#   P5.2 MLA-heavy : KDA in {40,60,80,100}, MLA=403520-2544*KDA, C=8192, chunk=8192
#   P5.3 dynamic   : KDA=80 (MLA=200000) + KDA=40 (MLA=301760), C=1024, chunk=32768
# Env: MODE=p51|p52|p53|all (default all); WORKDIR as in run_exp.sh.
# Each cell reboots the server (~2.5-3 min boot on CPU) before benching.
set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR="${WORKDIR:-/tmp/opencode}"
RES="$WORKDIR/exp_results"
mkdir -p "$RES"

run_cell() {
  local mode=$1 kda=$2 mla=$3 c=$4 chunk=$5 trace=$6 n=$7 label=$8
  echo "######## CELL $label (mode=$mode kda=$kda mla=$mla C=$c chunk=$chunk) $(date +%H:%M:%S) ########"
  bash "$SCRIPT_DIR/run_exp.sh" "$mode" "$kda" "$mla" "$c" "$chunk" "$trace" "$n" "$label" 30011 || {
    echo "CELL $label FAILED"; return 1; }
  python3 - "$label" "$WORKDIR" <<'PY'
import json, os, sys
label, workdir = sys.argv[1], sys.argv[2]
p = f"/tmp/hisim/{label}/metrics.json"
out = os.path.join(workdir, "exp_results", f"{label}.json")
if os.path.exists(p):
    json.dump(json.load(open(p)), open(out, "w"), indent=2)
    print(f"saved {out}")
else:
    print(f"{label}: no metrics.json")
PY
}

MODE="${MODE:-all}"
if [ "$MODE" = "all" ] || [ "$MODE" = "p51" ]; then
  for k in 40 60 80 100 120; do
    run_cell twopools "$k" 200000 1024 0 "$WORKDIR/k3_p51_kda_heavy.jsonl" 102 "p51_tp${k}"
  done
  # Attribution cell: docker's Two Pools uses a SHARED budget (MLA=403520-2544*KDA);
  # at KDA=120 that gives MLA=98240 < demand 98304 -> cascade -> 1 COW miss.
  run_cell twopools 120 98240 1024 0 "$WORKDIR/k3_p51_kda_heavy.jsonl" 102 p51_tp120_b98k
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "p52" ]; then
  for k in 40 60 80 100; do
    mla=$((403520 - 2544 * k))
    run_cell twopools "$k" "$mla" 8192 8192 "$WORKDIR/k3_p52_mla_heavy.jsonl" 11 "p52_tp${k}"
  done
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "p53" ]; then
  run_cell twopools 80 200000 1024 32768 "$WORKDIR/k3_p53_dynamic.jsonl" 119 p53_tp80
  run_cell twopools 40 301760 1024 32768 "$WORKDIR/k3_p53_dynamic.jsonl" 119 p53_tp40
fi

# One Pool cells (P5.1/5.2/5.3) — separate MODE so the Two Pools sweep is runnable alone
if [ "$MODE" = "all" ] || [ "$MODE" = "onepool" ]; then
  run_cell onepool 80 200000 1024 0   "$WORKDIR/k3_p51_kda_heavy.jsonl" 102 p51_op80
  run_cell onepool 80 200000 8192 8192 "$WORKDIR/k3_p52_mla_heavy.jsonl" 11 p52_op80
  run_cell onepool 80 200000 1024 32768 "$WORKDIR/k3_p53_dynamic.jsonl" 119 p53_op80
fi

echo "######## ALL CELLS DONE $(date +%H:%M:%S) ########"
