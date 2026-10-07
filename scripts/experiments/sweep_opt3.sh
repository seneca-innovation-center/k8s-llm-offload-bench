#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Round 3: tree speculation with the lean drafter (our layer code, DRAFT_LAYERS), budget <= 3.5 GB.
# Same 4 prompts as round 2. Appends to results/results6.jsonl.
cd "$HERE/../../src/engine"
OUT=/exp/results/results6.jsonl
run() { local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $ap $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $*"
}
echo "START $(date)"
D=DRAFT_LAYERS=/exp/weights_05b
for PF in "PROMPT_FILE=prompt_long.txt 512" "PROMPT_FILE=prompts/chat_explain.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_summary.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_code.txt 0 CHAT=1"; do
  set -- $PF; P=$1; N=$2; C=${3:-}
  run baseline 1 1 resident 3 48 $N $P $C
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=4,6,6,4,2
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=8,16,16,12,8,4
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=3,4,4,3,2,2,1
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=8,8,8,4
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=5 $D TREE=8,16,16,12,8,4
done
echo "DONE $(date)"
