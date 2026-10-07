#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Latency optimisation, round 2: tree speculation on the single-pod streamer (3B, budget <= 3.5 GB).
# Prose prompt (512 tokens) + three chat prompts (natural length, n_prompt=0). Appends to results/results5.jsonl.
cd "$HERE/../../src/engine"
OUT=/exp/results/results5.jsonl
run() { local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $ap $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $*"
}
echo "START $(date)"
D=DRAFT=/exp/draft
for PF in "PROMPT_FILE=prompt_long.txt 512" "PROMPT_FILE=prompts/chat_explain.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_summary.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_code.txt 0 CHAT=1"; do
  set -- $PF; P=$1; N=$2; C=${3:-}
  run baseline 1 1 resident 3 48 $N $P $C
  run spec    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D GAMMA=5
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=4,6,6,4,2
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=8,16,16,12,8,4
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=16,32,32,24,16,8
  run tree    36 1 cyclic   3 48 $N $P $C SLOTS=4 $D TREE=16,32,48,48,32,16,8
done
echo "DONE $(date)"
