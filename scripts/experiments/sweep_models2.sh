#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Cross-model test of the two best approaches + plain streaming + baseline. (sweep_models2: DRAFT_DIR=none skips the tree run)
#   sweep_models.sh NAME LAYERS DRAFT_DIR LIMIT1 LIMIT2 [LIMIT3]
# NAME: weights in /exp/weights_NAME; results -> /exp/results_models/NAME.jsonl
# Prompts: prose 512 + chat summary + chat code; 32 generated tokens; 3 repetitions.
cd "$HERE/../../src/engine"
NAME=$1; L=$2; DRAFT=$3; shift 3; LIMITS="$@"
W=WEIGHTS=/exp/weights_$NAME
mkdir -p results_models
OUT=/exp/results_models/$NAME.jsonl
run() { local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE $W "$@" || { echo "DEPLOY FAIL $ap $*"; return; }
  [ "$ap" = mmap ] && python3 dropcache.py $PWD/weights_$NAME
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 7200 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\",\"model\":\"$NAME\"$extra}" || echo "BENCH FAIL $ap $*"
}
echo "START $NAME $(date)"
for PF in "PROMPT_FILE=prompt_long.txt 512" "PROMPT_FILE=prompts/chat_summary.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_code.txt 0 CHAT=1"; do
  set -- $PF; P=$1; N=$2; C=${3:-}
  run baseline  1 1 resident 3 32 $N $P $C
  run seq1pod  $L 1 cyclic   3 32 $N $P $C SLOTS=4
  [ "$DRAFT" != none ] && run tree $L 1 cyclic 3 32 $N $P $C SLOTS=4 DRAFT_LAYERS=$DRAFT TREE=8,16,16,12,8,4
  for M in $LIMITS; do run mmap 1 1 resident 3 32 $N $P $C LOAD_MODE=mmap MEM_LIMIT=$M; done
done
echo "DONE $NAME $(date)"
