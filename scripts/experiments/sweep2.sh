#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Alternative approaches. Appends to results/results2.jsonl (tags carry an "approach" field).
cd "$HERE/../../src/engine"
OUT=/exp/results/results2.jsonl
run() { # approach G K MODE reps max_new prompts extra...
  local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $ap $G $K $MODE $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $G $K $MODE $*"
}
echo "START $(date)"
run baseline    1 1 resident 5 16 128,512,2048
run partial    36 1 cyclic   3 16 128,512,2048 RESIDENT=0-32 SLOTS=3
run partial    36 1 cyclic   3 16 128,512,2048 RESIDENT=0-29 SLOTS=3
run partial    36 1 cyclic   3 16 128,512,2048 RESIDENT=0-26 SLOTS=3
run partial    36 1 cyclic   3 16 128,512,2048 RESIDENT=0-23 SLOTS=4
run partial    36 1 cyclic   3 16 128,512,2048 RESIDENT=0-17 SLOTS=4
for M in 6Gi 4Gi 3Gi 2Gi 1.5Gi; do run mmap 1 1 resident 3 16 128,512,2048 LOAD_MODE=mmap MEM_LIMIT=$M; done
run spec        1 1 resident 3 32 128,512,2048 DRAFT=/exp/draft GAMMA=5
run spec       36 2 cyclic   3 32 128,512,2048 DRAFT=/exp/draft GAMMA=5
run spec       36 4 cyclic   3 32 128,512,2048 DRAFT=/exp/draft GAMMA=5
run spec       36 4 cyclic   3 32 128,512,2048 DRAFT=/exp/draft GAMMA=8
for B in 16 64 256; do
  run batch     1 1 resident 2 16 128 BATCH=$B MAXLEN=192
  run batch    36 4 cyclic   2 16 128 BATCH=$B MAXLEN=192
done
run batch      36 2 cyclic   2 16 128 BATCH=64 MAXLEN=192
run batch      36 2 cyclic   2 16 128 BATCH=256 MAXLEN=192
run batch       1 1 resident 1 16 128 BATCH=512 MAXLEN=192
run batch      36 4 cyclic   1 16 128 BATCH=512 MAXLEN=192
run baseline    1 1 resident 5 16 128,512,2048
echo "DONE $(date)"
