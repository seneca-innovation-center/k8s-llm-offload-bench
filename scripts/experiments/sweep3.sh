#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Single-pod sequential streaming: one pod, all 36 layers streamed in order through S prefetch slots
# (S=1: load, compute, load, compute; S>1: up to min(S,4) loads in flight while computing). No pods to hop between.
cd "$HERE/../../src/engine"
OUT=/exp/results/results3.jsonl
run() { local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $*"
}
echo "START $(date)"
run baseline 1 1 resident 3 16 128,512,2048
for S in 1 2 3 4 6 9; do run seq1pod 36 1 cyclic 3 16 128,512,2048 SLOTS=$S; done
echo "DONE $(date)"
