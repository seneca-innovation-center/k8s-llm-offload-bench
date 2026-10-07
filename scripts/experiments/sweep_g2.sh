#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# gpt-oss-20b, experiment 2: single-pod sequential, partial residency, mmap + limit (deploy, then drop cache), batching.
cd "$HERE/../../src/engine"
OUT=/exp/results_gptoss/results2.jsonl
W=WEIGHTS=/exp/weights_gptoss
run() { # approach G K MODE reps max_new prompts extra...
  local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE $W "$@" || { echo "DEPLOY FAIL $ap $G $K $MODE $*"; return; }
  [ "$ap" = mmap ] && python3 dropcache.py $PWD/weights_gptoss
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 7200 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $G $K $MODE $*"
}
echo "START $(date)"
run baseline    1 1 resident 5 16 128,512,2048
for S in 1 2 3 4 6; do run seq1pod 24 1 cyclic 3 16 128,512,2048 SLOTS=$S; done
run partial    24 1 cyclic   3 16 128,512,2048 RESIDENT=0-21 SLOTS=2
run partial    24 1 cyclic   3 16 128,512,2048 RESIDENT=0-19 SLOTS=2
run partial    24 1 cyclic   3 16 128,512,2048 RESIDENT=0-17 SLOTS=3
run partial    24 1 cyclic   3 16 128,512,2048 RESIDENT=0-15 SLOTS=4
run partial    24 1 cyclic   3 16 128,512,2048 RESIDENT=0-11 SLOTS=4
for M in 48Gi 32Gi 24Gi 16Gi 8Gi; do run mmap 1 1 resident 3 16 128,512,2048 LOAD_MODE=mmap MEM_LIMIT=$M; done
for B in 16 64 256; do
  run batch     1 1 resident 2 16 128 BATCH=$B MAXLEN=192
  run batch    24 4 cyclic   2 16 128 BATCH=$B MAXLEN=192
done
run batch      24 2 cyclic   2 16 128 BATCH=64 MAXLEN=192
run batch      24 2 cyclic   2 16 128 BATCH=256 MAXLEN=192
run batch       1 1 resident 1 16 128 BATCH=512 MAXLEN=192
run batch      24 4 cyclic   1 16 128 BATCH=512 MAXLEN=192
run baseline    1 1 resident 5 16 128,512,2048
echo "DONE $(date)"
