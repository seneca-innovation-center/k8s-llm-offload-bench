#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Reruns: mmap with the page cache dropped before each config; speculative decoding with a non-repeating prompt.
cd "$HERE/../../src/engine"
OUT=/exp/results/results2.jsonl
run() { # approach G K MODE reps max_new prompts extra...
  local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 dropcache.py >/dev/null
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $ap $G $K $MODE $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $G $K $MODE $*"
}
echo "START $(date)"
for M in 6Gi 4Gi 3Gi 2Gi 1.5Gi; do run mmap 1 1 resident 3 16 128,512,2048 LOAD_MODE=mmap MEM_LIMIT=$M; done
P=PROMPT_FILE=prompt_long.txt
run baseline    1 1 resident 3 32 512 $P
run cyclic     36 2 cyclic   3 16 512 $P
run cyclic     36 4 cyclic   3 16 512 $P
run spec        1 1 resident 3 32 512 $P DRAFT=/exp/draft GAMMA=5
run spec       36 2 cyclic   3 32 512 $P DRAFT=/exp/draft GAMMA=5
run spec       36 4 cyclic   3 32 512 $P DRAFT=/exp/draft GAMMA=5
run spec       36 4 cyclic   3 32 512 $P DRAFT=/exp/draft GAMMA=8
echo "DONE $(date)"
