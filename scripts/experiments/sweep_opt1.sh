#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Latency optimisation, round 1 (3B, non-repeating 512-token prompt): single-pod streamer + speculative decoding,
# direct I/O loads, and the existing points re-measured for a same-day comparison. Appends to results/results4.jsonl.
cd "$HERE/../../src/engine"
OUT=/exp/results/results4.jsonl
P=PROMPT_FILE=prompt_long.txt
run() { local ap=$1 G=$2 K=$3 MODE=$4 REPS=$5 NEW=$6 PR=$7; shift 7
  python3 ../k8s/deploy.py $G $K $MODE "$@" || { echo "DEPLOY FAIL $ap $*"; return; }
  local extra=""; for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"${kv#*=}\""; done
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps $REPS --max-new $NEW --prompts $PR \
     --tag "{\"approach\":\"$ap\",\"G\":$G,\"K\":$K,\"mode\":\"$MODE\"$extra}" || echo "BENCH FAIL $ap $*"
}
echo "START $(date)"
run baseline  1 1 resident 3 32 512 $P
run seq1pod  36 1 cyclic   3 16 512 $P SLOTS=4
run seq1pod  36 1 cyclic   3 16 512 $P SLOTS=4 LOAD_MODE=direct
run cyclic   36 4 cyclic   3 16 512 $P LOAD_MODE=direct
run spec     36 2 cyclic   3 32 512 $P DRAFT=/exp/draft GAMMA=5
run spec     36 1 cyclic   3 32 512 $P SLOTS=4 DRAFT=/exp/draft GAMMA=5
run spec     36 1 cyclic   3 32 512 $P SLOTS=6 DRAFT=/exp/draft GAMMA=5
run spec     36 1 cyclic   3 32 512 $P SLOTS=4 LOAD_MODE=direct DRAFT=/exp/draft GAMMA=5
run spec     36 4 cyclic   3 32 512 $P LOAD_MODE=direct DRAFT=/exp/draft GAMMA=5
run partial  36 1 cyclic   3 16 512 $P RESIDENT=0-32 SLOTS=3
run partial  36 1 cyclic   3 16 512 $P RESIDENT=0-32 SLOTS=3 LOAD_MODE=direct
echo "DONE $(date)"
