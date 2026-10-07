#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Full sweep: baseline, resident controls, cyclic (G,K) grid. Appends to results/results.jsonl
cd "$HERE/../../src/engine"
run() { # G K MODE REPS
  python3 ../k8s/deploy.py $1 $2 $3 || { echo "DEPLOY FAIL $*"; return; }
  kubectl exec coordinator -- python /exp/bench.py --tag "{\"G\":$1,\"K\":$2,\"mode\":\"$3\"}" \
     --reps $4 --max-new 16 --prompts 128,512,2048 || echo "BENCH FAIL $*"
}
echo "START $(date)"
run 1 1 resident 5
for K in 2 3 4 6 9 12 18; do run 36 $K cyclic 3; done
for K in 4 12 36; do run 36 $K resident 3; done
for K in 2 3 4 6; do run 12 $K cyclic 3; done
for K in 2 6 9; do run 18 $K cyclic 3; done
run 9 3 cyclic 3
run 6 2 cyclic 3; run 6 3 cyclic 3
run 4 2 cyclic 3
run 1 1 resident 5
echo "DONE $(date)"
