#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# gpt-oss-20b, experiment 1: baseline, cyclic G×K grid, resident controls, baseline again.
cd "$HERE/../../src/engine"
OUT=/exp/results_gptoss/results.jsonl
W=WEIGHTS=/exp/weights_gptoss
run() { # G K MODE REPS
  python3 ../k8s/deploy.py $1 $2 $3 $W || { echo "DEPLOY FAIL $*"; return; }
  timeout 7200 kubectl exec coordinator -- python /exp/bench.py --out $OUT --tag "{\"G\":$1,\"K\":$2,\"mode\":\"$3\"}" \
     --reps $4 --max-new 16 --prompts 128,512,2048 || echo "BENCH FAIL $*"
}
echo "START $(date)"
run 1 1 resident 5
for K in 2 3 4 6 8 12; do run 24 $K cyclic 3; done
for K in 4 12 24; do run 24 $K resident 3; done
for K in 2 3 4 6; do run 12 $K cyclic 3; done
run 8 2 cyclic 3; run 8 4 cyclic 3
run 6 2 cyclic 3; run 6 3 cyclic 3
run 4 2 cyclic 3
run 1 1 resident 5
echo "DONE $(date)"
