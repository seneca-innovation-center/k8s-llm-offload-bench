#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# gpt-oss mmap + memory limit: finish the rerun (24/16/8 GiB) that halo's reboot interrupted; deploy, then drop cache.
cd "$HERE/../../src/engine"
OUT=/exp/results_gptoss/results2.jsonl
python3 -c "
import json; L=[l for l in open('results_gptoss/results2.jsonl') if not (json.loads(l).get('LOAD_MODE')=='mmap' and json.loads(l).get('MEM_LIMIT') in ('24Gi','16Gi','8Gi'))]
open('results_gptoss/results2.jsonl','w').writelines(L); print('rows kept', len(L))"
run() { local M=$1
  python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_gptoss LOAD_MODE=mmap MEM_LIMIT=$M || { echo "DEPLOY FAIL $M"; return; }
  python3 dropcache.py $PWD/weights_gptoss || { echo "DROPCACHE FAIL $M"; return; }
  timeout 10800 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps 3 --max-new 16 --prompts 128,512,2048 \
     --tag "{\"approach\":\"mmap\",\"G\":1,\"K\":1,\"mode\":\"resident\",\"LOAD_MODE\":\"mmap\",\"MEM_LIMIT\":\"$M\"}" || echo "BENCH FAIL $M"
}
echo "START $(date)"
for M in 24Gi 16Gi 8Gi; do run $M; done
echo "DONE $(date)"
