#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Why is gpt-oss kernel paging at 16Gi 7.4x today but 1.87x on 1-2 Oct? Hypothesis: global page-cache pressure
# (81 GB of cached downloads/splits) makes the kernel evict the pod's hot experts. Test: evict our big files from the
# page cache, then rerun the same configuration; then rerun once more with 64 tokens to see the steady state.
cd "$HERE/../../src/engine"
until grep -q "^DONE fp32 checks" results_models/fp32_ref_all.txt 2>/dev/null; do sleep 60; done
echo "START recheck $(date)"; free -g | head -2
for d in models weights weights_05b weights_qwen2.5-1.5b weights_qwen2.5-7b weights_qwen2.5-14b weights_qwen3-30b-a3b weights_qwen3-0.6b weights_gptoss draft; do
  [ -d $d ] && python3 dropcache.py $PWD/$d 2>&1 | tail -1
done
free -g | head -2
OUT=/exp/results_models/gptoss_paging_recheck.jsonl
run() { local M=$1 NEW=$2 TAG=$3
  python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_gptoss LOAD_MODE=mmap MEM_LIMIT=$M || { echo "DEPLOY FAIL $M"; return; }
  python3 dropcache.py $PWD/weights_gptoss
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps 3 --max-new $NEW --prompts 512 \
     --tag "{\"approach\":\"mmap\",\"G\":1,\"K\":1,\"mode\":\"resident\",\"model\":\"gpt-oss-20b\",\"PROMPT_FILE\":\"prompt_long.txt\",\"LOAD_MODE\":\"mmap\",\"MEM_LIMIT\":\"$M\",\"recheck\":\"$TAG\"}" || echo "BENCH FAIL $M"
}
run 16Gi 32 after-evict
run 16Gi 64 after-evict-64tok
run 26Gi 32 after-evict
python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_gptoss && timeout 1800 kubectl exec coordinator -- python /exp/bench.py --out $OUT --reps 3 --max-new 32 --prompts 512 \
  --tag "{\"approach\":\"baseline\",\"G\":1,\"K\":1,\"mode\":\"resident\",\"model\":\"gpt-oss-20b\",\"PROMPT_FILE\":\"prompt_long.txt\",\"recheck\":\"after-evict\"}"
echo "DONE recheck $(date)"
