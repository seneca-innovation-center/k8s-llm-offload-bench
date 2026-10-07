#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Kernel-paging rows of both mixtures of experts, re-measured with our large files evicted from the host page cache
# (the recheck showed gpt-oss 16Gi at 284 ms after eviction vs 1120 ms during the sweep). Rows carry cache=evicted.
cd "$HERE/../../src/engine"
echo "START moe paging rerun $(date)"
evict_all() { for d in models weights weights_05b weights_qwen2.5-1.5b weights_qwen2.5-7b weights_qwen2.5-14b weights_qwen3-30b-a3b weights_qwen3-0.6b weights_gptoss draft; do
  [ -d $d ] && python3 dropcache.py $PWD/$d >/dev/null 2>&1; done; free -g | sed -n 2p; }
run() { local NAME=$1 M=$2 PR=$3 P=$4 C=$5
  python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_$NAME LOAD_MODE=mmap MEM_LIMIT=$M $P $C || { echo "DEPLOY FAIL $NAME $M"; return; }
  evict_all
  local extra=",\"${P%%=*}\":\"${P#*=}\""; [ -n "$C" ] && extra="$extra,\"${C%%=*}\":\"${C#*=}\""
  timeout 3600 kubectl exec coordinator -- python /exp/bench.py --out /exp/results_models/${NAME}_paging2.jsonl --reps 3 --max-new 32 --prompts $PR \
     --tag "{\"approach\":\"mmap\",\"G\":1,\"K\":1,\"mode\":\"resident\",\"model\":\"$NAME\",\"LOAD_MODE\":\"mmap\",\"MEM_LIMIT\":\"$M\",\"cache\":\"evicted\"$extra}" || echo "BENCH FAIL $NAME $M $P"
}
for PF in "PROMPT_FILE=prompt_long.txt 512" "PROMPT_FILE=prompts/chat_summary.txt 0 CHAT=1" "PROMPT_FILE=prompts/chat_code.txt 0 CHAT=1"; do
  set -- $PF; P=$1; N=$2; C=${3:-}
  for M in 26Gi 16Gi; do run gpt-oss-20b $M $N $P "$C"; done
  for M in 38Gi 26Gi 16Gi; do run qwen3-30b-a3b $M $N $P "$C"; done
done
echo "DONE moe paging rerun $(date)"
