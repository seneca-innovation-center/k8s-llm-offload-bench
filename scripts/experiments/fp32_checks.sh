#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# After everything: fp32 reference at every divergence, all models x prompts (prints "no divergence" where none).
cd "$HERE/../../src/engine"
until grep -q "^POST DONE" results_models/post_chain.log 2>/dev/null; do sleep 180; done
echo "START fp32 checks $(date)"
for M in qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen2.5-14b qwen3-30b-a3b gpt-oss-20b; do
  for P in "prompt_long.txt" "prompts/chat_summary.txt CHAT" "prompts/chat_code.txt CHAT"; do
    docker run --rm --cpuset-cpus 0-15 -e EVICT=1 -v $PWD:/exp -w /exp cyclic:v1 \
      sh -c "python fp32_ref.py /exp/weights_$M /exp/results_models/$M.jsonl $P" 2>&1 | grep -vE "Warning|warn|^\s*$"
  done
done
echo "DONE fp32 checks $(date)"
