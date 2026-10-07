#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# After the MoE chain: run the same protocol on the two models tested earlier (3B, gpt-oss-20b) for one consistent table.
cd "$HERE/../../src/engine"
until grep -q "^ALL DONE" results_models/moe_chain.log 2>/dev/null; do sleep 180; done
echo "moe chain done $(date)"
[ -e weights_qwen2.5-3b ] || ln -s weights weights_qwen2.5-3b
[ -e weights_gpt-oss-20b ] || ln -s weights_gptoss weights_gpt-oss-20b
"$HERE"/sweep_models2.sh qwen2.5-3b 36 /exp/weights_05b 4Gi 2800Mi > results_models/sweep_3b.log 2>&1
"$HERE"/sweep_models2.sh gpt-oss-20b 24 none 26Gi 16Gi > results_models/sweep_gptoss.log 2>&1
echo "POST DONE $(date)"
