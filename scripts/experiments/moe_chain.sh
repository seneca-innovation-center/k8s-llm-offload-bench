#!/bin/bash
# As run for the measurements in results/. Runs from the experiment directory (src/engine, mounted at /exp on the
# k3d node); needs k3d and kubectl on PATH and KUBECONFIG set. Output lands in src/engine/results*/ (git-ignored).
HERE="$(cd "$(dirname "$0")" && pwd)"
# Qwen3-30B-A3B pipeline: wait for download -> low-RAM split -> wait for the dense chain -> fp32 layer check -> sweep.
cd "$HERE/../../src/engine"
until grep -q "Qwen/Qwen3-30B-A3B-Instruct-2507 OK" results/dl_moe.log 2>/dev/null; do
  if grep -q "MISMATCH" results/dl_moe.log 2>/dev/null; then echo "DOWNLOAD MISMATCH"; exit 1; fi; sleep 120; done
echo "download done $(date)"
docker run --rm --cpuset-cpus 0-15 -v $PWD:/exp -w /exp cyclic:v1 python split_stream.py /exp/models/qwen3-30b-a3b /exp/weights_qwen3-30b-a3b 2>&1 | tail -3
docker run --rm -v $PWD:/exp alpine chmod -R a+rX /exp/weights_qwen3-30b-a3b
echo "split done $(date): $(ls weights_qwen3-30b-a3b | grep -c layer_) layer files, $(du -sh weights_qwen3-30b-a3b | cut -f1)"
until grep -q "^DONE qwen2.5-14b" results_models/sweep_14b.log 2>/dev/null; do sleep 120; done
echo "dense chain done $(date)"
echo "=== fp32 layer check (layers 0, 24, 47)"
docker run --rm --cpuset-cpus 16-31 -v $PWD:/exp -w /exp cyclic:v1 python layer_check.py /exp/weights_qwen3-30b-a3b 0,24,47 2>&1 | grep -vE "Warning|warn" | tee results_models/layer_check_qwen3moe.txt
"$HERE"/sweep_models.sh qwen3-30b-a3b 48 /exp/weights_qwen3-0.6b 38Gi 26Gi 16Gi > results_models/sweep_30b.log 2>&1
echo "ALL DONE $(date)"
