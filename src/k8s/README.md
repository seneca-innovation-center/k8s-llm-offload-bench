# Running the experiments on Kubernetes

The measurements ran on a single-node k3d cluster. The pods run the code from a hostPath `/exp` on the node, which is
one flat directory holding the code, the prompts and the model weights. In this repository that directory is
[`src/engine/`](../engine/); the weights go inside it (git-ignored).

## One-time setup

```bash
# from the repository root
docker build -t cyclic:v1 -f src/k8s/Dockerfile src/engine           # python 3.12 + torch 2.8 (CPU) + transformers
k3d cluster create cyclic --servers 1 --agents 0 --no-lb -v "$PWD/src/engine:/exp@server:0"
docker update --cpuset-cpus 16-31 k3d-cyclic-server-0                # optional: pin the node to 16 hardware threads
k3d image import cyclic:v1 -c cyclic

# weights: one safetensors file per decoder layer + embed_head.safetensors, written into src/engine/weights_<name>
cd src/engine
docker run --rm -v $PWD:/exp -w /exp -e HF_CACHE=/exp/hf cyclic:v1 python split_weights.py /exp/weights           # Qwen2.5-3B
docker run --rm -v $PWD:/exp -w /exp -e MODEL=Qwen/Qwen2.5-7B-Instruct -e HF_CACHE=/exp/hf cyclic:v1 python split_weights.py /exp/weights_qwen2.5-7b
docker run --rm -v $PWD:/exp -w /exp cyclic:v1 python split_stream.py /exp/models/<downloaded model dir> /exp/weights_<name>   # low-RAM splitter
docker run --rm -v $PWD:/exp -w /exp cyclic:v1 python split_gptoss.py /exp/gptoss /exp/weights_gptoss                         # gpt-oss: MXFP4 -> bf16
docker run --rm -v $PWD:/exp -w /exp cyclic:v1 python layer_check.py /exp/weights_<name> 0,1,27                              # fp32 check vs transformers
```

## One configuration

Run from `src/engine` (the generator writes a copy of each manifest to `k8s/` there):

```bash
P=PROMPT_FILE=prompt_long.txt     # always set it: the default prompt.txt is repetitive and makes some models loop
python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_qwen2.5-7b $P                                 # baseline: whole model resident
python3 ../k8s/deploy.py 28 1 cyclic WEIGHTS=/exp/weights_qwen2.5-7b SLOTS=4 $P                          # dense: one pod, 4 read slots
python3 ../k8s/deploy.py 28 1 cyclic WEIGHTS=/exp/weights_qwen2.5-7b SLOTS=4 $P \
        DRAFT_LAYERS=/exp/weights_05b TREE=8,16,16,12,8,4                                                 # dense: + 64-node tree speculation
python3 ../k8s/deploy.py 1 1 resident WEIGHTS=/exp/weights_qwen3-30b-a3b LOAD_MODE=mmap MEM_LIMIT=38Gi $P \
        && python3 dropcache.py weights_qwen3-30b-a3b                                                    # experts: kernel paging (drop cache AFTER deploy)
# chat prompts: PROMPT_FILE=prompts/chat_summary.txt CHAT=1, and bench.py --prompts 0 (whole prompt)
kubectl exec coordinator -- python /exp/bench.py --out /exp/results_local.jsonl --reps 3 --max-new 32 --prompts 512 \
        --tag '{"approach":"tree","model":"qwen2.5-7b"}'                                                 # measure: appends one JSON line per run
kubectl exec coordinator -- python /exp/smoke.py 512 16                                                  # quick check: tokens, timings, memory
```

Arguments: `G` layer groups, `K` stage pods, `MODE` (`resident` or `cyclic`). Every `KEY=VALUE` becomes an environment
variable of the pods; `MEM_LIMIT` becomes the stage pod's memory limit. The environment variables are documented at the
top of [`../engine/stage_server.py`](../engine/stage_server.py) and [`../engine/coordinator.py`](../engine/coordinator.py).

`DRY_RUN=1 python3 deploy.py ...` prints the manifest without touching a cluster. The files in [`examples/`](examples/)
were produced that way.

## Sweeps

[`../../scripts/experiments/`](../../scripts/experiments/) holds the sweep scripts exactly as run for the data in
`results/`, adapted only to this layout. Each one deploys a series of configurations and appends to a results file
under `src/engine/`. The mapping from script to data file is in [`../../results/README.md`](../../results/README.md).

## Notes

- The page cache matters. Streamed reads evict their files (`posix_fadvise(DONTNEED)`), but for kernel paging you
  must drop the weight files from the cache after the pod is up, or the pod reads from host RAM and the limit is
  meaningless (`dropcache.py`, `fincore.py` to check).
- Results depend on the drive. Measure its saturated bandwidth first:
  `docker run --rm -v $PWD:/exp -w /exp cyclic:v1 python diskbw.py` (reads the Qwen2.5-3B layer files in `weights/` with 1 to 18 parallel readers).
