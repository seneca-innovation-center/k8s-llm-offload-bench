# Engine

Everything that runs inside the pods or next to the weights. The files share `common.py` and expect to run from one flat
directory mounted at `/exp` (see [`../k8s/README.md`](../k8s/README.md)), so they are kept together here.

| File | Role |
|---|---|
| `common.py` | Weight loaders (cold read with page-cache eviction, O_DIRECT, mmap), model config, layer forward pass for Qwen2, gpt-oss, Qwen3 and Qwen3-MoE, tree attention masks, framed TCP messages, memory stats |
| `stage_server.py` | Stage pod: owns layer groups, streams or pins them, prefetches through read slots, computes, forwards activations |
| `coordinator.py` | Coordinator pod: tokenizer, embeddings, output layer, greedy loop, speculative decoding (chain and tree), batching, HTTP API |
| `draft_model.py` | Lean drafter: a small model run through `common.py`, drafts a tree level by level |
| `bench.py`, `smoke.py` | Benchmark a deployed configuration (one JSON line per run); quick check |
| `microbench.py`, `diskbw.py`, `contention.py`, `iotest.py`, `fincore.py`, `traces.py` | Load vs compute, disk bandwidth, compute slowdown while loading, buffered vs direct reads, page-cache residency, trace breakdown |
| `dropcache.py` | Evict weight files from the page cache (needed before measuring kernel paging) |
| `fp32_check.py`, `ref_check.py`, `layer_check.py`, `gptoss_check.py`, `moe_tiny_check.py`, `tree_test.py`, `tree_test2.py` | Correctness: our layer code vs Hugging Face in fp32, reference generation, exactness of tree speculation |
| `tie_check.py`, `fp32_ref.py` | fp32 logits at a divergence position (`fp32_ref.py` streams the model layer by layer, so a 30B model fits in a few GB) |
| `split_weights.py`, `split_stream.py`, `split_gptoss.py`, `dl_model.sh` | Download and split a checkpoint into one safetensors file per decoder layer |
| `prompt_long.txt`, `prompts/` | The 512-token prose prompt and the chat prompts used in the reports |
| `prompt.txt` | The original, repetitive prompt of the first experiments. Do not use it for cache-sensitive measurements (see `docs/REPORT.md` §6.3) |
