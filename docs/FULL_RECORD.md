# Cyclic layer streaming on halo — complete record

> **Lab record.** Written during the work (24 September to 5 October 2026) and kept as is, apart from neutral wording. The decision-oriented summary is [`REPORT.md`](REPORT.md); every test with its chart is in [`DETAILED_REPORT.md`](DETAILED_REPORT.md). Paths below refer to the original flat experiment directory on the test machine (hostname *halo*). In this repository the code is in `src/engine/` (deploy script in `src/k8s/`), the sweeps in `scripts/experiments/`, and the data in `results/qwen2.5-3b/` (was `results/`), `results/gpt-oss-20b/` (was `results_gptoss/`) and `results/six-models/` (was `results_models/`). Charts named `results/*.png` are in `docs/archive/results/`.

One file with everything: what was asked, what was decided, what was built, every test and its results, the questions that came up along the way and their answers, what went wrong, and what is left to try. Dates are 24–28 September 2026. Sizes are decimal (1 GB = 10⁹ bytes). Charts are in `results/` next to this file (also on halo in `~/cyclic-exp/results/`).

---

## 1. The request

Host a small model (~3B) on a one-node Kubernetes cluster on the `halo` machine and test this idea: split the model's layers into groups spread over several pods; a pod computes its group, unloads it, and loads its next group from disk while the other pods work, so that when the token comes back around, the next group is already in memory. Goal: **much less memory, same latency as the whole model in memory.**

Later additions from the conversation:
- "Two pods is maybe a bad number, test with more pods" → swept 2 to 18 pods.
- "No quantization."
- "Test other similar approaches, give me the best output and what we win each time, full report, no filler."

## 2. Decisions taken before starting

| Question | Decision |
|---|---|
| Topology: one pod with workers, or several pods? | 1 node, K pods, activations passed between pods over TCP |
| Where do unloaded weights come back from? | From the NVMe drive (the only option that really lowers memory; RAM would hide the cost) |
| Which latency must stay the same? | Both prompt processing (time to first token) and generation (time per token); vary the pod count until the reload is ready when the cycle comes back |
| Compute backend and model | PyTorch on CPU, bf16, Qwen2.5-3B-Instruct (36 layers) |
| Kubernetes without root on halo | k3d (k3s inside Docker) |
| Linux page cache would serve "disk" reads from RAM | `posix_fadvise(DONTNEED)` around every read, so reloads are real NVMe reads |
| What counts as "same latency" | Same as the full 36-layer model in one pod, measured the same way |
| CPU share on the shared machine; deliverable | Pin to 16 hardware threads (CPUs 16–31); report with charts |

## 3. Machine, model, cluster

- **halo**: AMD Ryzen AI Max+ 395 (16 cores / 32 threads), 125 GB unified RAM, one Micron 4600 NVMe SSD (1.9 TB, root filesystem). The drive is a PCIe 5.0 part but the CPU only has PCIe 4.0, so the link is 4.0 ×4: **7.1 GB/s measured maximum**, ~7.5 theoretical. No free ×4 slot for a second drive. Shared with ~10 other containers.
- **Model**: Qwen2.5-3B-Instruct, bf16. 36 decoder layers of 154 MB each = 5.55 GB, plus a 0.62 GB embedding/output matrix. Each layer written to its own `safetensors` file.
- **Cluster**: k3d, one server node, node container pinned with `docker update --cpuset-cpus 16-31`. Image: python 3.12 + torch 2.8 (CPU) + transformers.
- **Draft model** (for speculative decoding, added later): Qwen2.5-0.5B-Instruct, 0.99 GB bf16.

## 4. What was built

```
 coordinator ──► stage-0 ──► stage-1 ──► … ──► stage-(K-1) ──► stage-0 ──► … ──► coordinator
 (tokenizer,      layer 0     layer 1           layer K-1        layer K            (output layer → next token)
  embeddings,       ↓ free it, load layer K from NVMe while the others compute
  sampling,
  draft model)
```

- **Stage pod** (`stage_server.py`): pod *i* owns layer groups `i, i+K, i+2K, …` (a group = 36/G consecutive layers; one layer in the main setup). It holds one streamed group at a time (or `SLOTS` of them) plus any pinned groups. After computing a group it frees the buffer at once and a background thread starts reading the next group; the loader calls `posix_fadvise(DONTNEED)` before and after each read so nothing stays in the page cache. The KV cache (the attention state of the prompt) of every owned layer stays in memory: 151 MB in total, split across pods. Records per-group timings (arrival, wait for weights, compute, send).
- **Coordinator** (`coordinator.py`): tokenizer, embeddings, final norm, output layer, greedy sampling, HTTP API, and when enabled the draft model for speculative decoding or B prompts in lockstep for batching. Present and identical in every configuration including the baseline (~1.0 GB RAM).
- **Modes**: `resident` (all layers pinned: baseline and controls); `cyclic` (streamed); `RESIDENT=0-29 SLOTS=3` (partial residency); `LOAD_MODE=mmap` + a Kubernetes memory limit (kernel paging); `DRAFT=…` (speculative); `BATCH=B`.
- **Tools**: `deploy.py G K MODE [KEY=VALUE…]` generates and applies the manifests; `bench.py` runs prompts × repetitions inside the coordinator and appends JSON lines; `smoke.py` quick check; `sweep*.sh` the sweeps; `analyze.py` / `analyze2.py` tables and charts; `microbench.py` per-layer load vs compute; `diskbw.py` NVMe bandwidth vs parallel readers; `fp32_check.py` / `ref_check.py` equivalence against Hugging Face; `dropcache.py` evicts the weight files from the page cache.

## 5. Timeline

| When (halo time) | What |
|---|---|
| 24 Sep, afternoon | Cluster up, image built, weights split. Microbench: cold layer load 41 ms, warm 19 ms; compute per layer 3.5 ms (1 token), 9.9 ms (128 tokens), 27 ms (512), 166 ms (2048). Disk: 4.1 / 6.1 / 7.0 / 7.1 GB/s with 1 / 2 / 4 / 8+ readers. fp32 equivalence check vs Hugging Face: exact. |
| 24 Sep 18:19–19:18 | Sweep 1: baseline, 18 cyclic configurations (G ∈ {36,18,12,9,6,4}, K = 2…18), 3 resident controls, baseline again. 219 runs. |
| 25 Sep | Analysis, `REPORT.md`, two independent verification passes (see §12), corrections. |
| 27 Sep 12:21 | Sweep 2 (alternatives) started. Deadlocked after 20 min on the single-pod partial-residency config at the 2048-token prompt (see §11). Fixed at 13:45, restarted. |
| 27 Sep 13:45–16:51 | Sweep 2b: partial residency, mmap + limit, speculative decoding, batching, closing baseline. the remote connection to halo dropped for ~2 h in the middle; the sweep kept running. |
| 27 Sep 18:49–19:00 | Sweep 2c: speculative decoding redone with a non-repeating prompt (see §11). |
| 27 Sep 19:01–19:20 | Sweep 2d: mmap + limit redone with the page cache really dropped (see §11). |
| 27–28 Sep | `FINAL_REPORT.md`, third verification pass, corrections. This record. |
| 28 Sep – 2 Oct | gpt-oss-20b replication (`FINAL_REPORT_GPTOSS.md`); literature check; latency optimisation rounds 1–4 (§12b). |
| 4 Oct 17:47 – 5 Oct 04:30 | Cross-model test: 1.5B → 7B → 14B → Qwen3-30B-A3B → 3B and gpt-oss re-runs; fp32 checks; kernel-paging re-measurements (§12b, `REPORT_MODELS.md`). |

## 6. Experiment 1 — cyclic streaming (the original idea)

66 configuration × prompt combinations, 219 runs. Prompts of 128, 512 and 2048 tokens; 16 generated tokens; 3 repetitions (baseline 5 + 5, at the start and end of the sweep).

### 6.1 Memory

One layer per pod (G=36), 128-token prompt, stage pods only. Baseline stage pod: 5.55 GB of weights, 5.96 GB of RAM.

| Pods | Layers in memory at once | Weights | Weight saving | Stage-pod RAM | RAM saving |
|---|---|---|---|---|---|
| 2 | 2 | 0.31 GB | **94%** | 0.99 GB | **83%** |
| 3 | 3 | 0.46 GB | 92% | 1.40 GB | 76% |
| 4 | 4 | 0.62 GB | 89% | 1.81 GB | 70% |
| 6 | 6 | 0.92 GB | 83% | 2.64 GB | 56% |
| 9 | 9 | 1.39 GB | 75% | 3.86 GB | 35% |
| 12 | 12 | 1.85 GB | 67% | 5.10 GB | 14% |
| 18 | 18 | 2.77 GB | 50% | 7.57 GB | −27% (more than baseline) |

- Weight saving = 1 − K/36 exactly. Only K layers of 36 are in memory at any moment; the other layers exist only on the NVMe.
- Each pod carries ~250 MB of runtime (Python + PyTorch) plus its share of the 151 MB KV cache, so RAM breaks even with the baseline at about 14 pods.
- Counting the coordinator (present in both): 2 pods use 0.93 GB of weights instead of 6.17 GB (−85%) and 2.0 GB of RAM instead of 7.0 GB (−71%).
- Long prompts add 0.8–0.9 GB of temporary activations per pod; 18 pods at 2048 tokens peaked at 22.5 GB.
- Grouping several layers per pod (G < 36) holds more layers for the same K and buys no latency (see 6.4).

Chart: `results/memory.png`.

### 6.2 Prompt processing (time to first token)

| Prompt | Baseline (median; run-to-run range) | Best cyclic | Ratio | Pods |
|---|---|---|---|---|
| 128 tokens | 0.46 s (0.42–0.58) | 0.91 s | 1.99× | 4 |
| 512 tokens | 1.35 s (1.17–1.57) | 1.64 s | 1.22× | 3 |
| 2048 tokens | 6.67 s (6.37–7.10) | 6.66 s (6.42–7.12) | 1.00× | 2 |

With a long prompt each layer computes for 166 ms while a load takes 41 ms, so the loads finish in the background (0–22 ms of waiting in the whole 2048-token prefill with one layer per pod). Every other configuration at 2048 tokens was 5–15% slower (slower compute while other pods load, plus ~0.1 s of pod-to-pod hops). Short prompts don't give the loads enough cover. Chart: `results/ttft_ratio.png`.

### 6.3 Token generation

Per generated token, 512-token prompt, one layer per pod. Baseline: **130 ms**.

| Pods | Per token | vs baseline | Waiting for weights | Compute | Pod-to-pod hops |
|---|---|---|---|---|---|
| 2 | 1013 ms | 7.8× | 782 ms | 188 ms | 26 ms |
| 3 | 822 ms | 6.3× | 555 ms | 216 ms | 30 ms |
| 4 | 781 ms | 6.0× | 464 ms | 269 ms | 35 ms |
| 6 | 772 ms | 5.9× | 437 ms | 276 ms | 38 ms |
| 9 | 770 ms | 5.9× | 416 ms | 283 ms | 38 ms |
| 12 | 774 ms | 6.0× | 409 ms | 304 ms | 39 ms |
| 18 | 769 ms | 5.9× | 400 ms | 308 ms | 38 ms |

(The three breakdown columns come from the stage-pod traces and exclude the coordinator's 13–19 ms per token, so they don't sum exactly to the total.) With the 128-token prompt: 6.3–8.2×. With 2048 tokens: 3.6–5.0×, because the baseline itself slows to 214 ms there. In absolute terms: **0.77–1.07 s per token whatever the number of pods.** Charts: `results/decode_latency.png`, `results/decode_breakdown.png`.

### 6.4 Why more pods don't help

The idea needs *load time of a group ≤ compute time of the other K−1 pods*. Alone, a layer loads in 41 ms and computes in 3.5 ms, so ~13 pods should be enough. Three things break it:

1. **All pods share one NVMe.** Each pod starts its next load as soon as it finishes computing, so ~K reads are in flight, and each load gets slower: 49 ms with 2 pods, 62 with 3, 78 with 4, 119 with 6, 179 with 9, 231 with 12, 318 with 18. More pods add head start and load time in equal measure.
2. **The NVMe saturates**: 4.1 GB/s with one reader, 6.1 with two, 7.0 with four, 7.1 from eight on. Every token re-reads all 36 layers = 5.55 GB → 0.78 s per token from 4 pods on. That is the flat line. (2 pods: ~6.1 GB/s → ~0.91 s floor, measured 1.01 s.)
3. **Loading slows the computing**: compute per layer rises from ~4 ms (pods, nothing loading) to 5–9 ms while other pods load. Shared memory bandwidth and/or CPU time in the loader; not separated.

Only the number of pods matters, not the grouping: every 2-pod configuration is ~1.0 s per token whether 2 or 18 layers are in memory (G = 36 … 4); every 3-pod one 0.82–0.86 s; everything from 4 pods on 0.77–0.79 s.

Baseline latency this way would need ~43 GB/s of weight bandwidth (5.55 GB in 130 ms), 6× this NVMe.

### 6.5 Cost of splitting alone (controls)

K pods, each keeping its own 36/K layers permanently in memory, no reloading, 512-token prompt: 4 pods 167 ms per token (1.28×), 12 pods 171 ms (1.31×), 36 pods 172 ms (1.32×); first-token times within noise of the baseline. The pipeline itself costs ~30% per token (hops ~13 ms plus slightly slower compute per pod) before any reloading.

## 7. Experiment 2 — four alternatives

195 runs. Same machine, model, cluster and metrics.

### 7.1 Partial residency: one pod keeps most layers, streams the last few

| Pinned / streamed / slots | Layers saved | Total RAM (incl. coordinator) | Per token (512) | vs baseline |
|---|---|---|---|---|
| 33 / 3 / 3 | 0 | 7.08 GB (0%) | 176 ms | 1.33× |
| 30 / 6 / 3 | 3 | 6.60 GB (−7%) | 235 ms | 1.76× |
| 27 / 9 / 3 | 6 | 6.14 GB (−13%) | 298 ms | 2.24× |
| 24 / 12 / 4 | 8 | 5.84 GB (−17%) | 339 ms | 2.55× |
| 18 / 18 / 4 | 14 | 4.92 GB (−30%) | 478 ms | 3.59× |

The first row saves nothing and still costs 1.33×: hidden background reads slow the compute of the pinned layers. Each layer saved beyond the slot count costs one extra load per token (~20–30 ms after overlap). First-token time unaffected at 512/2048 tokens, 1.17–1.47× at 128.

### 7.2 mmap + memory limit: let the kernel page the weights

Weights memory-mapped, pod capped with a Kubernetes memory limit; the kernel evicts and re-faults pages.

| Limit | Total RAM | Per token (512) | vs baseline | First token (512) |
|---|---|---|---|---|
| 6 GiB (everything fits) | 7.07 GB (0%) | 131 ms | 0.99× | 0.91× |
| 4 GiB | 5.38 GB (−24%) | 1029 ms | 7.7× | 1.60× |
| 3 GiB | 4.31 GB (−39%) | 1399 ms | 10.5× | 1.90× |
| 2 GiB | 3.23 GB (−54%) | 1413 ms | 10.6× | 1.88× |
| 1.5 GiB | 2.70 GB (−62%) | 1455 ms | 10.9× | 1.95× |

Same disk floor, worse execution: a single reader at ~4 GB/s effective. At equal memory (3.2 GB) the explicit 4-pod pipeline is 1.8× faster. Zero engineering, worst result.

### 7.3 Cyclic streaming + speculative decoding

A 0.5B draft model in the coordinator proposes γ tokens; the streamed 3B model verifies all of them in one pass (a small prompt) and keeps the prefix it agrees with plus one token of its own. One read of the model yields several tokens. Output = the 3B model's own greedy output. Measured with a non-repeating 750-token text, 512-token prompt, 32 generated tokens.

| Config | Accepted drafts per cycle | Tokens per cycle | Draft | Verify | Cycle | Per token | vs baseline | Total RAM |
|---|---|---|---|---|---|---|---|---|
| all resident + draft, γ=5 | 1.82 / 5 | 2.8 | 167 ms | 137 ms | 309 ms | 120 ms | 0.91× | 8.35 GB (+18%) |
| cyclic 2 pods + draft, γ=5 | 1.82 / 5 | 2.8 | 194 ms | 1008 ms | 1209 ms | 441 ms | 3.3× | 3.48 GB (−51%) |
| cyclic 4 pods + draft, γ=5 | 1.82 / 5 | 2.8 | 212 ms | 778 ms | 994 ms | 365 ms | 2.75× | 4.47 GB (−37%) |
| cyclic 4 pods + draft, γ=8 | 2.10 / 8 | 3.1 | 321 ms | 786 ms | 1114 ms | 371 ms | 2.8× | 4.53 GB (−36%) |

The verify pass costs the same as a plain token (disk-bound) but produces 2.8 tokens: the cyclic penalty drops from 7.6× to 3.3× (2 pods). The draft model is slow on CPU (35–40 ms per drafted token), which is why speculative decoding barely helps the all-resident model here. More drafts (γ=8) don't help: acceptance falls. Chart: `results/speculative.png`.

### 7.4 Cyclic streaming + batching

B prompts of 128 tokens generated in lockstep. Per-user latency = time per step; throughput = B tokens per step.

| Concurrent requests | Baseline per user | Cyclic 4 pods per user | Ratio | Baseline tok/s | Cyclic 4 pods tok/s |
|---|---|---|---|---|---|
| 1 | 122 ms | 783 ms | 6.4× | 8 | 1.3 |
| 16 | 188 ms | 795 ms | 4.2× | 85 | 20 |
| 64 | 541 ms | 1015 ms | 1.9× | 118 | 63 |
| 256 | 1897 ms | 2475 ms | 1.3× | 135 | 103 |
| 512 | 3418 ms | 4058 ms | 1.19× | 150 | 126 |

The disk cost per step is fixed (~0.78 s) while compute grows with the batch, so the gap closes: at 512 users the pipeline delivers 84% of the baseline's throughput with 4 of 36 layers in memory. Per-user latency never drops below the floor. KV cache: 37 KB per token per request (3.6 GB at 512 × 192), same on both sides. 2 repetitions (1 at 512). Chart: `results/batching.png`.

## 8. All approaches side by side — what we win each time

Single user, 512-token prompt. Memory = all pods including the coordinator. Baseline 7.07 GB, 133 ms per token.

| Approach | Config | Total RAM | RAM saved | Per token | vs baseline | Output |
|---|---|---|---|---|---|---|
| Baseline | 36 layers in one pod | 7.07 GB | – | 133 ms | 1.0× | – |
| Cyclic streaming | 2 pods | 2.26 GB | **68%** | 1013 ms | 7.6× | identical |
| | 4 pods | 3.25 GB | 54% | 781 ms | 5.9× | identical |
| | 18 pods | 10.24 GB | −45% | 769 ms | 5.8× | identical |
| Partial residency | −3 layers | 6.60 GB | 7% | 235 ms | 1.8× | identical |
| | −14 layers | 4.92 GB | 30% | 478 ms | 3.6× | identical |
| mmap + memory limit | 2 GiB limit | 3.23 GB | 54% | 1413 ms | 10.6× | identical |
| **Cyclic + speculative** | 2 pods, γ=5 | 3.48 GB | **51%** | 441 ms | **3.3×** | identical |
| | 4 pods, γ=5 | 4.47 GB | 37% | 365 ms | 2.75× | identical |
| Speculative alone | all resident + draft | 8.35 GB | −18% | 120 ms | 0.91× | identical |
| Cyclic + batching (128-token prompt) | 4 pods, 512 users | – | 4 of 36 layers | 4058 ms/user | 1.19× batched baseline; 84% of throughput | identical* |

\* one near-tie token flips under bf16 rounding (see §10).

Chart: `results/frontier.png`.

**Conclusions**
- Memory: the idea works as intended. 2 pods → 85% less weight memory, 68–71% less RAM.
- Prompt processing: same latency for long prompts (within noise); 1.2–2.8× for short ones.
- Token generation: 6–8× slower with any number of pods. Not fixable on this machine: every token re-reads the full 5.55 GB, the NVMe gives 7.1 GB/s, floor 0.78 s.
- Best single-user trade-off: cyclic 2 pods + speculative decoding, 51% less RAM at 3.3×.
- Best for many users: cyclic 4 pods with batches of 256+, 76–84% of baseline throughput.
- Don't use a memory limit on a memory-mapped model: slowest option, and it doesn't reduce host memory unless the page cache is cold.
- Baseline latency for one user needs ~43 GB/s of weight bandwidth: different hardware (PCIe 5.0 + several NVMe in RAID-0, or a GPU that holds the model).

## 9. Questions that came up, and the answers

**How much memory did we gain each time?** Weight saving is 1 − pods/36: 94% with 2 pods, 92% with 3, 89% with 4, 83% with 6, 75% with 9, 67% with 12, 50% with 18. Real RAM saves less (each pod costs ~250 MB of runtime); from ~14 pods on you use more RAM than the single full-model pod.

**2 pods — how many layers on each?** Each pod is responsible for 18 layers (pod 0 the even ones, pod 1 the odd ones) but holds only one at a time. While pod 1 computes layer 1, pod 0 drops layer 0 and loads layer 2. Two of 36 layers in memory at any moment.

**3 pods?** 12 layers each (0,3,6…; 1,4,7…; 2,5,8…), one in memory at a time; 3 layers in memory in total.

**18 pods?** 2 layers each (pod i: layers i and i+18); 18 layers in memory at once, but 18 copies of the runtime push RAM above the baseline.

**If we add the memory together it's the same, right? We host the whole model, just split.** No. In cyclic mode only K layers exist in memory across all pods; the other 36−K are only on the NVMe. "The whole model split across pods with no reloading" is what the resident controls did: same 5.55 GB of weights plus extra runtime.

**When the cycle comes back, the new layer should be ready — we just need the right number of pods.** That was the goal and it was searched for. It has no solution here: adding a pod adds ~4–8 ms of cover per layer but also adds one more concurrent reader on the same NVMe, which slows every load by about the same amount. From 4 pods on the NVMe is saturated and latency is flat at ~0.78 s per token.

**Does the disk make latency higher?** Yes, that is the entire extra latency. Unloading is free (memory is just released). Loading is the cost, and it is set by the amount of data (5.55 GB per token), not by the number of requests.

**Does every token need all 36 layers, one after another?** Yes. Layer 0, then 1, … then 35, strictly in order, for each generated token; then the next token starts again at layer 0. A prompt goes through in one pass for all its tokens together; the answer needs one pass per token.

**So one cycle per generated token?** Yes. 100 generated tokens = 100 cycles = 3,600 layer loads ≈ 550 GB read from the NVMe.

**Can we do 100 tokens in one cycle?** Not for one conversation: token 2 depends on token 1. Two ways to get more tokens per cycle: batching (one cycle advances many users' conversations by one token each) and speculative decoding (a small model guesses several tokens, the big model checks them in one cycle).

**If we send 100 requests at once they all finish together at the same speed?** They advance in lockstep and finish together; throughput goes up ~100×. Each user still waits one cycle per token, and the full-model baseline also gets faster per token with batching, so the gap only closes at hundreds of concurrent users (measured: 1.9× at 64, 1.3× at 256, 1.19× at 512).

**Speculative decoding — do we host two models?** Yes: the 3B model split and cycled as before, plus a separate 0.5B model of the same family kept in the coordinator (~1 GB). Per cycle: the small model writes ~5 tokens, the big model checks them in one pass, keeps the ones it agrees with and corrects the first disagreement.

**Does it affect quality?** No. The big model checks every guess and only keeps tokens it would have produced itself; the output is identical to running the big model alone. Batching is also exact.

**Quantization?** Storing weights in 8 or 4 bits would cut the bytes per cycle 2–4×. Excluded on request.

**Disk or NVMe?** Same thing: halo has one drive, a Micron 4600 NVMe SSD holding the root filesystem. Every "disk" in the reports means this NVMe.

**Can't we speed the NVMe up?** No, not on this machine. The CPU (Strix Halo) only has PCIe 4.0, so the drive's PCIe 5.0 rating can't be used; there is no free ×4 slot for a second drive; we already get 95% of the link's capacity. What remains is more tokens per read: tree speculation (verify many candidates in one disk-bound pass), a faster draft model on the iGPU, prompt-lookup drafting; realistically ~1.5–2× baseline latency at ~50% less RAM. True 1× for one user needs ~43 GB/s of weight bandwidth, i.e. other hardware.

**The approach in three sentences.** The moment a pod finishes its layer it frees it (instant) and a background thread immediately starts reading the next layer from the NVMe, while the token moves on to the next pod and keeps being processed there. Processing waits only when the token comes back around to a pod whose next layer hasn't finished loading. That happens on almost every layer during generation (464 ms of waiting per token with 4 pods), because loading a layer (78 ms with 4 pods reading at once) takes longer than the other pods' compute on a single token (3 × ~7 ms), and adding pods slows each load by as much as it adds compute time.

## 10. Correctness

- Our layer implementation vs Hugging Face `transformers`, both fp32, 128-token prompt: maximum logit difference **0.0**, top-1 identical at every position (`results/fp32_check.txt`).
- Every cyclic, partial-residency and mmap run produced exactly the same tokens as the baseline for the same prompt: all 219 runs of sweep 1, all repetitions in sweep 2.
- Speculative decoding with the non-repeating prompt: identical in all runs. With the 128-token original prompt, all speculative and all batched runs differ from the baseline at token 15. fp32 shows a near-tie there: " simplicity" 18.532 vs " power" 18.437 (0.5% margin). A single-token bf16 pass picks " power", a multi-token bf16 pass picks " simplicity" (the fp32 choice). Both are legitimate bf16 roundings of a tie, not a pipeline bug (`results/spec_tie_check.txt`).
- Eviction is real: pods report at most `SLOTS` streamed groups resident; load counters equal tokens × owned groups; cold reload 41 ms vs 19 ms from the page cache.
- Baseline drift over the 1-hour sweep 1: first-token time +4/+15/+9% and per-token +2/+6/+5% (128/512/2048 tokens) between the start and end runs. Ratios below ~1.1× are within noise. The 6–8× and 3.3× results are far outside it.

## 11. Things that went wrong, and what was done about them

1. **Single-pod deadlock (27 Sep).** With one pod, each layer group's output was sent to the pod itself over TCP; at 2048-token prompts the 8 MB hidden state exceeded the socket buffer and the pod blocked on itself. Found with thread dumps (`faulthandler` on SIGUSR1). Fixed: consecutive groups on the same pod are now chained in-process. Single-pod configurations were rerun without the loopback hops; the two partial-residency results collected before the fix were discarded.
2. **Memory-limit measurements were fake, twice.** Attempt 1 didn't drop the page cache: the weights were served from host RAM cached by an earlier pod, and a "1.5 GiB" pod ran at full speed. Attempt 2 dropped the cache *before* deploying, while the previous pod still had the files mapped, so nothing was evicted. Attempt 3 (`sweep2d.sh`: deploy, then drop, then measure) gave the numbers in §7.2. Lesson: a cgroup limit on a memory-mapped model does not reduce host memory unless nothing else has the file cached.
3. **Repetitive test prompt.** The original test text repeats its paragraphs; at 512/2048 tokens the model continued the repetition and the draft model was right 100% of the time (6.0 tokens per cycle, 0.47× on the baseline). Those runs are an artifact and are excluded; speculative decoding was redone with a 750-token non-repeating text (36% acceptance).
4. **Baseline drift** on the shared machine (above). Baselines were repeated at the start and end; ratios use the pooled median.
5. **Connectivity.** the remote connection to halo dropped for ~2 hours on 27 Sep; the sweep ran unattended and completed. Several progress monitors were silent because of output buffering and had to be checked by hand.
6. **First report had errors** (mixed GB/GiB units, an inverted conclusion about layer grouping, an understated baseline drift, miscounted runs, an off-by-one in tokens per cycle, wrong cross-references). All were found by the verification passes below and corrected.

## 12. How the reports were checked

Each report was reviewed after writing by independent review passes that re-derived every number from the raw `results*.jsonl`, read the code to confirm the methodology statements, and checked internal consistency; each finding was then challenged by a second agent trying to refute it. Confirmed findings were applied: 69 on the first draft of `REPORT.md` (most of them units and wording, several substantive), 5 distinct ones on its second draft, 27 on `FINAL_REPORT.md`. Every number in this record is from the corrected versions.

## 12b. Follow-up work (28 Sep – 2 Oct 2026)

### Single-pod sequential streaming (3B)
The cycle run inside one pod (layers in order, layer i freed, layer i+S loading, up to 4 reads in flight) reaches the same 0.78 s floor as the multi-pod version with less RAM (2.14 GB at 4 slots vs 3.25 GB for 4 pods, same ~0.79 s). With 1 slot (no prefetch) 1.65 s per token. Conclusion: the pods were only ever a way to get several concurrent NVMe reads. Added to `FINAL_REPORT.md` §5.5.

### gpt-oss-20b (mixture of experts, 24 layers × 1.65 GB) — full replication
Own gpt-oss layer implementation (attention sinks, sliding window, YaRN rope, top-4-of-32 experts), verified against Hugging Face in fp32 (≤1e-7 relative). Results in `FINAL_REPORT_GPTOSS.md`:
- Cyclic streaming: −92% weight memory with 2 pods, but **33–44× slower per token** (floor 39.5 GB ÷ 7.1 GB/s = 5.5 s): the scheme loads all 32 experts of a layer while a token uses 4, i.e. 8× more bytes than used. Even the 2048-token prompt no longer hides the loads (1.44×), because streaming 1.65 GB layers slows the compute of every layer by 30–55%.
- Single-pod, partial residency and batching scale the same way (3.9× before saving anything; 1.78× per user at 512 users).
- **Kernel paging (mmap + memory limit) is the one approach that suits the mixture of experts**: the kernel loads only the experts a token touches. First reported as 1.22× with −15% RAM, 1.58× at −35%, 1.87× at −54% (16 GiB limit), 4.7× at −74%. **Corrected 5 Oct:** those rows were measured on a degenerate output (the repetitive prompt makes gpt-oss emit "…" sixteen times, which routes to the same experts every step). With real prompts: **3.0–4.0× at −30%, 7.4–8.0× at −54%** (see the cross-model section below). The first prompt pass pays 4–11×. For the dense 3B the same approach was the worst (8–11×) in both measurements.
- Things that went wrong: root-owned weight files made the cache-drop step fail silently, so 15 memory-limit rows measured the page cache and were discarded and rerun; a reboot of halo interrupted the rerun, finished with a third script.

### Literature check (2 Oct)
Layer streaming from disk is known and known to be bandwidth-bound (AirLLM, ZeRO-Inference, FlexGen); MoE systems stream and cache experts (MoE-Infinity, HOBBIT, Pre-gated MoE, ExpertFlow); SpecExec/Sequoia use trees of hundreds of candidates because an offloaded target verifies them for the price of one token; LLM-in-a-flash bundles reads to keep flash throughput high. Our contributions are the measurements on a unified-memory CPU with one NVMe (pod count vs concurrent reads, compute slowdown under load, single-pod equivalence, the MoE paging result) and the exact-output constraint. Sources are listed in the conversation summary of 2 Oct.

### Latency optimisation of the 3B at ≤ 3.5 GB (2 Oct)
Decisions taken at the start of the round: budget ≤ 3.5 GB; bf16 0.5B draft, no quantization anywhere; prompts = the 512-token prose text + 3 chat-style instructions; tree speculation first.
- Round 1: speculative decoding moved onto the single-pod streamer (4 slots): 428 → 350 ms per token (verify pass 1004 → 776 ms, no hops). A spectacular "direct I/O" result (483–505 ms for plain streaming) was an artifact: btrfs falls back to buffered reads for part of an O_DIRECT read and 48% of the weights were found in the page cache; the loader was fixed (EOF-safe, evicts fallbacks) and verified cache-free.
- Round 2: tree speculation (best-first tree from the draft, verified in one pass with tree attention and KV compaction; proven exact in fp32 locally): 23–65-node trees beat the 5-token chain on every prompt (3.9–5.4 tokens per read vs 2.4–3.8): prose 324 ms, chat 217–248 ms. Trees of 129–201 nodes lost because drafting them with the Hugging Face model cost 0.5–1.5 s per cycle.
- Round 3: lean drafter (0.5B through our layer code, tree attention, same KV compaction, one forward per level): drafting a 65-node tree 290–620 → 210–240 ms; **242 ms on prose (1.9× vs the normal 126–129 ms baseline; the same-round prose baseline ran under load at 173 ms), 198–248 ms on the chat prompts at 4 slots (1.7–2.2×), 194–238 ms at 5 slots; 3.2–3.4 GB (−51 to −54%)**. Output identical to the baseline in 48 of 60 runs; the 65-node tree is identical in 22 of 24 (one explain-prompt repetition flips at a near-tie token, fp32 margin 0.06 logits); the 12 differing runs all sit at near-tie tokens (margins 0.05–0.06 logits, 0.2%).
- Round 4 (direct I/O end to end, 129-node lean tree, deep 19-node tree; run under load from other users, so only within-round comparisons count): direct I/O gives no end-to-end gain for streaming (65-node tree 279 → 288 ms on prose, +2–3% on chat prompts; plain streaming 808 → 834 ms), helps only partial residency (209 → 195 ms) and first-token time for plain streaming; the 129-node tree is slower than the 65-node tree on every prompt; the deep 19-node tree matches or beats it on chat prompts (239 / 197 ms) but not on prose.
- Cache-free contention microbench: direct I/O does not raise bandwidth (6.8 vs 7.1 GB/s, link-bound) but reduces the compute slowdown during loads for single-token work (1.18× vs 1.44×), not for prompt-sized work (1.65× vs 1.72×).


### Cross-model test: which approach is best, on six models (4–5 Oct)
Question: is the 3B result (streaming + tree speculation) and the gpt-oss result (kernel paging) a rule? Same protocol on Qwen2.5-1.5B / 3B / 7B / 14B (dense) and gpt-oss-20b / Qwen3-30B-A3B (mixtures of experts): baseline, single-pod streaming (4 slots), streaming + 64-node tree speculation (0.5B or Qwen3-0.6B lean drafter), kernel paging at 2–3 limits; 512-token prose + 2 chat prompts, 32 tokens, 3 repetitions. 270 runs, plus 57 replication and diagnostic runs. Full report: `REPORT_MODELS.md`; chart `results_models/models.png`; data `results_models/*.jsonl`.
- New layer code for Qwen3 (q/k norm) and Qwen3-MoE (softmax over 128 experts, top-8 renormalised), checked in fp32 against transformers (Qwen3-0.6B ≤ 7.8e-3 on outputs of scale 10⁴; Qwen3-30B-A3B layers 0/24/47 ≤ 5e-5). Qwen3-30B-A3B (61 GB) downloaded at 4–15 MB/s over ~20 h and split with a low-RAM streaming splitter.
- **Dense models: streaming + tree speculation wins everywhere**, kernel paging is worst everywhere. 1.5B: −24…28% at 1.7–2.2× (the 1 GB drafter eats the saving); 3B: −51…54% at 1.5–1.9×; 7B: −61…62% at 1.7–2.9×; 14B: −74% at 2.1–2.7×. Plain streaming sits on the NVMe floor for all six models (within 5% of layer bytes ÷ 7.1 GB/s); the tree yields 3.3–6.4 tokens per read. Kernel paging on dense weights: 9–15×.
- **Mixtures of experts: kernel paging wins**, layer streaming is 32–37×, tree speculation does not help (each read still costs 5.6–8.4 s). gpt-oss-20b: 3.0–4.0× at −30%, 7.4–8.0× at −54%. Qwen3-30B-A3B: 1.7× at −38%, 2.6–3.3× at −57%, 4.3–5.2× at −72%. First-token cost of paging on MoE: 3.5–11×.
- **Correction to the gpt-oss report** found by this test: its kernel-paging rows (1.87× at −54%) used the old repetitive prompt and a degenerate "…"×16 output; see above. Two hypotheses were tested: host page-cache pressure (79 GB of cached downloads; refuted: evicting them and re-measuring with real prompts gave the same numbers as the sweep) and the degenerate output (confirmed: re-measuring with the repetitive prompt reproduces 1.85×).
- **Correctness**: 171 of 171 streaming/paging runs reproduce the baseline's tokens; tree speculation 34 of 45, the 11 others at bf16 near-ties (fp32 margins 0.27–1.16%; fp32 sides with the tree twice, with the baseline three times). Measured bf16 shape noise (full-sequence vs single-token computation, 20 positions on two small models): median 0.16–0.23, maximum 0.28–0.42 logit points; the maximum exceeds every gap (0.05–0.27) and three of the five gaps are below even the median. `fp32_ref.py` streams any model layer by layer in fp32 for these checks.
- Things that went wrong: the sweep script was copied without its executable bit (empty logs, relaunched); two helper launches were skipped because a `pgrep -f` matched its own command line (caught, relaunched); the remote connection to halo dropped several times (the chains on halo were unaffected).

### Files added
`draft_model.py` (lean drafter), `tree_test.py` / `tree_test2.py` (exactness tests, local 0.5B), `contention.py`, `iotest.py`, `fincore.py`, `tie_check.py`, `dl_gptoss.sh`, `split_gptoss.py`, `gptoss_check.py`, `traces.py`, `sweep_models.sh` / `sweep_models2.sh`, `moe_chain.sh`, `post_chain.sh`, `rerun_moe_paging.sh`, `recheck_paging.sh`, `fp32_checks.sh`, `fp32_ref.py`, `analyze_models.py`, `layer_check.py`, `moe_tiny_check.py`, `split_stream.py`, `dl_model.sh`, `sweep_g1/g2/g2d/g2e.sh`, `sweep3.sh`, `sweep_opt1–4.sh`, `analyze_opt.py`, `chart_opt.py`, `make_visual.py` (APPROACHES.html/pdf), `make_video.py` (APPROACHES.mp4), `prompts/`, `prompt_long.txt`; data in `results/results3–7.jsonl` and `results_gptoss/`.

## 13. What is left to try (not done)

- **Tree speculation**: verify many candidate continuations in one pass; the pass is disk-bound so its cost barely depends on the candidate count. Expected ~4–5 tokens per read → ~0.2 s per token (1.5–2× baseline) at ~50% less RAM.
- **Draft model on the iGPU** (Radeon 8060S): drafting drops from 170–320 ms per cycle to ~20 ms; a 1.5B draft with higher acceptance becomes affordable. Needs ROCm in the pod.
- **Prompt-lookup drafting** (n-gram, no model): free; strong for code/RAG-style output.
- **Other hardware**: a PCIe 5.0 box with 3–4 NVMe in RAID-0 reaches the ~43 GB/s needed for baseline latency.
- Quantized storage (excluded here) would cut the floor 2–4×.

## 14. Reproduce

```bash
cd <experiment directory> && . env.sh                    # PATH (k3d, kubectl) + KUBECONFIG
# one-time setup
docker build -t cyclic:v1 .
docker run --rm -v $PWD:/exp -w /exp -e HF_HOME=/exp/hf cyclic:v1 python split_weights.py /exp/weights
k3d cluster create cyclic --servers 1 --agents 0 --no-lb -v $HOME/cyclic-exp:/exp@server:0
docker update --cpuset-cpus 16-31 k3d-cyclic-server-0
k3d image import cyclic:v1 -c cyclic
# single configurations
python3 deploy.py 1 1 resident                                                 # baseline
python3 deploy.py 36 2 cyclic                                                  # cyclic, 2 pods
python3 deploy.py 36 1 cyclic RESIDENT=0-29 SLOTS=3                            # partial residency
python3 deploy.py 1 1 resident LOAD_MODE=mmap MEM_LIMIT=2Gi && python3 dropcache.py   # kernel paging (drop cache AFTER deploy)
python3 deploy.py 36 2 cyclic DRAFT=/exp/draft GAMMA=5 PROMPT_FILE=prompt_long.txt    # + speculative decoding
python3 deploy.py 36 4 cyclic BATCH=64 MAXLEN=192                              # + batching
kubectl exec coordinator -- python /exp/smoke.py 512 16                       # tokens, timings, memory
# sweeps and analysis
./sweep.sh                                                                     # experiment 1 → results/results.jsonl
./sweep2b.sh; ./sweep2c.sh                                                     # experiment 2 (their mmap rows are the invalid ones)
python3 -c "import json;L=[l for l in open('results/results2.jsonl') if json.loads(l)['approach']!='mmap'];open('results/results2.jsonl','w').writelines(L)"
./sweep2d.sh                                                                   # valid mmap rows
docker run --rm -v $PWD:/exp -w /exp cyclic:v1 sh -c "pip install -q matplotlib; python analyze.py > results/summary_table.md; python analyze2.py > results/summary2_tables.md"
k3d cluster delete cyclic                                                      # cleanup (weights and draft model stay)
```

## 15. Files (original experiment directory; see the note at the top for where they are in this repository)

| File | Content |
|---|---|
| `FULL_RECORD.md` | this file |
| `FINAL_REPORT.md` | comparison of all approaches |
| `REPORT.md` | detailed report of experiment 1 |
| `common.py`, `stage_server.py`, `coordinator.py`, `deploy.py`, `bench.py`, `smoke.py` | the system |
| `sweep.sh`, `sweep2b.sh`, `sweep2c.sh`, `sweep2d.sh` | the sweeps |
| `analyze.py`, `analyze2.py` | tables and charts |
| `microbench.py`, `diskbw.py`, `fp32_check.py`, `ref_check.py`, `dropcache.py`, `split_weights.py` | measurements, checks, tooling |
| `prompt.txt`, `prompt_long.txt` | test texts |
| `results/results.jsonl` (219 runs), `results/results2.jsonl` (195 runs) | raw data, one JSON per run, with per-group traces and memory stats |
| `results/summary.csv`, `summary_table.md`, `summary2.json`, `summary2_tables.md` | aggregated tables |
| `results/microbench.json`, `diskbw.json`, `fp32_check.txt`, `spec_tie_check.txt`, `ref_128.json` | supporting measurements |
| `results/*.png` | memory, ttft_ratio, decode_latency, decode_breakdown, frontier, speculative, batching |
