# Raw results

One JSON line per measured run, 1,431 runs in total. Test IDs refer to sections of
[`docs/DETAILED_REPORT.md`](../docs/DETAILED_REPORT.md); sweep scripts are in
[`scripts/experiments/`](../scripts/experiments/).

## Data files

| File | Content | Runs | Produced by | Tests |
|---|---|---|---|---|
| `qwen2.5-3b/results.jsonl` | 3B experiment 1: baseline, 18 cyclic configurations (G groups × K pods), resident controls; prompts 128 / 512 / 2048 tokens, 16 tokens | 219 | `sweep.sh` | B1–B5 |
| `qwen2.5-3b/results2.jsonl` | 3B experiment 2: partial residency, kernel paging, speculative decoding (chain), batching | 195 | `sweep2.sh` (deadlocked, rows removed), `sweep2b.sh`, `sweep2c.sh`, `sweep2d.sh` | B6–B9 |
| `qwen2.5-3b/results3.jsonl` | 3B single-pod streaming, 1–9 read slots | 63 | `sweep3.sh` | B10 |
| `qwen2.5-3b/results4.jsonl` | Latency optimisation round 1 (direct-I/O rows are invalid, see D1) | 33 | `sweep_opt1.sh` | D1 |
| `qwen2.5-3b/results5.jsonl` | Round 2: tree speculation, Hugging Face drafter | 72 | `sweep_opt2.sh` | D3 |
| `qwen2.5-3b/results6.jsonl` | Round 3: lean drafter | 72 | `sweep_opt3.sh` | D4 |
| `qwen2.5-3b/results7.jsonl` | Round 4: direct I/O end to end, 129-node and deep trees | 75 | `sweep_opt4.sh` | D5 |
| `gpt-oss-20b/results.jsonl` | gpt-oss experiment 1: cyclic streaming | 192 | `sweep_g1.sh` | C1–C4 |
| `gpt-oss-20b/results2.jsonl` | gpt-oss experiment 2: single pod, partial residency, kernel paging (repetitive prompt, see C7), batching | 183 | `sweep_g2.sh`, `sweep_g2d.sh`, `sweep_g2e.sh` | C5–C8 |
| `six-models/<model>.jsonl` | Six-model test: baseline, single-pod streaming, tree speculation, kernel paging at 2–3 limits; 3 prompts, 32 tokens | 270 | `sweep_models.sh` (1.5B, 7B, 14B; Qwen3-30B-A3B via `moe_chain.sh`), `sweep_models2.sh` (3B, gpt-oss via `post_chain.sh`) | E1–E3 |
| `six-models/*_paging2.jsonl` | Kernel-paging rows of both mixtures of experts, re-measured with the host page cache cleared (`cache: "evicted"`) | 45 | `rerun_moe_paging.sh` | E4 |
| `six-models/gptoss_paging_recheck.jsonl` | gpt-oss kernel paging with the old repetitive prompt (`recheck` field) | 12 | `recheck_paging.sh` | E5 |

## Fields of a run

| Field | Meaning |
|---|---|
| `approach`, `G`, `K`, `mode` | Configuration: approach label, layer groups, stage pods, `resident` or `cyclic`. Experiment-1 files have no `approach`; the baseline is `G=1, mode=resident` |
| `SLOTS`, `RESIDENT`, `LOAD_MODE`, `MEM_LIMIT`, `DRAFT`, `DRAFT_LAYERS`, `GAMMA`, `TREE`, `BATCH`, `PROMPT_FILE`, `CHAT`, `model` | The deployment's environment variables, as passed to `src/k8s/deploy.py` |
| `n_prompt` | Prompt length in tokens; `0` = the whole prompt file (chat prompts) |
| `rep` | Repetition index |
| `ttft_ms` | Time to first token |
| `tpot_ms` | List: time of each subsequent token |
| `tokens` | Generated token ids (a list of lists for batched runs; request 0 is the single-user prompt) |
| `spec` | Speculative runs: `accepted` drafts per cycle, `draft_ms`, `verify_ms`, `cycle_ms`, `nodes`, `decode_ms_total`, `tokens_out` |
| `traces` | Per-layer-group timings from the stage pods (arrival, wait for weights, compute, send) |
| `stats` | Memory: `VmHWM` (peak resident bytes) and more per stage pod and for the coordinator |

## Other files

| File | Content | Tests |
|---|---|---|
| `qwen2.5-3b/microbench.json`, `gpt-oss-20b/microbench.json` | One layer: cold and warm load, compute for 1–2048 tokens | A2 |
| `qwen2.5-3b/diskbw.json` (same file in `gpt-oss-20b/`) | Disk bandwidth with 1–18 parallel readers | A3 |
| `qwen2.5-3b/iotest2.txt` | Buffered vs O_DIRECT reads, page-cache change | A3 |
| `qwen2.5-3b/contention.json` | First contention run (its direct-I/O numbers are invalid: filesystem fallback) | D2 |
| `qwen2.5-3b/contention2.json` | Contention run with the fixed loader | D2 |
| `qwen2.5-3b/fp32_check.txt`, `gpt-oss-20b/gptoss_check.txt`, `six-models/layer_check_qwen3moe.txt` | Layer code vs Hugging Face in fp32 | A1 |
| `qwen2.5-3b/ref_128.json`, `gpt-oss-20b/ref_128.json` | Hugging Face `generate` output for the 128-token prompt | A1 |
| `qwen2.5-3b/spec_tie_check.txt`, `qwen2.5-3b/tie_check_round3.txt`, `six-models/fp32_ref_7b.txt`, `six-models/fp32_ref_all.txt` | fp32 logits at every divergence | F2 |
| `*/summary*.md`, `*/summary.csv` | Tables produced by the first analysis scripts (`scripts/legacy/`) | |
