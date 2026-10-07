# Decision Rules for Streaming LLM Layers under Memory Pressure: Dense vs Mixture-of-Experts

**Dense models:** stream the layers from one pod and add tree speculation.
**Mixture-of-experts models:** let the kernel page the weights under a pod memory limit.
**Any model:** time per token cannot beat *layer bytes ÷ disk bandwidth*; four concurrent reads already reach it.

1,431 measured runs on six models (Qwen2.5-1.5B / 3B / 7B / 14B, gpt-oss-20b, Qwen3-30B-A3B), on a single-node Kubernetes cluster with 125 GB of unified RAM and one 7.1 GB/s NVMe drive. bf16 weights, no quantisation, every output checked token by token against the fully resident model.

## Executive decision table

Cost per generated token for one user, relative to the same model fully resident in one pod; ranges over three prompts.

| Situation | Recommended approach | Cost per token vs resident | Memory saved |
|---|---|---|---|
| Dense model, one user | One pod streaming with 4 read slots + 64-node tree speculation | 1.5–2.9× (3B–14B); 1.7–2.1× at 1.5B | 51–74% (3B–14B); 24–28% at 1.5B |
| Dense model, many users | Streaming + batching | 1.19× per user at 512 users, 84% of throughput (3B) | 32 of 36 layers off memory |
| Mixture of experts, one user | Memory-mapped weights + pod memory limit (kernel paging) | 1.7–5.2× (Qwen3-30B-A3B); 3.0–8.0× (gpt-oss-20b) | 29–73% |
| Dense model, kernel paging | Do not use | 8.5–14.9× | |
| Mixture of experts, layer streaming | Do not use | 31.8–37.2× | |
| Several pods vs one pod with read slots | One pod | same floor, 34% less RAM (3B) | |

Recompute it from the raw data (Python standard library only, under a second):

```bash
python3 scripts/reproduce_decision_table.py
python3 scripts/validate_output_identity.py      # 1,136 of 1,200 runs identical to baseline; the rest are bf16 near-ties
```

## Documents

| Document | What it is |
|---|---|
| [`docs/REPORT.md`](docs/REPORT.md) ([PDF](docs/REPORT.pdf)) | The report: decision rules and the evidence for them, 9 sections |
| [`docs/DECISION_RULES.md`](docs/DECISION_RULES.md) | One page for operators: floor, rules, what to avoid, Kubernetes recipes |
| [`docs/DETAILED_REPORT.md`](docs/DETAILED_REPORT.md) ([PDF](docs/DETAILED_REPORT.pdf)) | Every one of the 36 tests with its reasoning, table and chart |
| [`docs/FULL_RECORD.md`](docs/FULL_RECORD.md) | Lab record: decisions, timeline, every problem met |
| [`docs/archive/`](docs/archive/) | Earlier reports, superseded, kept for the record |

## Repository layout

```
docs/                 reports (Markdown + PDF)
charts/               figures used in the reports (charts/detailed/: one per test)
results/              raw data: one JSON line per run (see results/README.md)
src/engine/           streaming engine, coordinator, drafter, microbenchmarks, correctness checks, weight splitters
src/k8s/              manifest generator, Dockerfile, example manifests, how to run
scripts/              analysis: decision table, output identity, charts, detailed report
scripts/experiments/  the sweep scripts exactly as run for the data in results/
```

## Re-running the experiments

See [`src/k8s/README.md`](src/k8s/README.md). You need Docker, k3d, kubectl, a fast NVMe drive and the model weights from Hugging Face.

## Citation

See [`CITATION.cff`](CITATION.cff).

```
Bejaoui, A. (2026). Decision Rules for Streaming LLM Layers under Memory Pressure: Dense vs Mixture-of-Experts.
https://github.com/seneca-innovation-center/k8s-llm-offload-bench
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
