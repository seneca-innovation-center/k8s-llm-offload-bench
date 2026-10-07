"""Recompute the Executive Decision Table of docs/REPORT.md from the raw result files.

    python3 scripts/reproduce_decision_table.py

No dependencies beyond the Python standard library. Every number printed here is derived from results/*/ *.jsonl
(or the microbenchmark JSON files); the report quotes these numbers.

Conventions (same as the report):
  per token   median of the per-token decode times; for speculative runs, total decode time / tokens produced
  RAM         peak resident memory (VmHWM) of all stage pods + the coordinator, decimal GB
  ratio       per token / per token of the baseline (whole model resident) for the same model and prompt
  saved       1 - RAM / baseline RAM
"""
import json, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
DENSE = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b"]
MOE = ["gpt-oss-20b", "qwen3-30b-a3b"]
NAME = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B",
        "gpt-oss-20b": "gpt-oss-20b", "qwen3-30b-a3b": "Qwen3-30B-A3B"}
PROMPTS = ["prompt_long.txt", "prompts/chat_summary.txt", "prompts/chat_code.txt"]
# Bytes of the per-layer weight files (bf16), i.e. what one full streaming pass reads. Measured with `cat layer_*.safetensors | wc -c`.
LAYER_BYTES = {"qwen2.5-1.5b": 2620709952, "qwen2.5-3b": 5549584896, "qwen2.5-7b": 13051268832, "qwen2.5-14b": 26425842432,
               "gpt-oss-20b": 39513014592, "qwen3-30b-a3b": 59821589376}
DISK_GBPS = 7.1  # saturated bandwidth of the NVMe link, results/qwen2.5-3b/diskbw.json


def jl(path):
    return [json.loads(l) for l in open(path)]


def ram(r):
    s = r["stats"]
    return (sum(x["VmHWM"] for x in s["stages"]) + s["coordinator"]["VmHWM"]) / 1e9


def per_token(rs):
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


def first_ram(rs):
    return ram(next(r for r in rs if r["stats"]))


def rng(vals, fmt, unit=""):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return f"{lo}{unit}" if lo == hi else f"{lo}–{hi}{unit}"


def pct(vals):
    return rng([100 * v for v in vals], "{:.0f}", "%")


def x(vals):
    return rng(vals, "{:.1f}", "×")


# ------------------------------------------------------------------------------------------------ six-model sweep
def sweep(model):
    g = defaultdict(list)
    for r in jl(R("six-models", f"{model}.jsonl")):
        g[(r["approach"], r.get("MEM_LIMIT"), r["PROMPT_FILE"])].append(r)
    out = {}
    for (a, lim, pf), rs in g.items():
        b = g[("baseline", None, pf)]
        out[(a, lim, pf)] = dict(ratio=per_token(rs) / per_token(b), saved=1 - first_ram(rs) / first_ram(b), tpot=per_token(rs))
    return out


S = {m: sweep(m) for m in DENSE + MOE}


def over_prompts(m, a, lim=None):
    vals = [S[m][(a, lim, pf)] for pf in PROMPTS if (a, lim, pf) in S[m]]
    return [v["ratio"] for v in vals], [v["saved"] for v in vals]


def limits(m):
    return sorted({k[1] for k in S[m] if k[0] == "mmap"}, key=lambda l: -float(l.rstrip("GiM")) * (1024 if l.endswith("Gi") else 1))


# ------------------------------------------------------------------------------------------------ other experiments
def batching(path, G):
    g = defaultdict(list)
    for r in jl(path):
        if r.get("approach") == "batch":
            g[(r["G"], r["K"], int(r["BATCH"]))].append(r)
    base, cyc = per_token(g[(1, 1, 512)]), per_token(g[(G, 4, 512)])
    return cyc / base, base / cyc  # per-user ratio; throughput share = (512/cyc) / (512/base)


def partial_zero_saved(path, resident):
    rows = [r for r in jl(path) if int(r["n_prompt"]) == 512]
    base = [r for r in rows if r["approach"] == "baseline"]
    part = [r for r in rows if r["approach"] == "partial" and r.get("RESIDENT") == resident]
    return per_token(part) / per_token(base), 1 - first_ram(part) / first_ram(base)


def pods_vs_single_pod():
    e1 = [r for r in jl(R("qwen2.5-3b", "results.jsonl")) if int(r["n_prompt"]) == 512]
    pods = [r for r in e1 if r["G"] == 36 and r["K"] == 4 and r["mode"] == "cyclic"]
    one = [r for r in jl(R("qwen2.5-3b", "results3.jsonl")) if int(r["n_prompt"]) == 512 and r.get("SLOTS") == "4"]
    return (per_token(pods), first_ram(pods)), (per_token(one), first_ram(one))


def optimisation_round3():
    """Best configuration of the latency optimisation (65-node tree, lean drafter, 1 pod, 4 slots) per prompt."""
    rows = jl(R("qwen2.5-3b", "results6.jsonl"))
    out = {}
    for pf in ["prompt_long.txt", "prompts/chat_explain.txt", "prompts/chat_summary.txt", "prompts/chat_code.txt"]:
        t = [r for r in rows if r.get("PROMPT_FILE") == pf and r["approach"] == "tree" and r.get("TREE") == "8,16,16,12,8,4" and r.get("SLOTS") == "4"]
        b = [r for r in rows if r.get("PROMPT_FILE") == pf and r["approach"] == "baseline"]
        out[pf] = (per_token(t), per_token(b), first_ram(t))
    return out


# ------------------------------------------------------------------------------------------------ print
print("# Executive Decision Table, recomputed from results/\n")
dense_tree = [over_prompts(m, "tree") for m in DENSE[1:]]
tr_r, tr_s = [v for r, _ in dense_tree for v in r], [v for _, s in dense_tree for v in s]
r15, s15 = over_prompts("qwen2.5-1.5b", "tree")
dense_pg = [v for m in DENSE for l in limits(m) for v in over_prompts(m, "mmap", l)[0]]
b3, bg = batching(R("qwen2.5-3b", "results2.jsonl"), 36), batching(R("gpt-oss-20b", "results2.jsonl"), 24)
p3, pg = partial_zero_saved(R("qwen2.5-3b", "results2.jsonl"), "0-32"), partial_zero_saved(R("gpt-oss-20b", "results2.jsonl"), "0-21")
disk = json.load(open(R("qwen2.5-3b", "diskbw.json")))["aggregate_GBps_by_concurrency"]
(pt, pr), (ot, orr) = pods_vs_single_pod()
moe_rows = []
for m in MOE:
    for l in limits(m):
        r_, s_ = over_prompts(m, "mmap", l)
        moe_rows.append(f"{NAME[m]} {l.replace('Gi', ' GiB')}: {x(r_)} at {pct(s_)} saved")
moe_stream = [v for m in MOE for v in over_prompts(m, "seq1pod")[0]]

print("| Situation | Recommended approach | Cost vs resident baseline (per token) | Memory saved | Evidence |")
print("|---|---|---|---|---|")
print(f"| Dense model, one user, RAM tight | Single-pod streaming (4 read slots) + 64-node tree speculation | {x(tr_r)} (3B–14B); {x(r15)} at 1.5B | {pct(tr_s)} (3B–14B); {pct(s15)} at 1.5B | E1, D6 |")
print(f"| Dense model, many users | Streaming + batching (measured with 4-pod cyclic streaming, 3B) | {b3[0]:.2f}× per user at 512 users; {100 * b3[1]:.0f}% of baseline throughput | 4 of 36 layers in memory | B9 |")
print(f"| Mixture of experts, one user | Kernel paging: memory-mapped weights + pod memory limit | {'; '.join(moe_rows)} | see cost column | E1, E4, E5 |")
print(f"| Mixture of experts, many users | Not measured with paging; layer streaming + batching measured (gpt-oss, 4 pods) | {bg[0]:.2f}× per user at 512 users; {100 * bg[1]:.0f}% of throughput | 4 of 24 layers | C8 |")
print(f"| Dense model: kernel paging | Do not use | {rng(dense_pg, '{:.1f}', '×')} | — | E1, B7 |")
print(f"| Mixture of experts: layer streaming | Do not use | {x(moe_stream)} | — | E1, C3 |")
print(f"| Partial residency (pin most layers, stream a few) | Only for savings under ~10% | {p3[0]:.2f}× (3B) and {pg[0]:.1f}× (gpt-oss) with nothing saved | 0% | B6, C6 |")
print(f"| Several pods vs one pod with read slots | One pod | same floor: {pt:.0f} ms (4 pods) vs {ot:.0f} ms (1 pod, 4 slots), 3B | {pr:.2f} GB vs {orr:.2f} GB total RAM | B10, C5 |")

print("\n# Rule F: the disk floor\n")
print("| parallel readers | " + " | ".join(sorted(disk, key=int)) + " |")
print("|---|" + "---|" * len(disk))
print("| GB/s | " + " | ".join(f"{disk[k]:.2f}" for k in sorted(disk, key=int)) + " |\n")
print("| model | layer bytes | predicted floor (÷ 7.1 GB/s) | measured streaming, 3 prompts | measured ÷ predicted |")
print("|---|---|---|---|---|")
for m in DENSE + MOE:
    pred = LAYER_BYTES[m] / (DISK_GBPS * 1e9) * 1e3
    meas = [S[m][("seq1pod", None, pf)]["tpot"] for pf in PROMPTS]
    print(f"| {NAME[m]} | {LAYER_BYTES[m] / 1e9:.2f} GB | {pred:.0f} ms | {rng(meas, '{:.0f}', ' ms')} | {rng([v / pred for v in meas], '{:.2f}')} |")

print("\n# Rule D: dense models, per model (three prompts)\n")
print("| model | streaming, 4 slots | streaming + tree speculation | kernel paging (each limit) |")
print("|---|---|---|---|")
for m in DENSE:
    sr, ss = over_prompts(m, "seq1pod")
    tr, ts = over_prompts(m, "tree")
    pgs = "; ".join(f"{l.replace('Gi', ' GiB').replace('Mi', ' MiB')}: {x(over_prompts(m, 'mmap', l)[0])} at {pct(over_prompts(m, 'mmap', l)[1])}" for l in limits(m))
    print(f"| {NAME[m]} | {x(sr)} at {pct(ss)} saved | {x(tr)} at {pct(ts)} saved | {pgs} |")

print("\n# Rule M: mixtures of experts, per model (three prompts)\n")
print("| model | streaming, 4 slots | streaming + tree speculation | kernel paging (each limit) |")
print("|---|---|---|---|")
for m in MOE:
    sr, ss = over_prompts(m, "seq1pod")
    tr, ts = over_prompts(m, "tree")
    pgs = "; ".join(f"{l.replace('Gi', ' GiB')}: {x(over_prompts(m, 'mmap', l)[0])} at {pct(over_prompts(m, 'mmap', l)[1])}" for l in limits(m))
    print(f"| {NAME[m]} | {x(sr)} at {pct(ss)} saved | {f'{x(tr)} at {pct(ts)} saved' if tr else 'no drafter in the model family'} | {pgs} |")

print("\n# Dense optimisation result (Qwen2.5-3B, 65-node tree, lean drafter, 1 pod, 4 slots; round 3)\n")
print("| prompt | per token | same-round baseline | total RAM |")
print("|---|---|---|---|")
for pf, (t, b, rr) in optimisation_round3().items():
    print(f"| {pf.split('/')[-1]} | {t:.0f} ms | {b:.0f} ms | {rr:.2f} GB |")
