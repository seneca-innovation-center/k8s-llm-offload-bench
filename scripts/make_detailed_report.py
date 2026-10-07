"""Detailed test report: every test of the project with its reasoning, method, table, chart and interpretation.

    python3 scripts/make_detailed_report.py   # -> docs/DETAILED_REPORT.md, docs/DETAILED_REPORT.html, charts/detailed/*.png

Numbers come from two sources, stated per section:
  * computed here from the raw result files (results/*/ *.jsonl, *.json), or
  * transcribed from the verified tables of docs/archive/FINAL_REPORT.md / FINAL_REPORT_GPTOSS.md / docs/FULL_RECORD.md
    (themselves derived from the same raw files and independently re-checked).
Each section's table and chart are built from the same Python data, so they always agree.
"""
import base64, json, os, re, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
OUT = os.path.join(HERE, "charts", "detailed")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d94f9a", "#d64545", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

doc = []
md = doc.append


def style(ax, title=None, xl=None, yl=None, grid="y"):
    if title:
        ax.set_title(title, loc="left")
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    return f"../charts/detailed/{name}"


def img(path, caption):
    md(f"![{caption}]({path})\n\n*Figure: {caption}*\n")


def table(headers, rows):
    s = "| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |\n"
    md(s)


def annotate_bars(ax, bars, fmt="{:.0f}", dy=3, fs=7.5):
    for b in bars:
        h = b.get_height()
        if h and h == h:
            f = "{:.1f}" if fmt == "{:.0f}" and h < 10 else fmt
            ax.annotate(f.format(h), (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=fs, color=INK2)


def rng(lo, hi, fmt="{:.1f}", unit=""):
    a, b = fmt.format(lo), fmt.format(hi)
    return f"{a}{unit}" if a == b else f"{a}–{b}{unit}"


def jl(path):
    return [json.loads(l) for l in open(os.path.join(HERE, path))]


def first_tokens(t):
    return t[0] if t and isinstance(t[0], list) else t


def ram_of(r):
    s = r["stats"]
    return (sum(x["VmHWM"] for x in s["stages"]) + s["coordinator"]["VmHWM"]) / 1e9


def per_token(rs):
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


PL = {"prompt_long.txt": "prose 512", "prompts/chat_explain.txt": "chat: explain", "prompts/chat_summary.txt": "chat: summary",
      "prompts/chat_code.txt": "chat: code"}
PROMPTS4 = ["prompt_long.txt", "prompts/chat_explain.txt", "prompts/chat_summary.txt", "prompts/chat_code.txt"]

# ---------------------------------------------------------------------------------------------------------------- raw data
EXP1_3B, EXP1_GO = jl("results/qwen2.5-3b/results.jsonl"), jl("results/gpt-oss-20b/results.jsonl")
EXP2_3B, EXP3_3B, EXP2_GO = jl("results/qwen2.5-3b/results2.jsonl"), jl("results/qwen2.5-3b/results3.jsonl"), jl("results/gpt-oss-20b/results2.jsonl")
OPT = {n: jl(f"results/qwen2.5-3b/results{n}.jsonl") for n in (4, 5, 6, 7)}
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "gpt-oss-20b", "qwen3-30b-a3b"]
MNAME = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B",
         "gpt-oss-20b": "gpt-oss-20b", "qwen3-30b-a3b": "Qwen3-30B-A3B"}
SWEEP = {m: jl(f"results/six-models/{m}.jsonl") for m in MODELS}
REPL = jl("results/six-models/gpt-oss-20b_paging2.jsonl") + jl("results/six-models/qwen3-30b-a3b_paging2.jsonl")
RECHECK = jl("results/six-models/gptoss_paging_recheck.jsonl")
MB3, MBG = json.load(open(os.path.join(HERE, "results/qwen2.5-3b/microbench.json"))), json.load(open(os.path.join(HERE, "results/gpt-oss-20b/microbench.json")))
DISK = json.load(open(os.path.join(HERE, "results/qwen2.5-3b/diskbw.json")))["aggregate_GBps_by_concurrency"]
CONT1, CONT2 = json.load(open(os.path.join(HERE, "results/qwen2.5-3b/contention.json"))), json.load(open(os.path.join(HERE, "results/qwen2.5-3b/contention2.json")))
FILES = [("results/qwen2.5-3b/results.jsonl", "3B experiment 1: cyclic streaming", EXP1_3B), ("results/qwen2.5-3b/results2.jsonl", "3B experiment 2: alternatives", EXP2_3B),
         ("results/qwen2.5-3b/results3.jsonl", "3B single-pod streaming", EXP3_3B), ("results/gpt-oss-20b/results.jsonl", "gpt-oss experiment 1", EXP1_GO),
         ("results/gpt-oss-20b/results2.jsonl", "gpt-oss experiment 2", EXP2_GO), ("results/qwen2.5-3b/results4.jsonl", "optimisation round 1", OPT[4]),
         ("results/qwen2.5-3b/results5.jsonl", "optimisation round 2", OPT[5]), ("results/qwen2.5-3b/results6.jsonl", "optimisation round 3", OPT[6]),
         ("results/qwen2.5-3b/results7.jsonl", "optimisation round 4", OPT[7])] + \
        [(f"results/six-models/{m}.jsonl", f"six-model test: {MNAME[m]}", SWEEP[m]) for m in MODELS] + \
        [("results/six-models/*_paging2.jsonl", "six-model test: kernel-paging replication", REPL),
         ("results/six-models/gptoss_paging_recheck.jsonl", "six-model test: degenerate-prompt recheck", RECHECK)]
TOTAL_RUNS = sum(len(x[2]) for x in FILES)


def exp1(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["G"], r["K"], r["mode"], int(r["n_prompt"]))].append(r)
    return g


G3, GG = exp1(EXP1_3B), exp1(EXP1_GO)


def med(rs, key):
    return st.median([x for r in rs for x in r[key]]) if isinstance(rs[0][key], list) else st.median([r[key] for r in rs])


# ================================================================================================================ intro
md(f"""# Detailed test report: streaming a language model's layers from NVMe

24 September to 5 October 2026 · test machine **halo** (AMD Ryzen AI Max+ 395, 16 cores, 125 GB unified RAM, one NVMe at 7.1 GB/s) · models Qwen2.5 1.5B / 3B / 7B / 14B, gpt-oss-20b, Qwen3-30B-A3B · **{TOTAL_RUNS:,} measured runs** in {len(FILES)} data files.

This report lists every test run in the project, in the order it was run. For each test it gives the reason for running it, how it was run, a table, a chart, and what the result means. Where a later test corrected an earlier one, both are shown and the correction is marked.

## How to read the numbers

- **Baseline**: the whole model in memory in one pod. Every comparison uses the baseline measured in the same round.
- **Per token**: time to generate one token for one user (decode). For speculative runs: total decode time ÷ tokens produced.
- **First token**: time from receiving the prompt to the first generated token (prompt processing, "prefill").
- **RAM**: peak resident memory (high-water mark) of all pods together, coordinator included, unless the table says "stage pod". Decimal GB (10⁹ bytes).
- **vs baseline**: per token ÷ baseline per token. 1.0× = same speed, 8× = eight times slower.
- **Same output**: the generated tokens are compared one by one with the baseline's. Greedy decoding, bf16, no quantization anywhere.
- **Prompts**: 128 / 512 / 2048 tokens of a technical text (first experiments), a 512-token non-repeating prose text, and chat-style instructions (explain, summary, code) in the model's chat template.
- Setup: a one-node k3d Kubernetes cluster pinned to 16 cores; stage pods hold layers, a coordinator pod runs the tokenizer, embeddings, output layer and sampling. Weight files are evicted from the page cache around every read, so a reload is a real disk read.

## Overview: all tests
""")
OVERVIEW = [
    ("A1", "Correctness of our model code", "Does our layer code compute exactly what Hugging Face computes?", "checks", "Yes, to fp32 precision, for all four architectures"),
    ("A2", "Load vs compute per layer", "How long does a layer take to load from disk, and to compute?", "microbench", "Loading is 12× (3B) to 94× (gpt-oss) slower than computing one token"),
    ("A3", "Disk bandwidth vs readers", "How fast is the NVMe, and how many parallel reads does it need?", "microbench", "7.1 GB/s, reached with 4 readers; direct I/O no faster"),
    ("B1", "3B cyclic: memory", "How much memory does cyclic streaming save?", "219", "94% of weights with 2 pods; RAM breaks even at ~14 pods"),
    ("B2", "3B cyclic: first token", "Does prompt processing stay as fast?", "219", "Yes for long prompts (1.00× at 2048), 1.2–2× for short ones"),
    ("B3", "3B cyclic: time per token", "Does generation stay as fast?", "219", "No: 6–8× slower with any number of pods"),
    ("B4", "3B: why more pods don't help", "Why does adding pods not fix it?", "219", "Every pod adds a reader on one drive; floor 0.78 s per token"),
    ("B5", "3B: cost of splitting alone", "What does the pipeline cost without reloading?", "219", "About +30% per token"),
    ("B6", "3B: partial residency", "Pin most layers, stream a few?", "195", "1.33× before saving anything; 3.6× for 30% saved"),
    ("B7", "3B: kernel paging", "Let Linux page the weights under a memory limit?", "195", "Worst option: 7.7–10.9×"),
    ("B8", "3B: speculative decoding", "Get several tokens per disk read with a small draft model?", "195", "3.3× at 51% less RAM (was 7.6×)"),
    ("B9", "3B: batching", "Serve many users per disk read?", "195", "84% of baseline throughput at 512 users"),
    ("B10", "3B: single-pod streaming", "Does the cycle need pods at all?", "63", "No: one pod with 4 slots, same speed, less RAM"),
    ("B11", "3B: all approaches compared", "Which approach is best?", "—", "Speculative decoding (then); see D6 for the optimised result"),
    ("C1", "gpt-oss: memory", "Same idea on a 20B mixture-of-experts model", "192", "92% of weights saved with 2 pods"),
    ("C2", "gpt-oss: first token", "", "192", "1.44× even at 2048 tokens: loading slows compute 30–55%"),
    ("C3", "gpt-oss: time per token", "", "192", "33–44× slower; floor 5.55 s"),
    ("C4", "gpt-oss: cost of splitting", "", "192", "+20–25% per token"),
    ("C5", "gpt-oss: single-pod streaming", "", "183", "Same floor, less RAM"),
    ("C6", "gpt-oss: partial residency", "", "183", "3.9× before saving anything"),
    ("C7", "gpt-oss: kernel paging", "", "183 + 57", "Best option for experts; first figure (1.87×) was an artifact, real 7.4–8.0× at −54%"),
    ("C8", "gpt-oss: batching", "", "183", "1.78× per user at 512 users"),
    ("C9", "3B vs gpt-oss", "What changes with size and experts?", "—", "Everything scales with bytes read per token"),
    ("D1", "Optimisation round 1", "Move speculation to one pod", "33", "441 → 350 ms; direct-I/O rows invalid (artifact)"),
    ("D2", "Direct I/O and contention", "Can direct reads raise bandwidth or stop the compute slowdown?", "microbench", "No bandwidth gain; less slowdown for single tokens only"),
    ("D3", "Optimisation round 2", "Tree speculation instead of a chain", "72", "3.9–5.4 tokens per read; best tree 23–65 candidates"),
    ("D4", "Optimisation round 3", "Lean drafter in our own code", "72", "198–248 ms on chat prompts (1.7–2.2×)"),
    ("D5", "Optimisation round 4", "Direct I/O end to end, bigger and deeper trees", "75", "No gain; 65-node tree stays best"),
    ("D6", "Optimisation: net result", "", "—", "441 → 198–248 ms at 3.2–3.4 GB"),
    ("E1", "Six models: all approaches", "Is the best approach the same for every model?", "270", "Dense: tree speculation; experts: kernel paging"),
    ("E2", "Six models: the disk floor", "Does streaming sit exactly on bytes ÷ bandwidth?", "270", "Within 5% for all six"),
    ("E3", "Six models: first token", "What does each approach cost on the prompt?", "270", "Streaming 1.2–4.5×; paging up to 11× on experts"),
    ("E4", "Kernel-paging replication", "Was host page-cache pressure distorting paging?", "45", "No: same numbers within 10%"),
    ("E5", "Degenerate-prompt recheck", "Why was gpt-oss paging 1.87× before?", "12", "Repeated '…' output kept the same experts hot"),
    ("F1", "Output identity, all runs", "Is every output the same as the baseline?", "all", "All non-speculative runs; speculative except near-ties"),
    ("F2", "Near-ties in fp32", "Are the divergences bugs?", "8 positions", "No: top-2 gaps of 0.2–1.2%"),
    ("F3", "bf16 shape noise", "How big is bf16 rounding noise?", "40 positions", "Up to 0.28–0.42 logit points, more than every gap"),
]
table(["test", "name", "question", "runs", "result"], OVERVIEW)
md("Not a test, but part of the work: a literature check on 2 October (layer streaming, tree speculation and expert offloading are all known methods; the measurements on this hardware are new). It is summarised in `docs/FULL_RECORD.md`.\n")

# ================================================================================================================ PART A
md("# Part A. Groundwork: is the code right, and how fast is the hardware?\n")

# ---- A1
md("""## A1. Correctness of our model code

**Why.** Every later test streams our own implementation of the model's layers, not the reference library. If our code computed anything different, a speed or memory result would mean nothing. So each architecture was checked against Hugging Face `transformers` in full precision (fp32) before it was used.

**How.** Same weights, same input, both in fp32. We compare outputs layer by layer (or the final logits), plus the decode path (one token with a cache) against the full-prompt path.
""")
A1 = [
    ("Qwen2.5-3B (Qwen2), whole model, logits at 128 positions", "max |Δ| 0.0 on scale 27.7; top-1 identical at every position", "exact", 0.0),
    ("Qwen2.5-3B, our bf16 baseline vs HF bf16 `generate`, 128-token prompt", "same first 5 tokens, then a different token at 6", "bf16 rounding order (the fp32 check above is exact)", None),
    ("gpt-oss-20b, layer 0 (sliding window), fp32, 256 tokens", "max |Δ| 1.5e-5 on scale 103", "exact to fp32", 1.526e-5 / 103.1),
    ("gpt-oss-20b, layer 1 (full attention)", "1.7e-5 on scale 67", "exact to fp32", 1.717e-5 / 67.3),
    ("gpt-oss-20b, layer 22", "2.0e-3 on scale 76,293", "exact to fp32", 1.953e-3 / 76293.1),
    ("gpt-oss-20b, layer 23", "1.7e-2 on scale 189,899", "exact to fp32", 1.66e-2 / 189898.9),
    ("gpt-oss-20b, our bf16 baseline vs HF bf16 `generate`, 16 tokens", "identical tokens", "exact", None),
    ("Qwen3 dense (Qwen3-0.6B), layers vs transformers", "max |Δ| 3.8e-6 to 7.8e-3 on scales 6.8 to 11,297", "exact to fp32", 7.8e-3 / 11297),
    ("Qwen3-MoE router and experts, tiny random model", "max |Δ| 1.2e-7", "exact", None),
    ("Qwen3-30B-A3B, layer 0", "4.8e-7 on scale 2.5", "exact to fp32", 4.77e-7 / 2.5),
    ("Qwen3-30B-A3B, layer 24", "4.9e-6 on scale 5.1", "exact to fp32", 4.89e-6 / 5.1),
    ("Qwen3-30B-A3B, layer 47", "5.0e-5 on scale 104", "exact to fp32", 4.96e-5 / 104.1),
    ("Lean drafter (Qwen3-0.6B in our code) vs transformers, fp32", "last-position logits Δ = 0.0; greedy path inside the drafted tree", "exact", 0.0),
    ("Tree speculation (tree attention + cache compaction), fp32, local 0.5B", "same tokens as plain greedy decoding", "exact", None),
    ("Low-memory weight splitter vs original splitter", "byte-identical layer files", "exact", None),
    ("Eviction (3B): reload from disk vs page cache", "41 ms vs 19 ms; load counters = tokens × layer groups", "reloads are real disk reads", None),
]
table(["check", "result", "verdict"], [(a, b, c) for a, b, c, _ in A1])
fig, ax = plt.subplots(figsize=(8.6, 3.9))
pts = [(a, v) for a, b, c, v in A1 if v is not None]
labels = ["Qwen2.5-3B, whole model (logits)", "gpt-oss-20b, layer 0", "gpt-oss-20b, layer 1", "gpt-oss-20b, layer 22", "gpt-oss-20b, layer 23",
          "Qwen3-0.6B, worst layer", "Qwen3-30B-A3B, layer 0", "Qwen3-30B-A3B, layer 24", "Qwen3-30B-A3B, layer 47", "lean drafter (Qwen3-0.6B)"]
vals = [max(v, 1e-9) for _, v in pts]
cols = [AQUA if v == 0 else BLUE for _, v in pts]
y = range(len(pts))
ax.barh(list(y), vals, color=cols, height=0.6)
for i, (_, v) in enumerate(pts):
    ax.annotate("0 (exact)" if v == 0 else f"{v:.1e}", (max(v, 1e-9), i), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_xscale("log")
ax.set_xlim(1e-9, 1e-1)
ax.axvline(1.19e-7, color=GRAY, ls=":", lw=1)
ax.axvline(7.8e-3, color=ORANGE, ls="--", lw=1)
ax.text(1.19e-7, len(pts) - 0.4, " fp32 epsilon", fontsize=7.5, color=GRAY)
ax.text(7.8e-3, len(pts) - 0.4, " bf16 epsilon", fontsize=7.5, color=ORANGE)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
style(ax, "A1 · Difference to Hugging Face, relative to the output size (log scale)", "max |difference| ÷ output scale", grid="x")
img(save(fig, "A1_correctness.png"), "every numeric check sits at fp32 precision, 4–5 orders of magnitude below what bf16 can even represent")
md("""**What it means.** Our layer code is the reference computation, to fp32 precision, for Qwen2, gpt-oss, Qwen3 and Qwen3-MoE. The only "difference" ever seen is in bf16, where two correct implementations can round in a different order: the 3B's bf16 output differs from Hugging Face's own bf16 output at token 6, while the fp32 comparison is exact. That rounding effect comes back in F1–F3.
""")

# ---- A2
md("""## A2. How long a layer takes to load from disk, and to compute

**Why.** Cyclic streaming hides the loading of the next layer behind the computation of the other pods. That works only if loading is not much slower than computing. This microbenchmark measures both, one process alone, before any Kubernetes test.

**How.** `microbench.py`: load one layer file cold (evicted from the page cache) and warm (from the cache), and time one layer's forward pass for 1, 128, 512 and 2048 tokens. 16 threads, nothing else running.
""")
A2 = [
    ("layer size", "154 MB", "1,646 MB"),
    ("load from disk, one reader", f"{MB3['cold_load_ms_med']:.0f} ms ({MB3['cold_GBps']:.2f} GB/s)", f"{MBG['cold_load_ms_med']:.0f} ms ({MBG['cold_GBps']:.2f} GB/s)"),
    ("load from page cache", f"{MB3['warm_load_ms_med']:.0f} ms", f"{MBG['warm_load_ms_med']:.0f} ms"),
    ("compute, 1 token (generation)", f"{MB3['compute_ms_s1']:.1f} ms", f"{MBG['compute_ms_s1']:.1f} ms"),
    ("compute, 128 tokens", f"{MB3['compute_ms_s128']:.1f} ms", f"{MBG['compute_ms_s128']:.1f} ms"),
    ("compute, 512 tokens", f"{MB3['compute_ms_s512']:.1f} ms", f"{MBG['compute_ms_s512']:.1f} ms"),
    ("compute, 2048 tokens", f"{MB3['compute_ms_s2048']:.0f} ms", f"{MBG['compute_ms_s2048']:.0f} ms"),
    ("load ÷ compute for one token", f"{MB3['cold_load_ms_med'] / MB3['compute_ms_s1']:.0f}×", f"{MBG['cold_load_ms_med'] / MBG['compute_ms_s1']:.0f}×"),
    ("pods needed to hide one load (1 + load ÷ compute)", f"{MB3['K_needed_decode']:.0f}", f"{MBG['K_needed_decode']:.0f}"),
]
table(["measurement", "Qwen2.5-3B", "gpt-oss-20b"], A2)
fig, ax = plt.subplots(figsize=(8.6, 3.4))
cats = ["load\n(disk)", "load\n(cache)", "compute\n1 token", "compute\n128", "compute\n512", "compute\n2048"]
k3 = [MB3["cold_load_ms_med"], MB3["warm_load_ms_med"], MB3["compute_ms_s1"], MB3["compute_ms_s128"], MB3["compute_ms_s512"], MB3["compute_ms_s2048"]]
kg = [MBG["cold_load_ms_med"], MBG["warm_load_ms_med"], MBG["compute_ms_s1"], MBG["compute_ms_s128"], MBG["compute_ms_s512"], MBG["compute_ms_s2048"]]
x = range(len(cats))
b1 = ax.bar([i - 0.2 for i in x], k3, 0.38, color=BLUE, label="Qwen2.5-3B (154 MB layer)")
b2 = ax.bar([i + 0.2 for i in x], kg, 0.38, color=ORANGE, label="gpt-oss-20b (1.65 GB layer)")
annotate_bars(ax, b1, "{:.0f}")
annotate_bars(ax, b2, "{:.0f}")
ax.set_yscale("log")
ax.set_xticks(list(x))
ax.set_xticklabels(cats)
ax.legend(frameon=False)
style(ax, "A2 · One layer: load time vs compute time (ms, log scale)", yl="ms")
img(save(fig, "A2_microbench.png"), "loading one layer costs as much as computing it for hundreds of tokens")
md("""**What it means.** During generation a layer computes one token in 3.5 ms (3B) but takes 41 ms to load, so about 13 pods would be needed to hide each load behind the others' work, and 95 for gpt-oss. During a long prompt the picture flips for the 3B (166 ms of compute per layer against 41 ms of load), so loads can hide there. This table predicted the two main results before the Kubernetes tests: long prompts can keep their speed, generation cannot.
""")

# ---- A3
md("""## A3. Disk bandwidth and the number of parallel reads

**Why.** Each pod reads its next layer while the others compute, so the drive serves several reads at once. The total bandwidth the drive can deliver sets the floor for streaming. A second test checked whether direct reads (O_DIRECT, bypassing the page cache) are faster.

**How.** `diskbw.py`: 1 to 18 threads each reading different layer files, evicted from the cache. `iotest.py`: buffered vs O_DIRECT reads, checking that the page cache does not grow (so the numbers are real disk reads).
""")
readers = sorted(DISK, key=int)
table(["parallel readers", "total GB/s", "per reader"], [(n, f"{DISK[n]:.2f}", f"{DISK[n] / int(n):.2f}") for n in readers])
table(["read mode (iotest)", "1 reader", "4 readers", "page cache change"],
      [("buffered + evict", "3.71 GB/s", "6.54 GB/s", "≤ 0.02 GB"), ("O_DIRECT", "4.17 GB/s", "6.49 GB/s", "0.00 GB")])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
ax.plot([int(n) for n in readers], [DISK[n] for n in readers], "o-", color=BLUE)
for n in readers:
    ax.annotate(f"{DISK[n]:.1f}", (int(n), DISK[n]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.axhline(7.5, color=GRAY, ls=":", lw=1)
ax.text(18, 7.55, "PCIe 4.0 ×4 link ≈ 7.5", ha="right", fontsize=7.5, color=GRAY)
ax.set_ylim(0, 8.5)
ax.set_xticks([int(n) for n in readers])
style(ax, "A3 · Total disk bandwidth vs parallel readers", "parallel readers", "GB/s")
ax = axs[1]
bb = ax.bar([0, 1, 3, 4], [3.71, 6.54, 4.17, 6.49], color=[BLUE, BLUE, ORANGE, ORANGE], width=0.8)
annotate_bars(ax, bb, "{:.1f}")
ax.set_xticks([0, 1, 3, 4])
ax.set_xticklabels(["1", "4", "1", "4"])
ax.text(0.5, -1.6, "buffered", ha="center", fontsize=8, color=BLUE)
ax.text(3.5, -1.6, "O_DIRECT", ha="center", fontsize=8, color=ORANGE)
ax.set_ylim(0, 8.5)
style(ax, "Buffered vs direct reads", "readers", "GB/s")
img(save(fig, "A3_disk.png"), "the drive is capped by its PCIe 4.0 ×4 link at 7.1 GB/s; four readers reach it, direct reads do not go faster")
md("""**What it means.** One reader gets 4.1 GB/s, four get 7.0, eight or more 7.1: 95% of what the PCIe 4.0 ×4 link can carry. The drive itself is a PCIe 5.0 part, but the processor only offers PCIe 4.0, and there is no second slot. So the floor for any scheme that re-reads all layers per token is model bytes ÷ 7.1 GB/s: 0.78 s for the 3B, 5.55 s for gpt-oss. Nothing in software changes that number.
""")

# ================================================================================================================ PART B
md("# Part B. Qwen2.5-3B: cyclic streaming (the original idea) and four alternatives\n")
md("""Experiment 1 (24 Sep, 219 runs): baseline, 18 cyclic configurations (G groups of layers ∈ {36, 18, 12, 9, 6, 4}, K pods from 2 to 18), 3 resident controls, baseline again. Prompts of 128 / 512 / 2048 tokens, 16 generated tokens, 3 repetitions (baseline 5 at the start + 5 at the end). Pod *i* owns groups *i, i+K, i+2K…*; after computing a group it frees it and starts loading its next group from disk.
""")

# ---- B1
md("""## B1. Cyclic streaming: memory

**Why.** The point of the idea is less memory. This measures how much, counting not only weights but everything each pod needs to run.

**How.** One layer per group (G = 36), 128-token prompt, peak resident memory of the stage pods. Baseline stage pod: 5.55 GB of weights, 5.96 GB of RAM. Numbers from the verified table in `FULL_RECORD.md` §6.1.
""")
B1 = [(2, 0.31, 94, 0.99, 83), (3, 0.46, 92, 1.40, 76), (4, 0.62, 89, 1.81, 70), (6, 0.92, 83, 2.64, 56), (9, 1.39, 75, 3.86, 35), (12, 1.85, 67, 5.10, 14), (18, 2.77, 50, 7.57, -27)]
table(["pods", "layers in memory", "weights", "weights saved", "stage-pod RAM", "RAM saved"],
      [(k, k, f"{w:.2f} GB", f"{ws}%", f"{r:.2f} GB", f"{rs}%") for k, w, ws, r, rs in B1])
fig, ax = plt.subplots(figsize=(8.6, 3.2))
ks = [b[0] for b in B1]
ax.plot(ks, [b[3] for b in B1], "o-", color=BLUE, label="stage-pod RAM")
ax.plot(ks, [b[1] for b in B1], "s-", color=AQUA, label="weights in memory")
ax.axhline(5.96, color=INK2, ls="--", lw=1)
ax.text(18, 6.1, "baseline RAM 5.96 GB", ha="right", fontsize=7.5, color=INK2)
ax.axhline(5.55, color=AQUA, ls=":", lw=1)
ax.text(18, 5.2, "baseline weights 5.55 GB", ha="right", fontsize=7.5, color=AQUA)
for k, w, ws, r, rs in B1:
    ax.annotate(f"{rs}%", (k, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=BLUE)
ax.legend(frameon=False, loc="upper left")
style(ax, "B1 · Memory of the stage pods vs number of pods (labels: RAM saved)", "pods (one layer per pod in memory)", "GB")
img(save(fig, "B1_memory.png"), "weights fall to K/36 of the model; each pod adds ~250 MB of runtime, so RAM breaks even at about 14 pods")
md("""**What it means.** The idea delivers its memory promise: with 2 pods only 2 of 36 layers exist in memory (94% less weight memory, 83% less stage-pod RAM; 71% less total RAM including the coordinator). Each pod carries ~250 MB of Python/PyTorch runtime plus its share of the 151 MB attention cache, so adding pods eats the saving, and 18 pods use more RAM than the baseline. Long prompts add 0.8–0.9 GB of temporary activations per pod.
""")

# ---- B2
md("""## B2. Cyclic streaming: time to the first token

**Why.** The first goal was the same latency, both for reading the prompt and for generating. Prompt processing computes many tokens per layer, so the loads have more cover.

**How.** Median time to first token, best cyclic configuration per prompt length, against the baseline median. Computed here from the raw file `results/qwen2.5-3b/results.jsonl`.
""")
B2 = []
for n in (128, 512, 2048):
    base = [r["ttft_ms"] for r in G3[(1, 1, "resident", n)]]
    best = min(((med(rs, "ttft_ms"), k) for k, rs in G3.items() if k[2] == "cyclic" and k[3] == n), key=lambda x: x[0])
    B2.append((n, st.median(base), min(base), max(base), best[0], best[1][1], best[1][0]))
table(["prompt", "baseline (median; range)", "best cyclic", "ratio", "pods (groups)"],
      [(f"{n} tokens", f"{b / 1e3:.2f} s ({lo / 1e3:.2f}–{hi / 1e3:.2f})", f"{c / 1e3:.2f} s", f"{c / b:.2f}×", f"{k} ({G})") for n, b, lo, hi, c, k, G in B2])
fig, ax = plt.subplots(figsize=(8.6, 3.2))
for n, col in ((128, ORANGE), (512, BLUE), (2048, AQUA)):
    b = med(G3[(1, 1, "resident", n)], "ttft_ms")
    pts = sorted((k[1], med(rs, "ttft_ms") / b) for k, rs in G3.items() if k[2] == "cyclic" and k[3] == n and k[0] == 36)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=f"{n}-token prompt")
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.set_xticks([2, 3, 4, 6, 9, 12, 18])
ax.legend(frameon=False)
style(ax, "B2 · First token, cyclic ÷ baseline (one layer per pod)", "pods", "× baseline")
img(save(fig, "B2_first_token.png"), "with a 2048-token prompt the loads hide completely; short prompts don't give them enough cover")
md("""**What it means.** With a 2048-token prompt each layer computes for ~166 ms while a load takes ~41 ms, so the loads finish in the background: 2 pods reach the baseline (1.00×, within its own run-to-run range). With 512 tokens the best is 1.22×, with 128 tokens 1.99×. Prompt processing keeps its speed only when the prompt is long.
""")

# ---- B3
md("""## B3. Cyclic streaming: time per generated token

**Why.** This is the number a user feels while text appears. Each generated token passes through all 36 layers in order, so the cycle runs once per token.

**How.** Median time per token, 512-token prompt, one layer per pod, and where the time goes (from the stage-pod traces: waiting for weights, compute, pod-to-pod hops; the coordinator's 13–19 ms per token is not in the three columns). Verified table, `docs/FULL_RECORD.md` §6.3. Baseline 130 ms.
""")
B3 = [(2, 1013, 782, 188, 26), (3, 822, 555, 216, 30), (4, 781, 464, 269, 35), (6, 772, 437, 276, 38), (9, 770, 416, 283, 38), (12, 774, 409, 304, 39), (18, 769, 400, 308, 38)]
table(["pods", "per token", "vs baseline", "waiting for weights", "compute", "hops"],
      [(k, f"{t} ms", f"{t / 130:.1f}×", f"{w} ms", f"{c} ms", f"{h} ms") for k, t, w, c, h in B3])
fig, ax = plt.subplots(figsize=(8.6, 3.4))
x = range(len(B3))
w_ = [b[2] for b in B3]
c_ = [b[3] for b in B3]
h_ = [b[4] for b in B3]
ax.bar(x, w_, color=ORANGE, label="waiting for weights")
ax.bar(x, c_, bottom=w_, color=BLUE, label="compute")
ax.bar(x, h_, bottom=[a + b for a, b in zip(w_, c_)], color=GRAY, label="hops")
for i, b in enumerate(B3):
    ax.annotate(f"{b[1]} ms\n{b[1] / 130:.1f}×", (i, b[1]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.axhline(130, color=AQUA, ls="--", lw=1.2, label="baseline 130 ms")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{b[0]} pods" for b in B3])
ax.set_ylim(0, 1200)
ax.legend(frameon=False, loc="upper right", ncol=4)
style(ax, "B3 · Time per token and where it goes (512-token prompt)", yl="ms per token")
img(save(fig, "B3_per_token.png"), "most of every token is spent waiting for weights; the total stays at 0.77–1.01 s whatever the pod count")
md("""**What it means.** Generation is 6–8× slower than the baseline with any number of pods, and most of each token is spent waiting for a layer to arrive. With 128-token prompts it is 6.3–8.2×; with 2048 tokens 3.6–5.0× only because the baseline itself slows to 214 ms there. The idea does not keep generation latency on this machine. B4 explains why.
""")

# ---- B4
md("""## B4. Why more pods don't help

**Why.** The intuition was: with enough pods, the next layer is always loaded by the time the token comes back. A2 says ~13 pods should do it. The sweep went to 18 and the time stayed flat, so this test looks at why.

**How.** From the same runs: the time one layer takes to load as the pod count grows (traces), and the time per token for every grouping of layers (G groups, K pods), computed here from `results/qwen2.5-3b/results.jsonl`.
""")
LOAD3 = [(2, 49), (3, 62), (4, 78), (6, 119), (9, 179), (12, 231), (18, 318)]
table(["pods", "one layer load", "≈ readers in flight", "cover from the other pods' compute (1 token)"],
      [(k, f"{l} ms", k, f"≈ {(k - 1) * 7} ms") for k, l in LOAD3])
grp = defaultdict(dict)
for k, rs in G3.items():
    if k[2] == "cyclic" and k[3] == 512:
        grp[k[0]][k[1]] = med(rs, "tpot_ms")
table(["groups G (layers per group)"] + [f"K={k}" for k in (2, 3, 4, 6, 9, 12, 18)],
      [(f"{G} ({36 // G})",) + tuple(f"{grp[G][k]:.0f}" if k in grp[G] else "" for k in (2, 3, 4, 6, 9, 12, 18)) for G in sorted(grp, reverse=True)])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
ax = axs[0]
ax.plot([a for a, _ in LOAD3], [b for _, b in LOAD3], "o-", color=ORANGE, label="load of one layer")
ax.plot([a for a, _ in LOAD3], [(a - 1) * 7 for a, _ in LOAD3], "s--", color=BLUE, label="cover: others' compute")
ax.legend(frameon=False)
style(ax, "Load time grows with the pods", "pods", "ms")
ax = axs[1]
cols = {36: BLUE, 18: AQUA, 12: ORANGE, 9: MAGENTA, 6: YELLOW, 4: RED}
for G in sorted(grp, reverse=True):
    ks_ = sorted(grp[G])
    ax.plot(ks_, [grp[G][k] for k in ks_], "o-", color=cols[G], label=f"{36 // G} layer(s) per group", ms=4)
ax.axhline(780, color=INK2, ls=":", lw=1)
ax.text(18, 800, "disk floor 0.78 s", ha="right", fontsize=7.5, color=INK2)
ax.set_ylim(0, 1150)
ax.legend(frameon=False, fontsize=7)
style(ax, "Per token: only the pod count matters", "pods", "ms per token")
fig.suptitle("B4 · Why adding pods does not help (3B, 512-token prompt)", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "B4_why_pods.png"), "each pod adds a concurrent reader, so loads slow down as fast as the cover grows; grouping layers changes nothing")
md("""**What it means.** Three effects break the intuition:

1. **All pods share one drive.** Each pod starts loading as soon as it finishes computing, so about K reads are in flight and each one is slower: 49 ms with 2 pods, 318 ms with 18. A pod adds as much load time as it adds cover.
2. **The drive saturates.** Every token re-reads all 36 layers (5.55 GB). At 7.1 GB/s that is 0.78 s per token, reached from 4 pods on: the flat line.
3. **Loading slows computing.** Compute per layer rises from ~4 ms to 5–9 ms while other pods load (shared memory bandwidth). D2 measures this directly.

Grouping layers does not matter: every 2-pod configuration is ~1.0 s whatever the group size, every 3-pod one 0.82–0.86 s, everything from 4 pods on 0.77–0.79 s. Baseline speed by this route would need ~43 GB/s, six times this drive.
""")

# ---- B5
md("""## B5. The cost of splitting alone (resident controls)

**Why.** Part of the slowdown could come from the pipeline itself (sending activations between pods), not from reloading. These controls split the model over K pods but keep every layer in memory, so there is no reloading at all.

**How.** K pods, each permanently holding its 36/K layers, 512-token prompt. Computed from `results/qwen2.5-3b/results.jsonl`.
""")
B5 = [("baseline (1 pod)", med(G3[(1, 1, "resident", 512)], "tpot_ms"))] + \
     [(f"{k[1]} pods, all resident", med(rs, "tpot_ms")) for k, rs in sorted(G3.items(), key=lambda kv: kv[0][1]) if k[2] == "resident" and k[0] != 1 and k[3] == 512]
B5b = med(G3[(1, 1, "resident", 512)], "tpot_ms")
table(["configuration", "per token", "vs baseline"], [(a, f"{v:.0f} ms", f"{v / B5b:.2f}×") for a, v in B5] +
      [("4 pods, cyclic (for comparison)", f"{grp[36][4]:.0f} ms", f"{grp[36][4] / B5b:.2f}×")])
fig, ax = plt.subplots(figsize=(8.6, 2.9))
labs = [a for a, _ in B5] + ["4 pods, cyclic"]
vals = [v for _, v in B5] + [grp[36][4]]
bb = ax.bar(range(len(vals)), vals, color=[AQUA] + [BLUE] * (len(vals) - 2) + [ORANGE], width=0.6)
annotate_bars(ax, bb, "{:.0f} ms")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labs, fontsize=8)
style(ax, "B5 · Splitting without reloading costs ~30%; reloading costs the rest", yl="ms per token")
img(save(fig, "B5_controls.png"), "the pipeline itself adds ~40 ms per token; reloading adds ~600 ms")
md("""**What it means.** Splitting the model over pods with no reloading costs about 30% per token (hops of ~13 ms plus slightly slower compute per pod); first-token times stay within noise. So of the 781 ms per token with 4 cyclic pods, about 40 ms is the pipeline and about 610 ms is reloading.
""")

# ---- B6
md("""## B6. Partial residency: keep most layers, stream only a few

**Why.** If streaming all layers is too slow, stream just a few and pin the rest. The pinned layers' compute could hide the few loads.

**How.** Experiment 2 (27 Sep, 195 runs for B6–B9). One pod pins the first layers and streams the last ones through a few slots. 512-token prompt. Verified table, `docs/FULL_RECORD.md` §7.1. Baseline 133 ms.
""")
B6 = [("33 / 3 / 3", 0, 7.08, 0, 176, 1.33), ("30 / 6 / 3", 3, 6.60, 7, 235, 1.76), ("27 / 9 / 3", 6, 6.14, 13, 298, 2.24), ("24 / 12 / 4", 8, 5.84, 17, 339, 2.55),
      ("18 / 18 / 4", 14, 4.92, 30, 478, 3.59)]
table(["pinned / streamed / slots", "layers saved", "total RAM", "RAM saved", "per token", "vs baseline"],
      [(a, b, f"{c:.2f} GB", f"{d}%", f"{e} ms", f"{f:.2f}×") for a, b, c, d, e, f in B6])
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.plot([b[3] for b in B6], [b[5] for b in B6], "o-", color=BLUE)
for b in B6:
    ax.annotate(f"{b[0]}\n{b[5]:.2f}×", (b[3], b[5]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.set_ylim(0.8, 4.4)
ax.set_xlim(-3, 33)
style(ax, "B6 · Partial residency: RAM saved vs slowdown", "RAM saved (%)", "× baseline per token")
img(save(fig, "B6_partial.png"), "even saving nothing costs 1.33×; each layer saved costs about one extra disk read per token")
md("""**What it means.** Streaming 3 layers and saving none already costs 1.33×: the background reads slow the pinned layers' compute. Each layer saved beyond the slot count adds ~20–30 ms per token. Saving 30% of RAM costs 3.6×. This approach is only useful for small savings.
""")

# ---- B7
md("""## B7. Kernel paging: memory-mapped weights under a memory limit

**Why.** No loading code at all: map the weight files into memory and give the pod a Kubernetes memory limit. Linux then keeps what fits and re-reads the rest on demand. If this worked, it would be the simplest possible solution.

**How.** One pod, weights memory-mapped, limit from 6 GiB (everything fits) down to 1.5 GiB. The page cache is dropped after deploying, so pages are really charged to the pod (two earlier attempts measured the cache by mistake and were discarded, see Part G). Verified table, `docs/FULL_RECORD.md` §7.2.
""")
B7 = [("6 GiB (fits)", 7.07, 0, 131, 0.99, 0.91), ("4 GiB", 5.38, 24, 1029, 7.7, 1.60), ("3 GiB", 4.31, 39, 1399, 10.5, 1.90), ("2 GiB", 3.23, 54, 1413, 10.6, 1.88),
      ("1.5 GiB", 2.70, 62, 1455, 10.9, 1.95)]
table(["memory limit", "total RAM", "RAM saved", "per token", "vs baseline", "first token vs baseline"],
      [(a, f"{b:.2f} GB", f"{c}%", f"{d} ms", f"{e}×", f"{f}×") for a, b, c, d, e, f in B7])
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.plot([b[2] for b in B7], [b[4] for b in B7], "s-", color=YELLOW, label="kernel paging")
for b in B7:
    ax.annotate(b[0], (b[2], b[4]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.plot([54], [5.9], "o", color=BLUE)
ax.annotate("cyclic, 4 pods\n(same memory)", (54, 5.9), xytext=(8, -4), textcoords="offset points", fontsize=7.5, color=BLUE)
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.set_ylim(0, 12.5)
style(ax, "B7 · Kernel paging on the dense 3B: RAM saved vs slowdown", "RAM saved (%)", "× baseline per token")
img(save(fig, "B7_paging.png"), "as soon as the model does not fit, every token re-faults the whole model page by page")
md("""**What it means.** As soon as the limit is below the model size, each token touches every page, and the kernel re-reads the model in small pieces with one reader (~4 GB/s effective). That is 7.7–10.9× slower, and at equal memory the explicit 4-pod pipeline is 1.8× faster. For a dense model this is the worst option. Part C and E show it becomes the best option for mixture-of-experts models, for a structural reason.
""")

# ---- B8
md("""## B8. Speculative decoding on top of cyclic streaming

**Why.** The disk read is fixed per pass, so the only way to go faster is to get more than one token out of each pass. A small draft model (Qwen2.5-0.5B, 1 GB, kept in memory) guesses several tokens; the streamed 3B model checks all of them in one pass and keeps those it agrees with, plus one of its own. The output stays exactly the 3B model's.

**How.** Draft proposes γ tokens per cycle. Non-repeating 750-token text, 512-token prompt, 32 generated tokens. (A first run with the original, repetitive text gave 6.0 tokens per cycle because the model just continued the repetition; it was discarded, see Part G.) Verified table, `docs/FULL_RECORD.md` §7.3. Baseline 133 ms.
""")
B8 = [("all resident + draft, γ=5", 1.82, 2.8, 167, 137, 309, 120, 0.91, "8.35 GB (+18%)"), ("cyclic 2 pods + draft, γ=5", 1.82, 2.8, 194, 1008, 1209, 441, 3.3, "3.48 GB (−51%)"),
      ("cyclic 4 pods + draft, γ=5", 1.82, 2.8, 212, 778, 994, 365, 2.75, "4.47 GB (−37%)"), ("cyclic 4 pods + draft, γ=8", 2.10, 3.1, 321, 786, 1114, 371, 2.8, "4.53 GB (−36%)")]
table(["configuration", "drafts accepted per cycle", "tokens per cycle", "draft", "verify", "cycle", "per token", "vs baseline", "total RAM"],
      [(a, f"{b:.2f}", f"{c:.1f}", f"{d} ms", f"{e} ms", f"{f} ms", f"{g} ms", f"{h}×", i) for a, b, c, d, e, f, g, h, i in B8])
fig, ax = plt.subplots(figsize=(8.6, 3.1))
grpb = [("all resident", 133, 120), ("cyclic, 2 pods", 1013, 441), ("cyclic, 4 pods", 781, 365)]
x = range(len(grpb))
b1 = ax.bar([i - 0.2 for i in x], [g[1] for g in grpb], 0.38, color=BLUE, label="plain")
b2 = ax.bar([i + 0.2 for i in x], [g[2] for g in grpb], 0.38, color=AQUA, label="+ speculative (γ=5)")
annotate_bars(ax, b1, "{:.0f}")
annotate_bars(ax, b2, "{:.0f}")
ax.set_xticks(list(x))
ax.set_xticklabels([g[0] for g in grpb])
ax.legend(frameon=False)
style(ax, "B8 · Speculative decoding: ms per token, plain vs speculative", yl="ms per token")
img(save(fig, "B8_speculative.png"), "one disk-bound verify pass now yields 2.8 tokens, cutting the cyclic penalty from 7.6× to 3.3×")
md("""**What it means.** The verify pass costs about the same as a plain token (it is disk-bound), but now produces 2.8 tokens. The 2-pod cyclic penalty drops from 7.6× to 3.3× for 1 GB of extra memory, with identical output. More guesses (γ=8) do not help because acceptance drops. On the all-resident model it barely helps (0.91×) because the draft model is slow on CPU (35–40 ms per guess). This was the best single-user result of the first phase, and the starting point of the optimisation in Part D.
""")

# ---- B9
md("""## B9. Batching: many users per disk read

**Why.** One disk pass can advance many conversations by one token each. With enough users, the fixed disk cost per pass is shared.

**How.** B independent 128-token prompts generated in lockstep. Per-user latency = time per step; throughput = B tokens per step. Verified table, `docs/FULL_RECORD.md` §7.4; 2 repetitions (1 at 512).
""")
B9 = [(1, 122, 783, 8, 1.3), (16, 188, 795, 85, 20), (64, 541, 1015, 118, 63), (256, 1897, 2475, 135, 103), (512, 3418, 4058, 150, 126)]
table(["users", "baseline per user", "cyclic 4 pods per user", "ratio", "baseline tokens/s", "cyclic tokens/s", "cyclic share of throughput"],
      [(b, f"{p} ms", f"{c} ms", f"{c / p:.2f}×", t1, t2, f"{100 * t2 / t1:.0f}%") for b, p, c, t1, t2 in B9])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
bs = [b[0] for b in B9]
axs[0].plot(bs, [b[1] for b in B9], "o-", color=AQUA, label="baseline")
axs[0].plot(bs, [b[2] for b in B9], "o-", color=BLUE, label="cyclic, 4 pods")
axs[0].set_xscale("log", base=2)
axs[0].set_xticks(bs)
axs[0].set_xticklabels([str(b) for b in bs])
axs[0].minorticks_off()
axs[0].legend(frameon=False)
style(axs[0], "Per-user time per token", "concurrent users", "ms")
axs[1].plot(bs, [b[3] for b in B9], "o-", color=AQUA, label="baseline")
axs[1].plot(bs, [b[4] for b in B9], "o-", color=BLUE, label="cyclic, 4 pods")
for b in B9:
    axs[1].annotate(f"{100 * b[4] / b[3]:.0f}%", (b[0], b[4]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7.5, color=BLUE)
axs[1].set_xscale("log", base=2)
axs[1].set_xticks(bs)
axs[1].set_xticklabels([str(b) for b in bs])
axs[1].minorticks_off()
style(axs[1], "Throughput (labels: cyclic share)", "concurrent users", "tokens/s")
fig.suptitle("B9 · Batching closes the gap with many users", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "B9_batching.png"), "the disk cost per step is fixed while compute grows with the batch, so the pipeline catches up at hundreds of users")
md("""**What it means.** At 512 concurrent users the 4-pod pipeline delivers 84% of the baseline's throughput with 4 of 36 layers in memory, but each user still waits at least one full cycle (0.78 s) per token. Batching is a server-throughput solution, not a single-user latency one. The attention cache (37 KB per token per request) is the same on both sides and becomes the memory limit.
""")

# ---- B10
md("""## B10. Single-pod sequential streaming: the same cycle without pods

**Why.** B4 suggested that pods only matter as a way to have several reads in flight. This test removes the pods: one pod computes the 36 layers in order, frees layer *i* and starts loading layer *i+S*, so up to S layers are in memory.

**How.** One pod, S = 1 to 9 slots (up to 4 reads in flight), 512-token prompt. Verified table, `docs/archive/FINAL_REPORT.md` §5.5 (raw: `results/qwen2.5-3b/results3.jsonl`, 63 runs).
""")
B10 = [(1, 1.69, 76, 1664, 12.5, 2.03), (2, 1.85, 74, 1044, 7.85, 1.35), (3, 1.99, 72, 868, 6.5, 1.36), (4, 2.14, 70, 798, 6.0, 1.43), (6, 2.44, 66, 778, 5.85, 1.39), (9, 2.93, 59, 784, 5.9, 1.44)]
table(["slots", "total RAM", "RAM saved", "per token", "vs baseline", "first token vs baseline"],
      [(a, f"{b:.2f} GB", f"{c}%", f"{d} ms", f"{e}×", f"{f}×") for a, b, c, d, e, f in B10])
MULTI = [(2, 2.26, 1013), (3, 2.75, 822), (4, 3.25, 781), (6, 4.26, 772)]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
axs[0].plot([b[0] for b in B10], [b[3] for b in B10], "o-", color=AQUA, label="single pod, S slots")
axs[0].plot([m[0] for m in MULTI], [m[2] for m in MULTI], "s--", color=BLUE, label="K pods (B3)")
axs[0].axhline(780, color=INK2, ls=":", lw=1)
axs[0].legend(frameon=False)
style(axs[0], "Per token vs slots / pods", "slots or pods", "ms")
axs[1].plot([b[1] for b in B10], [b[3] for b in B10], "o-", color=AQUA, label="single pod")
axs[1].plot([m[1] for m in MULTI], [m[2] for m in MULTI], "s--", color=BLUE, label="K pods")
axs[1].legend(frameon=False)
style(axs[1], "Same speed, less RAM", "total RAM (GB)", "ms per token")
fig.suptitle("B10 · One pod with prefetch slots matches K pods", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "B10_single_pod.png"), "the curve is the same as for pods: it was always the number of reads in flight that mattered")
md("""**What it means.** One slot means no overlap: 36 × (41 ms load + 4 ms compute) ≈ 1.65 s. Each extra slot adds a reader until the drive saturates at 4, then the curve is flat at 0.78 s, exactly like the pod sweep. At equal speed the single pod uses less RAM (2.14 GB vs 3.25 GB at ~0.79 s) because it has one runtime and no traffic between pods. From here on, all streaming is done in one pod.
""")

# ---- B11
md("""## B11. All approaches compared (first phase)

**Why.** To pick the best approach per goal, all results are put on one scale: total RAM and time per token for one user.

**How.** 512-token prompt, total RAM including the coordinator, baseline 7.07 GB / 133 ms (experiment 2's baseline, so the cyclic ratios here are slightly lower than in B3, which used experiment 1's 130 ms). Verified table, `docs/FULL_RECORD.md` §8 (single-pod row from B10).
""")
B11 = [("baseline", "36 layers in one pod", 7.07, 133, BLUE), ("cyclic streaming", "2 pods", 2.26, 1013, BLUE), ("cyclic streaming", "4 pods", 3.25, 781, BLUE),
       ("cyclic streaming", "18 pods", 10.24, 769, BLUE), ("single-pod streaming", "4 slots", 2.14, 798, AQUA), ("partial residency", "−3 layers", 6.60, 235, ORANGE),
       ("partial residency", "−14 layers", 4.92, 478, ORANGE), ("kernel paging", "2 GiB limit", 3.23, 1413, YELLOW), ("cyclic + speculative", "2 pods, γ=5", 3.48, 441, MAGENTA),
       ("cyclic + speculative", "4 pods, γ=5", 4.47, 365, MAGENTA), ("speculative alone", "all resident + draft", 8.35, 120, MAGENTA)]
table(["approach", "configuration", "total RAM", "RAM saved", "per token", "vs baseline"],
      [(a, b, f"{c:.2f} GB", f"{100 * (1 - c / 7.07):.0f}%", f"{d} ms", f"{d / 133:.1f}×") for a, b, c, d, _ in B11])
fig, ax = plt.subplots(figsize=(8.6, 3.6))
for a, b, c, d, col in B11:
    ax.scatter(c, d / 133, color=col, s=45, zorder=3)
    ax.annotate(f"{a.replace('cyclic streaming', 'cyclic')}, {b}", (c, d / 133), xytext=(5, 2), textcoords="offset points", fontsize=7, color=INK2)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10])
ax.set_yticklabels(["1×", "2×", "5×", "10×"])
ax.minorticks_off()
ax.axhline(1, color=INK2, ls="--", lw=1)
style(ax, "B11 · Memory vs slowdown for every approach (down-left is better)", "total RAM (GB)", "× baseline per token", grid="both")
img(save(fig, "B11_frontier.png"), "speculative decoding on top of streaming was the best trade-off of the first phase")
md("""**What it means.** Memory: the idea works (68% less RAM with 2 pods). Latency: nothing that streams weights from this drive gets close to the baseline for a single user. The best trade-off was cyclic streaming + speculative decoding (51% less RAM at 3.3×). Kernel paging is the worst. Part D later improves the speculative approach to 1.7–2.2× at the same memory.
""")

# ================================================================================================================ PART C
md("# Part C. gpt-oss-20b: the same tests on a 20B mixture-of-experts model\n")
md("""gpt-oss-20b: 24 layers of 1.65 GB (39.5 GB of layer weights, bf16, no quantization), 32 experts per layer of which 4 are used per token. Experiment 1 (192 runs): G ∈ {24, 12, 8, 6, 4}, K from 2 to 12, resident controls, prompts of 128 / 512 / 2048 tokens, 16 tokens, 3 repetitions. Experiment 2 (183 runs): the alternatives. Coordinator ~2.9 GB in every configuration.
""")

# ---- C1
md("""## C1. Memory

**Why / how.** Same question as B1 for a model whose layers are ten times larger. One layer per pod, 128-token prompt, stage pods. Baseline stage pod 39.51 GB of weights, 40.08 GB of RAM. Verified table, `docs/archive/FINAL_REPORT_GPTOSS.md` §3.1.
""")
C1 = [(2, 3.29, 92, 4.18, 90), (3, 4.94, 88, 6.13, 85), (4, 6.59, 83, 8.07, 80), (6, 9.88, 75, 11.93, 70), (8, 13.17, 67, 15.76, 61), (12, 19.76, 50, 23.45, 41)]
table(["pods", "layers in memory", "weights", "weights saved", "stage-pod RAM", "RAM saved"], [(k, k, f"{w:.2f} GB", f"{ws}%", f"{r:.2f} GB", f"{rs}%") for k, w, ws, r, rs in C1])
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.plot([c[0] for c in C1], [c[3] for c in C1], "o-", color=ORANGE, label="stage-pod RAM")
ax.plot([c[0] for c in C1], [c[1] for c in C1], "s-", color=AQUA, label="weights in memory")
ax.axhline(40.08, color=INK2, ls="--", lw=1)
ax.text(12, 41, "baseline RAM 40.1 GB", ha="right", fontsize=7.5, color=INK2)
for c in C1:
    ax.annotate(f"{c[4]}%", (c[0], c[3]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=ORANGE)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
ax.set_ylim(0, 45)
style(ax, "C1 · gpt-oss memory vs pods (labels: RAM saved)", "pods", "GB")
img(save(fig, "C1_memory.png"), "with 1.65 GB layers the per-pod runtime is negligible: RAM saving tracks the weight saving")
md("""**What it means.** Memory works the same way and even better: the ~0.3 GB runtime per pod is small next to 1.65 GB layers, so 2 pods save 90% of stage-pod RAM and 12 pods still save 41%. At 2048-token prompts each pod adds ~2.6 GB of temporary expert activations.
""")

# ---- C2
md("""## C2. First token

**Why.** For the 3B, long prompts hid the loads completely. Does that still hold with layers ten times larger?

**How.** Best cyclic configuration per prompt length (computed from `results/gpt-oss-20b/results.jsonl`), and where the 2048-token prefill time goes (traces; verified table §3.2).
""")
C2 = []
for n in (128, 512, 2048):
    b = med(GG[(1, 1, "resident", n)], "ttft_ms")
    best = min(((med(rs, "ttft_ms"), k) for k, rs in GG.items() if k[2] == "cyclic" and k[3] == n), key=lambda x: x[0])
    C2.append((n, b, best[0], best[1][1]))
table(["prompt", "baseline", "best cyclic", "ratio", "pods"], [(f"{n} tokens", f"{b / 1e3:.2f} s", f"{c / 1e3:.2f} s", f"{c / b:.2f}×", k) for n, b, c, k in C2])
C2b = [("baseline", 9.3, 0, 9.29, 0, 387), ("resident control, 4 pods", 10.6, 0, 10.51, 0.09, 438), ("cyclic, 2 pods", 14.3, 1.89, 12.21, 0.17, 509),
       ("cyclic, 4 pods", 14.4, 0.12, 14.04, 0.18, 585), ("cyclic, 12 pods", 14.8, 0, 14.57, 0.19, 607)]
table(["2048-token prompt", "first token", "waiting for weights", "compute", "hops", "compute per layer"],
      [(a, f"{b} s", f"{c} s", f"{d} s", f"{e} s", f"{f} ms") for a, b, c, d, e, f in C2b])
fig, ax = plt.subplots(figsize=(8.6, 3.0))
x = range(len(C2b))
ax.bar(x, [c[3] for c in C2b], color=BLUE, label="compute")
ax.bar(x, [c[2] for c in C2b], bottom=[c[3] for c in C2b], color=ORANGE, label="waiting for weights")
ax.bar(x, [c[4] for c in C2b], bottom=[c[3] + c[2] for c in C2b], color=GRAY, label="hops")
for i, c in enumerate(C2b):
    ax.annotate(f"{c[5]} ms/layer", (i, c[1]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(list(x))
ax.set_xticklabels([c[0] for c in C2b], fontsize=8)
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, 17.5)
style(ax, "C2 · Where the 2048-token prefill time goes (labels: compute per layer)", yl="seconds")
img(save(fig, "C2_prefill.png"), "the loads are hidden from 3 pods on, but the compute itself slows by 30–55% while 1.65 GB layers stream in")
md("""**What it means.** Even at 2048 tokens the first token takes 1.44× the baseline. From 3 pods on the loads are hidden (waiting ≤ 0.24 s), but every layer's compute slows from 387 ms to 509–607 ms (+30–55%) because the disk transfers and the matrix multiplications share the same memory bandwidth on this chip. For the 3B the same effect was +8%. Short prompts are 2.3× (512) and 4.4× (128).
""")

# ---- C3
md("""## C3. Time per generated token

**Why / how.** Same as B3 and B4. 512-token prompt, one layer per pod; breakdown from the traces (verified table §3.3), per grouping computed from the raw file. Baseline 167 ms.
""")
C3 = [(2, 7087, 6762, 210, 81, 573), (3, 5772, 5240, 288, 113, 697), (4, 5636, 5231, 300, 126, 929), (6, 5499, 5005, 327, 123, 1356), (8, 5463, 5034, 357, 131, 1797), (12, 5533, 4917, 372, 128, 2647)]
table(["pods", "per token", "vs baseline", "waiting for weights", "compute", "hops", "one layer load"],
      [(k, f"{t} ms", f"{t / 167:.0f}×", f"{w} ms", f"{c} ms", f"{h} ms", f"{l} ms") for k, t, w, c, h, l in C3])
grpg = defaultdict(dict)
for k, rs in GG.items():
    if k[2] == "cyclic" and k[3] == 512:
        grpg[k[0]][k[1]] = med(rs, "tpot_ms")
table(["groups G (layers per group)"] + [f"K={k}" for k in (2, 3, 4, 6, 8, 12)],
      [(f"{G} ({24 // G})",) + tuple(f"{grpg[G][k]:.0f}" if k in grpg[G] else "" for k in (2, 3, 4, 6, 8, 12)) for G in sorted(grpg, reverse=True)])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax = axs[0]
x = range(len(C3))
ax.bar(x, [c[2] / 1e3 for c in C3], color=ORANGE, label="waiting")
ax.bar(x, [c[3] / 1e3 for c in C3], bottom=[c[2] / 1e3 for c in C3], color=BLUE, label="compute")
ax.bar(x, [c[4] / 1e3 for c in C3], bottom=[(c[2] + c[3]) / 1e3 for c in C3], color=GRAY, label="hops")
for i, c in enumerate(C3):
    ax.annotate(f"{c[1] / 167:.0f}×", (i, c[1] / 1e3), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(list(x))
ax.set_xticklabels([str(c[0]) for c in C3])
ax.legend(frameon=False, fontsize=7)
ax.set_ylim(0, 8)
style(ax, "Per token and where it goes", "pods", "seconds per token")
ax = axs[1]
colsg = {24: ORANGE, 12: AQUA, 8: BLUE, 6: MAGENTA, 4: YELLOW}
for G in sorted(grpg, reverse=True):
    ks_ = sorted(grpg[G])
    ax.plot(ks_, [grpg[G][k] / 1e3 for k in ks_], "o-", color=colsg[G], label=f"{24 // G} layer(s)/group", ms=4)
ax.axhline(5.55, color=INK2, ls=":", lw=1)
ax.text(12, 5.75, "floor 5.55 s", ha="right", fontsize=7.5, color=INK2)
ax.set_ylim(0, 8)
ax.legend(frameon=False, fontsize=7)
style(ax, "Only the pod count matters", "pods", "seconds per token")
fig.suptitle("C3 · gpt-oss: 33–42× slower, flat at the disk floor", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "C3_per_token.png"), "same shape as the 3B, at 39.5 GB ÷ 7.1 GB/s = 5.55 s per token")
md("""**What it means.** The same shape as the 3B, with much bigger numbers: 7.1 s with 2 pods, then flat at the 5.55 s floor (39.5 GB ÷ 7.1 GB/s), 33× the baseline. One layer's load grows from 573 ms to 2.6 s with the pod count. A fourth effect makes it worse than the 3B: **the scheme loads 8× more bytes than a token uses**, because which 4 of the 32 experts a token needs is only known when it reaches the layer, so whole layers must be prefetched. Baseline speed this way would need 236 GB/s.
""")

# ---- C4
md("""## C4. Cost of splitting alone

**Why / how.** Same as B5. K pods each holding 24/K layers permanently, 512-token prompt, from the raw file.
""")
C4b = med(GG[(1, 1, "resident", 512)], "tpot_ms")
C4 = [("baseline (1 pod)", C4b)] + [(f"{k[1]} pods, all resident", med(rs, "tpot_ms")) for k, rs in sorted(GG.items(), key=lambda kv: kv[0][1]) if k[2] == "resident" and k[0] != 1 and k[3] == 512]
table(["configuration", "per token", "vs baseline"], [(a, f"{v:.0f} ms", f"{v / C4b:.2f}×") for a, v in C4])
fig, ax = plt.subplots(figsize=(8.6, 2.6))
bb = ax.bar(range(len(C4)), [v for _, v in C4], color=[AQUA] + [ORANGE] * (len(C4) - 1), width=0.55)
annotate_bars(ax, bb, "{:.0f} ms")
ax.set_xticks(range(len(C4)))
ax.set_xticklabels([a for a, _ in C4])
ax.set_ylim(0, 250)
style(ax, "C4 · gpt-oss: splitting without reloading costs 20–25%", yl="ms per token")
img(save(fig, "C4_controls.png"), "the pipeline costs less than for the 3B because each layer computes longer relative to a hop")
md("""**What it means.** The pipeline itself costs 20–25% per token here, less than the 3B's 30%, because each gpt-oss layer computes for longer compared with the ~5 ms hop. Again, nearly all of the 33× comes from reloading.
""")

# ---- C5
md("""## C5. Single-pod sequential streaming

**Why / how.** Same as B10. One pod, S slots, 512-token prompt. Verified table, `docs/archive/FINAL_REPORT_GPTOSS.md` §6.1. Baseline 168 ms, 43.3 GB.
""")
C5 = [(1, 5.4, 87, 12.4, 73), (2, 7.3, 83, 7.18, 43), (3, 8.7, 80, 5.95, 35), (4, 10.5, 76, 5.56, 33), (6, 12.6, 71, 5.53, 33)]
table(["slots", "total RAM", "RAM saved", "per token", "vs baseline"], [(a, f"{b} GB", f"{c}%", f"{d} s", f"{e}×") for a, b, c, d, e in C5])
fig, ax = plt.subplots(figsize=(8.6, 2.8))
ax.plot([c[0] for c in C5], [c[3] for c in C5], "o-", color=AQUA, label="single pod, S slots")
ax.plot([c[0] for c in C3], [c[1] / 1e3 for c in C3], "s--", color=ORANGE, label="K pods (C3)")
ax.axhline(5.55, color=INK2, ls=":", lw=1)
ax.legend(frameon=False)
style(ax, "C5 · gpt-oss: one pod with slots reaches the same floor", "slots or pods", "seconds per token")
img(save(fig, "C5_single_pod.png"), "4 slots reach the 5.55 s floor with less RAM than 4 pods")
md("""**What it means.** Same conclusion as the 3B: 3–4 slots in one pod reach the 5.5 s floor with less RAM than K pods (10.5 GB with 4 slots vs 12.1 GB with 4 pods). With one slot, load and compute alternate with no overlap: 24 × (510 + 6) ms = 12.4 s.
""")

# ---- C6
md("""## C6. Partial residency

**Why / how.** Same as B6. Verified table, `docs/archive/FINAL_REPORT_GPTOSS.md` §6.2.
""")
C6 = [("22 / 2 / 2", 0, 43.3, 0, 648, 3.9), ("20 / 4 / 2", 2, 40.1, 8, 1229, 7.3), ("18 / 6 / 3", 3, 38.4, 11, 1472, 8.8), ("16 / 8 / 4", 4, 36.8, 15, 1851, 11.0), ("12 / 12 / 4", 8, 30.2, 30, 2755, 16.4)]
table(["pinned / streamed / slots", "layers saved", "total RAM", "RAM saved", "per token", "vs baseline"], [(a, b, f"{c} GB", f"{d}%", f"{e} ms", f"{f}×") for a, b, c, d, e, f in C6])
fig, ax = plt.subplots(figsize=(8.6, 2.8))
ax.plot([c[3] for c in C6], [c[5] for c in C6], "o-", color=ORANGE, label="gpt-oss-20b")
ax.plot([b[3] for b in B6], [b[5] for b in B6], "o--", color=BLUE, label="Qwen2.5-3B (B6)")
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.legend(frameon=False)
style(ax, "C6 · Partial residency: RAM saved vs slowdown", "RAM saved (%)", "× baseline per token")
img(save(fig, "C6_partial.png"), "with 1.65 GB layers there is nothing to hide the loads behind")
md("""**What it means.** Streaming 2 layers and saving none costs 3.9× (3B: 1.33×): re-reading 3.3 GB takes ~0.47 s while the 22 pinned layers compute for only 0.14 s. Saving 30% of RAM costs 16×. Not usable on this model.
""")

# ---- C7
md("""## C7. Kernel paging on a mixture of experts (with the correction)

**Why.** On the dense 3B, paging was the worst option because every token touches every page. A gpt-oss token touches only 4 of 32 experts per layer, so the kernel might keep the hot experts and read only what is used.

**How.** Same method as B7. First measured on 1–2 October with the original prompt (5 limits). Re-measured on 4–5 October in the six-model test with the real prompts (prose + 2 chat prompts, 32 tokens, 3 repetitions), then replicated with the host page cache cleared (E4). The cause of the difference is in E5.
""")
C7old = [("48 GiB (fits)", 10, 0.99), ("32 GiB", 15, 1.22), ("24 GiB", 35, 1.58), ("16 GiB", 54, 1.87), ("8 GiB", 74, 4.67)]
table(["limit", "RAM saved", "per token vs baseline (1–2 Oct, repetitive prompt)", "status"], [(a, f"{b}%", f"{c}×", "**invalid**: degenerate output, see E5") for a, b, c in C7old])


def sweep_groups(rows, extra=()):
    g = defaultdict(list)
    for r in rows:
        if r.get("recheck"):
            continue
        g[(r["approach"], r.get("MEM_LIMIT"), r["PROMPT_FILE"]) + tuple(r.get(e) for e in extra)].append(r)
    return g


GOS = sweep_groups(SWEEP["gpt-oss-20b"])
gob = {pf: per_token(GOS[("baseline", None, pf)]) for pf in PROMPTS4 if ("baseline", None, pf) in GOS}
C7new = []
for lim in ("26Gi", "16Gi"):
    vals = [per_token(GOS[("mmap", lim, pf)]) for pf in gob]
    ratios = [v / gob[pf] for v, pf in zip(vals, gob)]
    rams = [ram_of(next(r for r in GOS[("mmap", lim, pf)] if r["stats"])) for pf in gob]
    baseram = ram_of(next(r for r in GOS[("baseline", None, "prompt_long.txt")] if r["stats"]))
    C7new.append((lim.replace("Gi", " GiB"), rams[0], 100 * (1 - rams[0] / baseram), vals, ratios))
table(["limit", "total RAM", "RAM saved", "per token: prose / chat summary / chat code", "vs baseline (4–5 Oct, real prompts)"],
      [(a, f"{b:.1f} GB", f"{c:.0f}%", " / ".join(f"{v:.0f} ms" for v in d), rng(min(e), max(e), "{:.1f}", "×")) for a, b, c, d, e in C7new])
fig, ax = plt.subplots(figsize=(8.6, 3.1))
ax.plot([c[1] for c in C7old], [c[2] for c in C7old], "o--", color=GRAY, mfc="white", label="1–2 Oct, repetitive prompt (invalid)")
ax.plot([c[2] for c in C7new], [st.mean(c[4]) for c in C7new], "o-", color=ORANGE, label="4–5 Oct, real prompts (mean of 3)")
ax.errorbar([c[2] for c in C7new], [st.mean(c[4]) for c in C7new], yerr=[[st.mean(c[4]) - min(c[4]) for c in C7new], [max(c[4]) - st.mean(c[4]) for c in C7new]], fmt="none", ecolor=ORANGE, capsize=3)
ax.axhline(33, color=BLUE, ls=":", lw=1)
ax.text(75, 30, "layer streaming: 33×", ha="right", fontsize=7.5, color=BLUE)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 30])
ax.set_yticklabels(["1×", "2×", "5×", "10×", "30×"])
ax.minorticks_off()
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.legend(frameon=False, loc="upper left")
style(ax, "C7 · gpt-oss kernel paging: RAM saved vs slowdown", "RAM saved (%)", "× baseline per token (log)")
img(save(fig, "C7_paging.png"), "with real text, paging costs 3–4× at −30% and 7.4–8× at −54%; still 4–12× better than layer streaming")
md("""**What it means.** With real prompts, kernel paging costs 3.0–4.0× at 30% less RAM and 7.4–8.0× at 54% less. The first measurement (1.87× at 54% less) was an artifact: the old repetitive prompt made gpt-oss output "…" sixteen times in a row, which routes to the same experts at every step, so the cache always hit (E5 reproduces it). Even corrected, paging is 4–12× cheaper than any layer-streaming scheme on this model, because it reads only the experts a token touches. The first prompt pass pays 4–11×, because it touches almost every expert.
""")

# ---- C8
md("""## C8. Batching

**Why / how.** Same as B9. 128-token prompts in lockstep. Verified table, `docs/archive/FINAL_REPORT_GPTOSS.md` §6.4.
""")
C8 = [(1, 162, 5500, 6, 0.2), (16, 568, 5520, 28, 3), (64, 976, 5580, 66, 11), (256, 2370, 6000, 108, 43), (512, 3780, 6740, 135, 76)]
table(["users", "baseline per user", "cyclic 4 pods per user", "ratio", "baseline tokens/s", "cyclic tokens/s", "cyclic share"],
      [(b, f"{p} ms", f"{c / 1e3:.2f} s", f"{c / p:.0f}×" if c / p >= 10 else f"{c / p:.2f}×", t1, t2, f"{100 * t2 / t1:.0f}%") for b, p, c, t1, t2 in C8])
fig, ax = plt.subplots(figsize=(8.6, 2.8))
ax.plot([c[0] for c in C8], [c[2] / c[1] for c in C8], "o-", color=ORANGE, label="gpt-oss-20b")
ax.plot([b[0] for b in B9], [b[2] / b[1] for b in B9], "o--", color=BLUE, label="Qwen2.5-3B (B9)")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 16, 64, 256, 512])
ax.set_xticklabels(["1", "16", "64", "256", "512"])
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 30])
ax.set_yticklabels(["1×", "2×", "5×", "10×", "30×"])
ax.minorticks_off()
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.legend(frameon=False)
style(ax, "C8 · Per-user slowdown vs concurrent users", "concurrent users", "cyclic ÷ baseline")
img(save(fig, "C8_batching.png"), "the gap closes with users, from a much higher start: 1.78× at 512 users")
md("""**What it means.** The gap closes with more users but from 34×: at 512 users the pipeline is still 1.78× slower per user and delivers 56% of the baseline's throughput (3B: 1.19×, 84%).
""")

# ---- C9
md("""## C9. Small dense vs big mixture of experts, side by side

**Why.** To see what changes with model size and with experts, and why.

**How.** Key numbers from C1–C8 and B1–B10 (corrected paging figure). Verified table, `docs/archive/FINAL_REPORT_GPTOSS.md` §7.
""")
C9 = [("one layer: load ÷ compute (1 token)", "12×", "94×"), ("bytes loaded per token ÷ bytes used", "1", "8"), ("disk floor per token", "0.78 s", "5.55 s"),
      ("cyclic, 4+ pods", "6×", "33×"), ("first token, 2048-token prompt, best cyclic", "1.00×", "1.44×"), ("compute slowdown while loading (prefill)", "+8%", "+30–55%"),
      ("single pod, 4 slots: RAM saved · slowdown", "70% · 6.0×", "76% · 33×"), ("partial residency, nothing saved", "1.33×", "3.9×"),
      ("kernel paging at ~54% less RAM", "10.6×", "7.4–8.0×"), ("batching, 512 users, per user", "1.19×", "1.78×")]
table(["", "Qwen2.5-3B (dense)", "gpt-oss-20b (experts)"], C9)
fig, ax = plt.subplots(figsize=(8.6, 3.0))
labs = ["cyclic 4+ pods", "single pod 4 slots", "partial, 0 saved", "paging ~54%", "batching 512"]
v3 = [6, 6.0, 1.33, 10.6, 1.19]
vg = [33, 33, 3.9, 7.7, 1.78]
x = range(len(labs))
b1 = ax.bar([i - 0.2 for i in x], v3, 0.38, color=BLUE, label="Qwen2.5-3B")
b2 = ax.bar([i + 0.2 for i in x], vg, 0.38, color=ORANGE, label="gpt-oss-20b")
annotate_bars(ax, b1, "{:.1f}×")
annotate_bars(ax, b2, "{:.1f}×")
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 30])
ax.set_yticklabels(["1×", "2×", "5×", "10×", "30×"])
ax.minorticks_off()
ax.set_ylim(0.8, 60)
ax.set_xticks(list(x))
ax.set_xticklabels(labs)
ax.legend(frameon=False)
style(ax, "C9 · Slowdown vs baseline per approach (log)", yl="× baseline")
img(save(fig, "C9_compare.png"), "every streaming approach gets worse with layer size; only paging improves, because experts are touched sparsely")
md("""**What it means.** Memory savings scale in favour of the big model (the runtime per pod becomes negligible). Everything about latency gets worse in proportion to bytes read per token. The mixture of experts makes layer streaming worse still (8× more bytes than used) but makes kernel paging relatively better, because only touched experts are read. The six-model test (Part E) checks whether this is a rule.
""")

# ================================================================================================================ PART D
md("# Part D. Latency optimisation of the 3B at the same memory (≤ 3.5 GB)\n")
md("""Goal (2 October, decisions taken at the start of the round): keep the ~3.5 GB of the best speculative configuration (B8) and cut the time per token, with no quantization, a bf16 0.5B draft model, and exact output as far as bf16 allows. Prompts: the 512-token prose text and three chat instructions (explain, summary, code). Four rounds; rounds 1–3 and 4 ran at different machine loads, so only comparisons within a round are used. Tables in this part are computed here from the raw files.
""")


def opt_groups(rows):
    KEYS = ("approach", "G", "K", "SLOTS", "LOAD_MODE", "GAMMA", "TREE", "RESIDENT", "PROMPT_FILE", "n_prompt")
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in KEYS)].append(r)
    out = defaultdict(list)
    for k, rs in g.items():
        d = dict(zip(KEYS, k))
        pf = d["PROMPT_FILE"] or "prompt_long.txt"
        where = "all resident" if d["G"] == 1 else (f"1 pod, {d['SLOTS']} slots" if d["K"] == 1 else f"{d['K']} pods")
        if d.get("RESIDENT"):
            where = f"pinned {d['RESIDENT']}, {d['SLOTS']} slots"
        io = " · direct I/O" if d.get("LOAD_MODE") == "direct" else ""
        a = d["approach"]
        nodes = None
        if a == "spec":
            lab, nodes = f"chain γ={d['GAMMA']} · {where}{io}", int(d["GAMMA"]) + 1
        elif a == "tree":
            nodes = 1 + sum(int(w) for w in d["TREE"].split(","))
            lab = f"tree {nodes} nodes ({d['TREE']}) · {where}{io}"
        elif a == "seq1pod":
            lab = f"plain streaming · {where}{io}"
        else:
            lab = f"{a} · {where}{io}"
        rec = dict(label=lab, approach=a, io=bool(io), nodes=nodes, tree=d["TREE"], slots=d["SLOTS"], tpot=per_token(rs), ttft=med(rs, "ttft_ms"),
                   ram=ram_of(next(r for r in rs if r["stats"])), toks=[first_tokens(r["tokens"]) for r in rs])
        if rs[0].get("spec"):
            sp = [r["spec"] for r in rs]
            acc = [x for s in sp for x in s["accepted"]]
            rec.update(tpc=st.mean(acc) + 1, draft=st.median([x for s in sp for x in s["draft_ms"]]), verify=st.median([x for s in sp for x in s["verify_ms"]]))
        out[pf].append(rec)
    for pf, recs in out.items():
        base = next(r for r in recs if r["approach"] == "baseline")
        for r in recs:
            r["ratio"] = r["tpot"] / base["tpot"]
            bt = base["toks"][0]
            r["same"] = sum(1 for t in r["toks"] if t[:len(bt)] == bt[:len(t)])
            r["n"] = len(r["toks"])
        recs.sort(key=lambda r: (r["approach"] != "baseline", r["tpot"]))
    return out


O = {n: opt_groups(OPT[n]) for n in OPT}


def opt_table(recs, invalid=lambda r: False):
    rows = []
    for r in recs:
        rows.append((r["label"] + (" **(invalid)**" if invalid(r) else ""), f"{r['ram']:.2f} GB", f"{r['tpot']:.0f} ms", f"{r['ratio']:.2f}×",
                     f"{r['tpc']:.2f}" if "tpc" in r else "", f"{r['draft']:.0f}" if "draft" in r else "", f"{r['verify']:.0f}" if "verify" in r else "",
                     "yes" if r["same"] == r["n"] else f"{r['same']}/{r['n']}"))
    table(["configuration", "total RAM", "per token", "vs baseline", "tokens per read", "draft ms", "verify ms", "same output"], rows)


# ---- D1
md("""## D1. Round 1: speculation in one pod (and a direct-I/O artifact)

**Why.** B10 showed one pod with 4 slots streams as fast as many pods with less memory. Moving the speculative chain (B8) into one pod should remove the hops. Direct I/O (reads that bypass the page cache) was also tried as a way to load faster.

**How.** 512-token prose prompt, 32 generated tokens, 3 repetitions (`results/qwen2.5-3b/results4.jsonl`).
""")
inv = lambda r: r["io"]
opt_table(O[4]["prompt_long.txt"], inv)
fig, ax = plt.subplots(figsize=(8.6, 3.6))
recs = [r for r in O[4]["prompt_long.txt"]]
cols = [AQUA if r["approach"] == "baseline" else (GRAY if r["io"] else (MAGENTA if r["approach"] == "spec" else BLUE)) for r in recs]
bb = ax.barh(range(len(recs)), [r["tpot"] for r in recs], color=cols, hatch=None)
for b, r in zip(bb, recs):
    if r["io"]:
        b.set_hatch("///")
        b.set_edgecolor(SURF)
    ax.annotate(f"{r['tpot']:.0f} ms" + ("  invalid" if r["io"] else ""), (r["tpot"], b.get_y() + b.get_height() / 2), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(recs)))
ax.set_yticklabels([r["label"].replace(" · direct I/O", " · direct") for r in recs], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(0, 1000)
style(ax, "D1 · Round 1 (prose): ms per token; hatched = direct-I/O rows, invalid", "ms per token", grid="x")
img(save(fig, "D1_round1.png"), "the single-pod chain cuts 428 → 350 ms; the impressive direct-I/O rows were served partly from the page cache")
md("""**What it means.** Valid result: the speculative chain in one pod with 4 slots is 350 ms per token (2.7×) against 428 ms with 2 pods, because the verify pass drops from 1004 to 776 ms (four reads in flight, no hops). The direct-I/O rows looked spectacular (plain streaming at 483–505 ms, below the 0.78 s disk floor) but were impossible: on halo's btrfs filesystem, part of each direct read silently falls back to buffered reads, and 48% of the weight files were found in the page cache right after those runs. They are invalid and were re-measured properly in D2 and D5.
""")

# ---- D2
md("""## D2. Direct I/O and memory-bandwidth contention

**Why.** Two questions: can direct reads raise the bandwidth, and does loading slow the computation (as B4 and C2 suggested) and can direct reads avoid that?

**How.** `contention.py`: compute one layer for 1 and 512 tokens while 4 threads load other layers, buffered vs direct, and measure bandwidth with 1/2/4 readers. Run once with the buggy loader (artifact) and again with an end-of-file-safe loader that evicts any fallback pages, verified to leave the page cache unchanged.
""")
table(["measurement", "buffered", "direct (fixed loader)", "direct (first run, invalid)"],
      [("bandwidth, 1 reader", f"{CONT2['bw_GBps_buffered_1readers']} GB/s", f"{CONT2['bw_GBps_direct_1readers']} GB/s", f"{CONT1['bw_GBps_direct_1readers']} GB/s"),
       ("bandwidth, 2 readers", f"{CONT2['bw_GBps_buffered_2readers']} GB/s", f"{CONT2['bw_GBps_direct_2readers']} GB/s", f"{CONT1['bw_GBps_direct_2readers']} GB/s"),
       ("bandwidth, 4 readers", f"{CONT2['bw_GBps_buffered_4readers']} GB/s", f"{CONT2['bw_GBps_direct_4readers']} GB/s", f"{CONT1['bw_GBps_direct_4readers']} GB/s"),
       ("compute slowdown during loads, 1 token", f"{CONT2['slowdown_s1_buffered']}×", f"{CONT2['slowdown_s1_direct']}×", f"{CONT1['slowdown_s1_direct']}×"),
       ("compute slowdown during loads, 512 tokens", f"{CONT2['slowdown_s512_buffered']}×", f"{CONT2['slowdown_s512_direct']}×", f"{CONT1['slowdown_s512_direct']}×")])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0))
rr = [1, 2, 4]
axs[0].plot(rr, [CONT2[f"bw_GBps_buffered_{n}readers"] for n in rr], "o-", color=BLUE, label="buffered")
axs[0].plot(rr, [CONT2[f"bw_GBps_direct_{n}readers"] for n in rr], "o-", color=ORANGE, label="direct (fixed)")
axs[0].plot(rr, [CONT1[f"bw_GBps_direct_{n}readers"] for n in rr], "o--", color=GRAY, mfc="white", label="direct, first run (invalid)")
axs[0].axhline(7.5, color=GRAY, ls=":", lw=1)
axs[0].legend(frameon=False, fontsize=7)
axs[0].set_xticks(rr)
style(axs[0], "Bandwidth (above 7.5 = impossible)", "readers", "GB/s")
x = range(2)
b1 = axs[1].bar([i - 0.2 for i in x], [CONT2["slowdown_s1_buffered"], CONT2["slowdown_s512_buffered"]], 0.38, color=BLUE, label="buffered")
b2 = axs[1].bar([i + 0.2 for i in x], [CONT2["slowdown_s1_direct"], CONT2["slowdown_s512_direct"]], 0.38, color=ORANGE, label="direct (fixed)")
annotate_bars(axs[1], b1, "{:.2f}×")
annotate_bars(axs[1], b2, "{:.2f}×")
axs[1].axhline(1, color=INK2, ls="--", lw=1)
axs[1].set_xticks(list(x))
axs[1].set_xticklabels(["1 token", "512 tokens"])
axs[1].set_ylim(0, 2.1)
axs[1].legend(frameon=False, fontsize=7)
style(axs[1], "Compute slowdown while loading", yl="× idle compute time")
fig.suptitle("D2 · Direct I/O: no bandwidth gain; less slowdown for single tokens only", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "D2_direct_io.png"), "11 GB/s from a 7.5 GB/s link was the tell-tale sign of the artifact")
md("""**What it means.** Loading really does slow computation: one-token compute takes 1.44× longer and 512-token compute 1.72× longer while layers stream in, confirming B4 and C2. Direct reads do not raise bandwidth (6.8 vs 7.1 GB/s; the link is the limit). They reduce the slowdown for one-token work (1.18× vs 1.44×) but not for prompt-sized work (1.65× vs 1.72×). The first run reported 11 GB/s through a 7.5 GB/s link, which is how the artifact was caught.
""")

# ---- D3
md("""## D3. Round 2: tree speculation instead of a chain

**Why.** A chain of 5 guesses is wasted after the first wrong one. Because the verify pass costs the same whether it checks 5 or 65 candidates (it is disk-bound), the draft can propose a tree of alternatives, and the big model picks the longest correct branch in one pass (the SpecExec / Sequoia idea). Implemented with tree attention masks and cache compaction, proven exact in fp32 (A1).

**How.** Single pod, 4 slots, Hugging Face drafter. Trees of 23, 65, 129 and 201 nodes and the chain, on all four prompts, 48 tokens, 3 repetitions (`results/qwen2.5-3b/results5.jsonl`).
""")
for pf in PROMPTS4:
    md(f"**{PL[pf]}**\n")
    opt_table(O[5][pf])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
pcol = {"prompt_long.txt": BLUE, "prompts/chat_explain.txt": ORANGE, "prompts/chat_summary.txt": AQUA, "prompts/chat_code.txt": MAGENTA}
for pf in PROMPTS4:
    rs = sorted([r for r in O[5][pf] if r["nodes"]], key=lambda r: r["nodes"])
    axs[0].plot([r["nodes"] for r in rs], [r["tpot"] for r in rs], "o-", color=pcol[pf], label=PL[pf], ms=4)
    axs[1].plot([r["nodes"] for r in rs], [r["tpc"] for r in rs], "o-", color=pcol[pf], ms=4)
for a in axs:
    a.set_xscale("log")
    a.set_xticks([6, 23, 65, 129, 201])
    a.set_xticklabels(["chain 6", "23", "65", "129", "201"])
    a.minorticks_off()
axs[0].legend(frameon=False, fontsize=7)
style(axs[0], "ms per token", "candidates verified per pass", "ms")
style(axs[1], "tokens produced per disk read", "candidates verified per pass", "tokens")
fig.suptitle("D3 · Round 2: trees beat the chain; beyond 65 nodes the drafting cost wins", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "D3_round2.png"), "tokens per read keep rising with tree size, but drafting a big tree with the HF model costs more than it saves")
md("""**What it means.** Trees of 23–65 candidates beat the 5-token chain on every prompt: tokens per disk read rise from 2.4–3.8 to 3.9–5.4, giving 324 ms on prose and 217–248 ms on chat prompts. Bigger trees produce even more tokens per read (up to 6.9), but drafting them with the Hugging Face model took 0.4–1.5 s per cycle and the verify pass rose to 1.0–1.3 s, so they lost. The drafter became the bottleneck, which round 3 attacks. Two configurations differ from the baseline on the summary prompt, at one near-tie token (F2).
""")

# ---- D4
md("""## D4. Round 3: a lean drafter in our own layer code

**Why.** Drafting a 65-node tree with Hugging Face cost 290–620 ms per cycle. Running the 0.5B draft through our own layer code, with tree attention and the same cache compaction as the big model, means one forward pass per tree level and no cache copies.

**How.** Trees of 20–65 nodes (including a deep, narrow 20-node shape) at 4 slots, and the 65-node tree at 5 slots, on all four prompts (`results/qwen2.5-3b/results6.jsonl`). The prose baseline of this round ran while other users loaded the machine (173 ms instead of 126–129); chat baselines were normal.
""")
for pf in PROMPTS4:
    md(f"**{PL[pf]}**\n")
    opt_table(O[6][pf])
fig, ax = plt.subplots(figsize=(8.6, 3.3))
shapes = sorted({(r["nodes"], r["tree"], r["slots"]) for pf in PROMPTS4 for r in O[6][pf] if r["nodes"]}, key=lambda s: (s[0], s[2]))
w = 0.8 / len(shapes)
scol = [BLUE, ORANGE, AQUA, MAGENTA, YELLOW, RED]
for j, s in enumerate(shapes):
    vals = [next((r["tpot"] for r in O[6][pf] if (r["nodes"], r["tree"], r["slots"]) == s), float("nan")) for pf in PROMPTS4]
    ax.bar([i + (j - (len(shapes) - 1) / 2) * w for i in range(4)], vals, w * 0.95, color=scol[j % len(scol)], label=f"{s[0]} nodes ({s[1]}), {s[2]} slots")
bases = [next(r["tpot"] for r in O[6][pf] if r["approach"] == "baseline") for pf in PROMPTS4]
for i, b in enumerate(bases):
    ax.plot([i - 0.42, i + 0.42], [b, b], color=INK2, ls="--", lw=1)
ax.set_xticks(range(4))
ax.set_xticklabels([PL[p] for p in PROMPTS4])
ax.legend(frameon=False, fontsize=6.5, ncol=2)
ax.set_ylim(0, 380)
style(ax, "D4 · Round 3: ms per token by tree shape (dashed: same-round baseline)", yl="ms per token")
img(save(fig, "D4_round3.png"), "with the lean drafter the 65-node tree is best or tied on every prompt")
md("""**What it means.** Drafting a 65-node tree dropped from 290–620 ms to 210–240 ms per cycle, making the 65-node tree the best or tied shape on every prompt: 242 ms on prose, 198–248 ms on the chat prompts (1.7–2.2× the normal baselines) at 3.2–3.4 GB. A cycle is now ~80% verify pass (~860 ms: the 0.78 s disk floor plus compute). Five slots help a little on chat prompts and hurt on prose. Twelve of 60 runs differ from the baseline, all at the same two near-tie tokens (F2).
""")

# ---- D5
md("""## D5. Round 4: direct I/O end to end, a bigger tree, a deeper tree

**Why.** Check, end to end, the three remaining ideas: direct I/O with the fixed loader (D2 showed less compute slowdown), a 129-node tree with the lean drafter, and the deep 20-node tree.

**How.** Same configurations with and without direct I/O, back to back (`results/qwen2.5-3b/results7.jsonl`). This round ran while other users loaded halo (baselines 143–172 ms), so only comparisons inside the round are valid.
""")
for pf in PROMPTS4:
    md(f"**{PL[pf]}**\n")
    opt_table(O[7][pf])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
pairs = []
for pf in PROMPTS4:
    a = next((r for r in O[7][pf] if r["nodes"] == 65 and not r["io"]), None)
    b = next((r for r in O[7][pf] if r["nodes"] == 65 and r["io"]), None)
    if a and b:
        pairs.append((PL[pf], a["tpot"], b["tpot"]))
pa = next(r for r in O[7]["prompt_long.txt"] if r["approach"] == "seq1pod" and not r["io"])
pb = next(r for r in O[7]["prompt_long.txt"] if r["approach"] == "seq1pod" and r["io"])
pairs.append(("plain streaming, prose", pa["tpot"], pb["tpot"]))
x = range(len(pairs))
b1 = axs[0].bar([i - 0.2 for i in x], [p[1] for p in pairs], 0.38, color=BLUE, label="buffered")
b2 = axs[0].bar([i + 0.2 for i in x], [p[2] for p in pairs], 0.38, color=ORANGE, label="direct I/O")
annotate_bars(axs[0], b1, "{:.0f}", fs=6.5)
annotate_bars(axs[0], b2, "{:.0f}", fs=6.5)
axs[0].set_xticks(list(x))
axs[0].set_xticklabels(["prose", "explain", "summary", "code", "plain\nstreaming"][:len(pairs)], fontsize=7)
axs[0].legend(frameon=False, fontsize=7)
style(axs[0], "Direct I/O: no gain (65-node tree)", yl="ms per token")
for pf in PROMPTS4:
    rs = sorted([r for r in O[7][pf] if r["nodes"] in (65, 129) and not r["io"]], key=lambda r: r["nodes"])
    if len(rs) == 2:
        axs[1].plot([65, 129], [r["tpot"] for r in rs], "o-", color=pcol[pf], label=PL[pf])
axs[1].set_xticks([65, 129])
axs[1].legend(frameon=False, fontsize=7)
style(axs[1], "65 vs 129 nodes", "tree nodes", "ms per token")
fig.suptitle("D5 · Round 4: direct I/O and bigger trees do not help", x=0.01, ha="left", fontsize=10, color=INK)
img(save(fig, "D5_round4.png"), "direct I/O is 1–3% slower end to end; the 129-node tree is slower than the 65-node tree on every prompt")
md("""**What it means.** Direct I/O gives no end-to-end gain for streaming (65-node tree 279 → 288 ms on prose, +2–3% on chat prompts; plain streaming 808 → 834 ms). It only helps partial residency (209 → 195 ms) and plain streaming's first token. The 129-node tree is slower than the 65-node tree on every prompt: verify rises to 1.0–1.1 s while tokens per read barely move. The deep 20-node tree matches or beats the 65-node tree on explain and code, but not on summary or prose. Conclusion: 65 nodes, buffered reads, 4 slots.
""")

# ---- D6
md("""## D6. Optimisation: the net result

**Why / how.** The best configuration of each step, per prompt, in ms per token (rounds 1–3; verified table `docs/archive/FINAL_REPORT.md` §6b). All at 3.1–3.5 GB, 51–54% less than the baseline's 7.07 GB.
""")
D6 = [("start: chain γ=5, 2 pods (B8)", 441, None, None, None), ("1: chain γ=5, single pod, 4 slots", 350, 315, 301, 260),
      ("2: tree, 23–65 nodes, HF drafter", 324, 237, 245, 217), ("3: tree, 65 nodes, lean drafter", 242, 248, 200, 198)]
D6R = [("3.3×", None, None, None), ("2.7×", "2.9×", "2.4×", "2.3×"), ("2.6×", "2.15×", "2.0×", "1.95×"), ("1.9×", "2.2×", "1.7×", "1.7×")]
table(["step", "prose 512", "chat: explain", "chat: summary", "chat: code"],
      [(d[0],) + tuple(f"{v} ms ({r})" if v else "–" for v, r in zip(d[1:], rr_)) for d, rr_ in zip(D6, D6R)] +
      [("baselines", "126–133 ms", "110–115 ms", "118–123 ms", "111–115 ms")])
md("Ratios use the baseline of the round each step was measured in; the round-3 prose baseline ran under load (173 ms), so step 3's prose ratio uses the normal 126–129 ms.\n")
fig, ax = plt.subplots(figsize=(8.6, 3.0))
for j, (name, col) in enumerate(zip(["prose 512", "chat: explain", "chat: summary", "chat: code"], [BLUE, ORANGE, AQUA, MAGENTA])):
    xs = [i for i, d in enumerate(D6) if d[1 + j]]
    ax.plot(xs, [D6[i][1 + j] for i in xs], "o-", color=col, label=name)
ax.axhspan(110, 126, color=GRAY, alpha=0.18)
ax.text(3, 135, "baselines 110–126 ms", ha="right", fontsize=7.5, color=INK2)
ax.set_xticks(range(len(D6)))
ax.set_xticklabels(["start", "1: one pod", "2: tree", "3: lean drafter"])
ax.set_ylim(0, 480)
ax.legend(frameon=False, fontsize=7.5)
style(ax, "D6 · ms per token through the optimisation steps (3.1–3.5 GB throughout)", yl="ms per token")
img(save(fig, "D6_net.png"), "from 441 ms to 198–248 ms per token at the same memory")
md("""**What it means.** From 441 ms to 198–248 ms per token (1.7–2.2× the baselines) at 3.2–3.4 GB, 51–54% less memory than the baseline. Best general configuration: one pod, 4 slots, 65-node tree, lean drafter. What is left, in the order the data suggests: choosing the tree shape per prompt from the draft's confidence, drafting during the verify pass, and pinning the layers that fit in the leftover memory.
""")

# ================================================================================================================ PART E
md("# Part E. Six models: is the best approach the same for every model?\n")
md("""Question (4 October): is "streaming + tree speculation" the best approach for every model? Same protocol on six models (the Qwen2.5 1.5B / 7B / 14B trio plus Qwen3-30B-A3B, with the 3B and gpt-oss re-run under the same protocol): baseline, single-pod streaming (4 slots), streaming + 64-node tree speculation with a lean drafter (Qwen2.5-0.5B, or Qwen3-0.6B for the Qwen3 model; gpt-oss has no drafter), kernel paging at 2–3 memory limits. Prompts: prose 512, chat summary, chat code; 32 tokens; 3 repetitions. 270 runs. All tables in this part are computed here from `results/six-models/*.jsonl`.
""")
LAYERB = {}
LAYERB = {"qwen2.5-1.5b": 2620709952, "qwen2.5-3b": 5549584896, "qwen2.5-7b": 13051268832, "qwen2.5-14b": 26425842432, "gpt-oss-20b": 39513014592,
          "qwen3-30b-a3b": 59821589376}
PROMPTS3 = ["prompt_long.txt", "prompts/chat_summary.txt", "prompts/chat_code.txt"]
SG = {m: sweep_groups(SWEEP[m]) for m in MODELS}
E = {}
for m in MODELS:
    g = SG[m]
    for pf in PROMPTS3:
        base = g[("baseline", None, pf)]
        bt, bram, bttft = per_token(base), ram_of(next(r for r in base if r["stats"])), med(base, "ttft_ms")
        btok = first_tokens(base[0]["tokens"])
        for k, rs in g.items():
            if k[2] != pf:
                continue
            a, lim = k[0], k[1]
            lab = {"baseline": "baseline", "seq1pod": "streaming, 4 slots", "tree": "streaming + tree speculation"}.get(a, f"kernel paging, {(lim or '').replace('Gi', ' GiB').replace('Mi', ' MiB')}")
            same = sum(1 for r in rs if first_tokens(r["tokens"])[:len(btok)] == btok[:len(first_tokens(r["tokens"]))])
            tpc = st.mean([x for r in rs for x in r["spec"]["accepted"]]) + 1 if rs[0].get("spec") else None
            ram = ram_of(next(r for r in rs if r["stats"]))
            E[(m, pf, lab)] = dict(tpot=per_token(rs), ratio=per_token(rs) / bt, ram=ram, saved=1 - ram / bram, ttft=med(rs, "ttft_ms"), tratio=med(rs, "ttft_ms") / bttft,
                                   same=same, n=len(rs), tpc=tpc, lim=lim, a=a)


def lab_order(lab):
    return (["baseline", "streaming, 4 slots", "streaming + tree speculation"].index(lab) if not lab.startswith("kernel") else 3, -float(re.sub(r"[^0-9.]", "", lab.split(", ")[-1]) or 0) if lab.startswith("kernel") else 0)


# ---- E1
md("""## E1. Six models: every approach, every prompt

**Why.** The 3B and gpt-oss pointed to two different winners. Six models of different sizes and both architectures show whether that is a rule.

**How.** One table per model; "best" = lowest slowdown on the prose prompt among configurations that save at least a fifth of the memory.
""")
for m in MODELS:
    md(f"### {MNAME[m]} — {LAYERB[m] / 1e9:.1f} GB of layer weights\n")
    labs = sorted({k[2] for k in E if k[0] == m}, key=lab_order)
    rows = []
    for pf in PROMPTS3:
        for lab in labs:
            e = E.get((m, pf, lab))
            if not e:
                continue
            rows.append((PL[pf], lab, f"{e['ram']:.1f} GB", f"{100 * e['saved']:.0f}%", f"{e['tpot']:.0f} ms", f"{e['ratio']:.1f}×", f"{e['tpc']:.1f}" if e["tpc"] else "",
                         "—" if lab == "baseline" else f"{e['same']}/{e['n']}"))
    table(["prompt", "approach", "RAM", "saved", "per token", "vs baseline", "tokens per read", "same output"], rows)
img("../charts/models.png", "memory saved vs slowdown for every configuration of every model (prose prompt); lower-right is better")
best = []
for m in MODELS:
    cands = [(k[2], e) for k, e in E.items() if k[0] == m and k[1] == "prompt_long.txt" and k[2] != "baseline" and e["saved"] >= 0.20]
    lab, e = min(cands, key=lambda x: x[1]["ratio"])
    rr = [E[(m, pf, lab)]["ratio"] for pf in PROMPTS3 if (m, pf, lab) in E]
    sv = [E[(m, pf, lab)]["saved"] for pf in PROMPTS3 if (m, pf, lab) in E]
    best.append((MNAME[m], lab, rng(100 * min(sv), 100 * max(sv), "{:.0f}", "%"), rng(min(rr), max(rr), "{:.1f}", "×")))
md("**Best approach per model** (ranges over the three prompts):\n")
table(["model", "best approach", "RAM saved", "per token vs baseline"], best)
md("""**What it means.** The rule holds on all six. Dense models (Qwen2.5 1.5B to 14B): streaming + tree speculation wins every time and kernel paging is the worst every time (9–15×). Mixtures of experts (gpt-oss, Qwen3-30B-A3B): kernel paging is the only approach with a moderate cost, while layer streaming is 32–37× and tree speculation cannot rescue it (each read is 40–60 GB). The drafter must be small next to the target: at 1.5B the 1 GB drafter eats most of the saving (24–28% left); from 7B up it costs 2–7% of the model.
""")

# ---- E2
md("""## E2. Does plain streaming sit exactly on the disk floor?

**Why.** The core claim of the project is a predictive rule: time per token ≥ bytes not in memory ÷ disk bandwidth. If plain streaming matches bytes ÷ 7.1 GB/s on all six models, the rule predicts any model on this machine without running it.

**How.** Exact layer bytes per model (sum of the layer files) ÷ 7.1 GB/s, against the measured single-pod streaming time per token on the three prompts.
""")
E2 = []
for m in MODELS:
    pred = LAYERB[m] / 7.1e9 * 1e3
    meas = [E[(m, pf, "streaming, 4 slots")]["tpot"] for pf in PROMPTS3]
    E2.append((m, LAYERB[m] / 1e9, pred, min(meas), max(meas)))
table(["model", "layer bytes", "predicted floor (÷ 7.1 GB/s)", "measured streaming", "measured ÷ predicted"],
      [(MNAME[m], f"{b:.2f} GB", f"{p:.0f} ms", rng(lo, hi, "{:.0f}", " ms"), rng(lo / p, hi / p, "{:.2f}")) for m, b, p, lo, hi in E2])
fig, ax = plt.subplots(figsize=(8.6, 3.3))
mx = max(e[4] for e in E2) * 1.15
ax.plot([100, mx], [100, mx], color=INK2, ls="--", lw=1)
ax.fill_between([100, mx], [100 * 0.95, mx * 0.95], [100 * 1.05, mx * 1.05], color=GRAY, alpha=0.18)
for m, b, p, lo, hi in E2:
    ax.errorbar([p], [(lo + hi) / 2], yerr=[[(lo + hi) / 2 - lo], [hi - (lo + hi) / 2]], fmt="o", color=ORANGE if m in ("gpt-oss-20b", "qwen3-30b-a3b") else BLUE, capsize=3)
    ax.annotate(MNAME[m], (p, hi), xytext=(-4, 6), textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(250, mx)
ax.set_ylim(250, mx)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_ticks([300, 1000, 3000, 10000])
    a_.set_ticklabels(["300", "1,000", "3,000", "10,000"])
ax.minorticks_off()
style(ax, "E2 · Measured streaming time vs predicted floor (band: ±5%)", "predicted: layer bytes ÷ 7.1 GB/s (ms)", "measured per token (ms)", grid="both")
img(save(fig, "E2_floor.png"), "all six models sit on the line: time per token = bytes read ÷ 7.1 GB/s")
md("""**What it means.** Measured streaming time is within 0–5% of layer bytes ÷ 7.1 GB/s for all six models, from 2.6 GB to 60 GB of layers, dense or experts. The rule is predictive: for any model on this machine, divide the bytes you must read per token by 7.1 GB/s.
""")

# ---- E3
md("""## E3. First-token cost of each approach

**Why.** Generation speed is not the whole story: a user also waits for the prompt to be processed. Streaming reads the whole model once for the prompt; kernel paging faults in almost every expert on its first pass.

**How.** Median first-token time, ÷ the same model's baseline, prose 512 prompt and the range over the two chat prompts.
""")
E3 = []
for m in MODELS:
    row = [MNAME[m], f"{E[(m, 'prompt_long.txt', 'baseline')]['ttft'] / 1e3:.1f} s"]
    for lab in ("streaming, 4 slots", "streaming + tree speculation"):
        v = [E[(m, pf, lab)]["tratio"] for pf in PROMPTS3 if (m, pf, lab) in E]
        row.append(f"{v[0]:.1f}× / {min(v[1:]):.1f}–{max(v[1:]):.1f}×" if v else "—")
    kp = [e["tratio"] for k, e in E.items() if k[0] == m and k[2].startswith("kernel")]
    kpp = [e["tratio"] for k, e in E.items() if k[0] == m and k[2].startswith("kernel") and k[1] == "prompt_long.txt"]
    row.append(f"{rng(min(kpp), max(kpp), '{:.1f}', '×')} / up to {max(kp):.1f}×")
    E3.append(row)
table(["model", "baseline first token (prose)", "streaming (prose / chat)", "streaming + tree (prose / chat)", "kernel paging (prose / worst)"], E3)
fig, ax = plt.subplots(figsize=(8.6, 3.0))
x = range(len(MODELS))
s1 = [E[(m, "prompt_long.txt", "streaming, 4 slots")]["tratio"] for m in MODELS]
s2 = [min(e["tratio"] for k, e in E.items() if k[0] == m and k[1] == "prompt_long.txt" and k[2].startswith("kernel")) for m in MODELS]
s3 = [max(e["tratio"] for k, e in E.items() if k[0] == m and k[1] == "prompt_long.txt" and k[2].startswith("kernel")) for m in MODELS]
b1 = ax.bar([i - 0.27 for i in x], s1, 0.26, color=BLUE, label="streaming")
b2 = ax.bar([i for i in x], s2, 0.26, color=YELLOW, label="paging, largest limit")
b3 = ax.bar([i + 0.27 for i in x], s3, 0.26, color=ORANGE, label="paging, smallest limit")
for bars in (b1, b2, b3):
    annotate_bars(ax, bars, "{:.1f}×", fs=6.5)
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.set_xticks(list(x))
ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=7.5)
ax.legend(frameon=False, fontsize=7.5)
style(ax, "E3 · First token ÷ baseline, 512-token prose prompt", yl="× baseline")
img(save(fig, "E3_first_token.png"), "streaming costs 1.2–2.4× on the long prompt; paging up to 6× on the mixtures of experts")
md("""**What it means.** Streaming costs 1.2–1.4× on the 512-token prompt for dense models and 2.0–2.4× for the mixtures of experts, more on short chat prompts (the prompt pass is one full read of the weights). Kernel paging costs about 2× on dense models and 3.5–6× on the mixtures of experts on the long prompt, up to 11× on short prompts (11–26 s for Qwen3-30B-A3B), because the first pass touches nearly every expert. If time to first token matters, paging on a mixture of experts needs a warm-up.
""")

# ---- E4
md("""## E4. Replication of the mixture-of-experts paging rows with the host cache cleared

**Why.** The gpt-oss paging numbers in E1 were 3–4× worse than the first measurement (C7). One hypothesis: during the sweep the host held 79 GB of cached files (downloads and splits), and global memory pressure might evict the pod's hot experts. This test removes that pressure.

**How.** Evict all large files from the page cache (79 → 51 GB cached, 26 → 54 GB free), then re-measure every kernel-paging configuration of both mixtures of experts, all three prompts, 3 repetitions (45 runs).
""")
E4 = []
for r_m in ("gpt-oss-20b", "qwen3-30b-a3b"):
    rep = defaultdict(list)
    for r in REPL:
        if r["model"] == r_m:
            rep[(r["MEM_LIMIT"], r["PROMPT_FILE"])].append(r)
    for (lim, pf), rs in sorted(rep.items(), key=lambda kv: (-float(kv[0][0].rstrip("Gi")), PROMPTS3.index(kv[0][1]))):
        sw = per_token(SG[r_m][("mmap", lim, pf)])
        rp = per_token(rs)
        E4.append((MNAME[r_m], lim.replace("Gi", " GiB"), PL[pf], sw, rp))
table(["model", "limit", "prompt", "sweep", "replication (cache cleared)", "difference"], [(a, b, c, f"{d:.0f} ms", f"{e:.0f} ms", f"{round(100 * (e / d - 1)):+d}%".replace("+0%", "0%")) for a, b, c, d, e in E4])
fig, ax = plt.subplots(figsize=(8.6, 3.2))
mx = max(max(e[3], e[4]) for e in E4) * 1.1
ax.plot([300, mx], [300, mx], color=INK2, ls="--", lw=1)
ax.fill_between([300, mx], [300 * 0.9, mx * 0.9], [300 * 1.1, mx * 1.1], color=GRAY, alpha=0.18)
for a, b, c, d, e in E4:
    ax.scatter(d, e, color=ORANGE if a == "gpt-oss-20b" else MAGENTA, s=30, zorder=3)
ax.legend(handles=[Patch(color=ORANGE, label="gpt-oss-20b"), Patch(color=MAGENTA, label="Qwen3-30B-A3B")], frameon=False)
ax.set_xlim(300, mx)
ax.set_ylim(300, mx)
style(ax, "E4 · Sweep vs replication with the host cache cleared (band: ±10%)", "sweep (ms per token)", "replication (ms per token)", grid="both")
img(save(fig, "E4_replication.png"), "all 15 configurations reproduce within 10%: host cache pressure was not the cause")
md("""**What it means.** All 15 configurations reproduce within 10%, and 12 of 15 within 4%. Host page-cache pressure did not distort the sweep; the E1 paging numbers stand. The cause of the earlier, better gpt-oss figure had to be elsewhere (E5).
""")

# ---- E5
md("""## E5. Why gpt-oss paging looked like 1.87× before: the degenerate prompt

**Why.** The first gpt-oss paging runs used the original repetitive prompt (`prompt.txt`, 373 tokens tiled to 512). On that prompt gpt-oss generates the token "…" sixteen times in a row. A repeated token routes to the same experts at every step, so a cache of hot experts would hit almost every time.

**How.** Same deployment type, page cache cleared, the old repetitive prompt: 16 GiB limit with 32 and 64 generated tokens, 26 GiB, and the baseline (12 runs). Compared with the same limits on the real prose prompt from the sweep.
""")
rc = defaultdict(list)
for r in RECHECK:
    rc[(r["approach"], r.get("MEM_LIMIT"), r["recheck"])].append(r)
rbase = per_token(rc[("baseline", None, "after-evict")])
E5 = [("baseline", "repetitive", rbase, 1.0), ("paging 26 GiB", "repetitive", per_token(rc[("mmap", "26Gi", "after-evict")]), None),
      ("paging 16 GiB", "repetitive", per_token(rc[("mmap", "16Gi", "after-evict")]), None), ("paging 16 GiB, 64 tokens", "repetitive", per_token(rc[("mmap", "16Gi", "after-evict-64tok")]), None),
      ("baseline", "real prose", gob["prompt_long.txt"], 1.0), ("paging 26 GiB", "real prose", per_token(GOS[("mmap", "26Gi", "prompt_long.txt")]), None),
      ("paging 16 GiB", "real prose", per_token(GOS[("mmap", "16Gi", "prompt_long.txt")]), None)]
E5 = [(a, b, c, c / (rbase if b == "repetitive" else gob["prompt_long.txt"])) for a, b, c, _ in E5]
table(["configuration", "prompt", "per token", "vs baseline"], [(a, b, f"{c:.0f} ms", f"{d:.2f}×") for a, b, c, d in E5] + [("first measurement, 16 GiB (1–2 Oct)", "repetitive", "314 ms", "1.87×")])
fig, ax = plt.subplots(figsize=(8.6, 2.9))
labs = ["26 GiB", "16 GiB"]
rep = [E5[1][3], E5[2][3]]
real = [E5[5][3], E5[6][3]]
x = range(2)
b1 = ax.bar([i - 0.2 for i in x], rep, 0.38, color=GRAY, label="repetitive prompt ('…' × 16)")
b2 = ax.bar([i + 0.2 for i in x], real, 0.38, color=ORANGE, label="real prose prompt")
annotate_bars(ax, b1, "{:.2f}×")
annotate_bars(ax, b2, "{:.2f}×")
ax.axhline(1, color=INK2, ls="--", lw=1)
ax.set_xticks(list(x))
ax.set_xticklabels(labs)
ax.legend(frameon=False)
ax.set_ylim(0, 9)
style(ax, "E5 · gpt-oss kernel paging, same deployment, two prompts", yl="× baseline per token")
img(save(fig, "E5_degenerate.png"), "the repetitive prompt reproduces the old 1.87× (1.85×); real text costs 7.4×")
md("""**What it means.** With the repetitive prompt the 16 GiB limit gives 284 ms (1.85×), reproducing the first measurement's 1.87× almost exactly; with real prose the same deployment gives 1120 ms (7.4×). The earlier figure described a degenerate output, not normal text. The gpt-oss report, the record, the summary, the explainer and the video were corrected. Lesson: check that generated tokens are not degenerate before trusting a cache-sensitive result.
""")

# ================================================================================================================ PART F
md("# Part F. Correctness of the outputs\n")

# ---- F1
md("""## F1. Is every output the same as the baseline?

**Why.** All approaches except speculative decoding compute exactly what the baseline computes, just with weights coming from a different place, so their output must be identical. Speculative decoding keeps only tokens the big model itself chose, so it should be identical too, except where bf16 rounding of a differently shaped computation picks the other side of a near-tie.

**How.** Every run's generated tokens compared one by one with the same round's baseline, computed here from the raw files.
""")


def identity(rows, is_base, key):
    base = {}
    for r in rows:
        if is_base(r) and key(r) not in base:
            base[key(r)] = first_tokens(r["tokens"])
    same = diff = 0
    for r in rows:
        if is_base(r):
            continue
        b = base.get(key(r))
        t = first_tokens(r["tokens"])
        n = min(len(b), len(t))
        if t[:n] == b[:n]:
            same += 1
        else:
            diff += 1
    return same, diff


kA = lambda r: (str(r.get("n_prompt")), r.get("PROMPT_FILE"))
F1 = [("3B experiment 1 (cyclic, controls)", *identity(EXP1_3B, lambda r: r["G"] == 1 and r["mode"] == "resident", lambda r: r["n_prompt"]), "—"),
      ("3B experiment 2 (alternatives)", *identity(EXP2_3B, lambda r: r["approach"] == "baseline", kA), "token 15 of the 128-token prompt (speculative and batched runs)"),
      ("3B single-pod streaming", *identity(EXP3_3B, lambda r: r["approach"] == "baseline", kA), "—"),
      ("gpt-oss experiment 1", *identity(EXP1_GO, lambda r: r["G"] == 1 and r["mode"] == "resident", lambda r: r["n_prompt"]), "—"),
      ("gpt-oss experiment 2", *identity(EXP2_GO, lambda r: r["approach"] == "baseline", kA), "—")] + \
     [(f"optimisation round {i}", *identity(OPT[n], lambda r: r["approach"] == "baseline", kA), w) for i, n, w in
      ((1, 4, "—"), (2, 5, "token 32 of the summary prompt"), (3, 6, "token 36 of explain, token 32 of summary"), (4, 7, "token 36 of explain, token 32 of summary"))]
ns = ns_d = sp = sp_d = 0
for m in MODELS:
    for pf in PROMPTS3:
        for k, e in E.items():
            if k[0] == m and k[1] == pf and k[2] != "baseline":
                if e["a"] == "tree":
                    sp += e["same"]
                    sp_d += e["n"] - e["same"]
                else:
                    ns += e["same"]
                    ns_d += e["n"] - e["same"]
rs_same = rs_diff = 0
for r in REPL:
    b = first_tokens(SG[r["model"]][("baseline", None, r["PROMPT_FILE"])][0]["tokens"])
    t = first_tokens(r["tokens"])
    n = min(len(b), len(t))
    rs_same += t[:n] == b[:n]
    rs_diff += t[:n] != b[:n]
rc_same, rc_diff = identity(RECHECK, lambda r: r["approach"] == "baseline", lambda r: 0)
F1 += [("six models: streaming and paging", ns, ns_d, "—"), ("six models: tree speculation", sp, sp_d, "five near-tie tokens (F2)"),
       ("paging replication (E4)", rs_same, rs_diff, "—"), ("degenerate-prompt recheck (E5)", rc_same, rc_diff, "—")]
table(["experiment", "runs identical", "runs diverging", "where they diverge"], [(a, b, c, d) for a, b, c, d in F1])
tot_s, tot_d = sum(f[1] for f in F1), sum(f[2] for f in F1)
md(f"Total: **{tot_s} of {tot_s + tot_d} non-baseline runs identical**; the {tot_d} others diverge at one of eight near-tie tokens, never elsewhere.\n")
fig, ax = plt.subplots(figsize=(8.6, 3.6))
y = range(len(F1))
ax.barh(list(y), [f[1] for f in F1], color=AQUA, label="identical")
ax.barh(list(y), [f[2] for f in F1], left=[f[1] for f in F1], color=ORANGE, label="diverge at a near-tie")
for i, f in enumerate(F1):
    ax.annotate(f"{f[1]}/{f[1] + f[2]}", (f[1] + f[2], i), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_yticks(list(y))
ax.set_yticklabels([f[0] for f in F1], fontsize=7.5)
ax.invert_yaxis()
ax.legend(frameon=False, loc="lower right")
style(ax, "F1 · Runs whose tokens match the baseline exactly", "runs", grid="x")
img(save(fig, "F1_identity.png"), "every non-speculative run is identical; speculative and batched runs diverge only at near-ties")
md("""**What it means.** Every streaming, partial-residency, kernel-paging and control run reproduces the baseline exactly. All divergences come from runs whose computation has a different shape (speculative verify passes, batched passes) and all sit at a handful of specific tokens. F2 shows those tokens are near-ties.
""")

# ---- F2
md("""## F2. The divergences are near-ties, not bugs

**Why.** If a divergence were a bug, the model in full precision would clearly prefer one token. If it is a near-tie, the top two candidates are almost equal and bf16 rounding can tip either way.

**How.** At each divergence position, the full model in fp32 (`tie_check.py`, and `fp32_ref.py` which streams the model layer by layer in fp32 so even the 30B fits in a few GB) gives the top-2 logits and their gap.
""")
F2 = [("3B, 128-token prompt (exp. 2)", 15, "simplicity", 18.532, "power", 18.437, "power", "simplicity", "speculative / batched"),
      ("3B, chat explain (rounds 3–4)", 36, "frequently", 30.392, "copies", 30.329, "copies", "frequently", "speculative"),
      ("3B, chat summary (rounds 2–4)", 32, "increased", 25.998, "made", 25.945, "increased", "made", "baseline"),
      ("1.5B, prose (six models)", 10, "move", 21.428, "load", 21.356, "load", "move", "speculative"),
      ("7B, prose (six models)", 23, "seem", 22.108, "look", 22.047, "look", "seem", "speculative"),
      ("7B, chat summary (six models)", 4, "measured", 23.611, "replaced", 23.337, "measured", "replaced", "baseline"),
      ("14B, prose (six models)", 5, "solution", 15.603, "only", 15.549, "solution", "only", "baseline"),
      ("30B-A3B, prose (six models)", 16, ".", 27.819, ",", 27.603, ".", ",", "baseline")]
table(["where", "token", "fp32 first choice", "fp32 second", "gap (logits)", "gap (%)", "baseline picked", "other run picked", "fp32 agrees with"],
      [(a, b, f'"{c}" {d}', f'"{e}" {f}', f"{d - f:.3f}", f"{100 * (d - f) / d:.2f}%", f'"{g}"', f'"{h}"', i) for a, b, c, d, e, f, g, h, i in F2])
fig, ax = plt.subplots(figsize=(8.6, 3.2))
gaps = [d - f for _, _, _, d, _, f, _, _, _ in F2]
ax.axhspan(0.16, 0.42, color=YELLOW, alpha=0.25)
ax.text(len(F2) - 0.5, 0.43, "band: measured bf16 shape noise, median 0.16 to maximum 0.42", ha="right", fontsize=7.5, color=INK2)
bb = ax.bar(range(len(F2)), gaps, color=[AQUA if x[8] == "speculative" or x[8] == "speculative / batched" else BLUE for x in F2], width=0.6)
annotate_bars(ax, bb, "{:.3f}", fs=7)
ax.set_xticks(range(len(F2)))
ax.set_xticklabels([x[0].split(" (")[0] for x in F2], fontsize=7, rotation=20, ha="right")
ax.set_ylim(0, 0.5)
ax.legend(handles=[Patch(color=AQUA, label="fp32 sides with the speculative/batched run"), Patch(color=BLUE, label="fp32 sides with the baseline")], frameon=False, loc="upper left", fontsize=7.5)
style(ax, "F2 · fp32 gap between the top two tokens at each divergence", yl="logit gap")
img(save(fig, "F2_near_ties.png"), "every gap is below the largest bf16 shape noise measured; six of eight are below even its median")
md("""**What it means.** At every divergence the top two tokens are within 0.05–0.27 logit points (0.2–1.2%) of each other in fp32. Six of the eight gaps are smaller than even the median bf16 shape noise (0.16, F3), and all eight are smaller than the largest shift measured on each model (0.28 and 0.42). The full-precision model sides with the speculative run four times and with the baseline four times, so neither is "wrong": both are valid bf16 roundings of a tie. The same divergence can even change between repetitions (7B prose: 2 of 3 runs matched).
""")

# ---- F3
md("""## F3. How large is bf16 rounding noise?

**Why.** F2's argument needs a number: how much do logits move when the same position is computed in a different shape?

**How.** On two small models (Qwen2.5-0.5B, Qwen3-0.6B), 20 positions each: logits from one full-sequence pass vs. a prefix pass plus a single-token step with the cache, both in bf16.
""")
F3 = [("Qwen2.5-0.5B", 20, 0.160, 0.281, "1 of 20"), ("Qwen3-0.6B", 20, 0.234, 0.422, "0 of 20")]
table(["model", "positions", "median max-|Δ logit|", "largest", "top-1 flips"], [(a, b, f"{c:.3f}", f"{d:.3f}", e) for a, b, c, d, e in F3])
fig, ax = plt.subplots(figsize=(8.6, 2.6))
x = range(2)
b1 = ax.bar([i - 0.2 for i in x], [f[2] for f in F3], 0.38, color=BLUE, label="median")
b2 = ax.bar([i + 0.2 for i in x], [f[3] for f in F3], 0.38, color=ORANGE, label="largest")
annotate_bars(ax, b1, "{:.2f}")
annotate_bars(ax, b2, "{:.2f}")
ax.axhline(max(gaps), color=AQUA, ls="--", lw=1)
ax.text(1.45, max(gaps) + 0.01, "largest near-tie gap (F2)", ha="right", fontsize=7.5, color=AQUA)
ax.set_xticks(list(x))
ax.set_xticklabels([f[0] for f in F3])
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, loc="upper left")
style(ax, "F3 · Logit movement from computing the same position in a different shape (bf16)", yl="logit points")
img(save(fig, "F3_noise.png"), "shape alone moves logits by up to 0.28–0.42 points, more than any near-tie gap")
md("""**What it means.** Computing the same position in a different shape moves bf16 logits by 0.16–0.23 points at the median and up to 0.42, and flipped the top token once in 40 positions. The largest shifts (0.28 and 0.42) exceed every near-tie gap in F2 (at most 0.274), and the medians exceed six of the eight, so those divergences are expected bf16 behaviour, not bugs. Exact-output claims for speculative decoding in bf16 should therefore always be checked against an fp32 reference.
""")

# ================================================================================================================ PART G + appendix
md("# Part G. What went wrong during testing, and what was done\n")
table(["problem", "how it was found", "what was done", "effect on results"], [
    ("Page cache served 'disk' reads (kernel paging, twice)", "a 1.5 GiB pod ran at full speed", "deploy, then drop the cache, then measure", "first two attempts discarded"),
    ("Root-owned weight files blocked the cache drop (gpt-oss)", "cgroup peak 0.95 GB at every limit", "files made readable, 15 rows rerun", "rows discarded and rerun"),
    ("Single-pod deadlock at 2048 tokens", "thread dumps", "chain consecutive groups in-process", "2 rows discarded"),
    ("Repetitive prompt inflated speculative decoding (3B)", "100% draft acceptance", "750-token non-repeating text", "artifact rows excluded (B8)"),
    ("Direct I/O 'faster than the link' (btrfs fallback)", "11 GB/s on a 7.5 GB/s link; 48% of files in cache", "end-of-file-safe loader that evicts fallbacks", "round-1 direct rows invalid (D1)"),
    ("Degenerate '…' output inflated gpt-oss paging", "six-model test disagreed with C7", "replication (E4) and recheck (E5)", "gpt-oss paging corrected to 7.4–8.0×"),
    ("Other users' load on halo", "baselines up to 40% higher in some rounds", "same-round comparisons only", "flagged in D4–D5 and E"),
    ("Report errors (units, run counts, cross-references)", "independent re-checks of every number", "100+ corrections applied", "reports corrected"),
    ("Sweep script copied without its executable bit; a process check matched itself", "empty logs", "relaunched", "none"),
    ("Remote connection to the test machine dropped", "ssh timeouts", "reconnect; sweeps ran unattended on the machine", "none"),
])
md("# Appendix. Data files\n")
table(["file", "content", "runs"], [(f, d, len(r)) for f, d, r in FILES] + [("**total**", "", f"**{TOTAL_RUNS:,}**")])
md("""Charts in this report: `charts/detailed/*.png` (generated by `scripts/make_detailed_report.py` from the data above) and `charts/models.png` (from `scripts/analyze_models.py`). Other documents: `docs/REPORT.md` (decision rules), `docs/FULL_RECORD.md` (lab record), and in `docs/archive/` the earlier reports `FINAL_REPORT.md` (3B), `FINAL_REPORT_GPTOSS.md` and `REPORT_MODELS.md` (six models).
""")

# ================================================================================================================ write
text = "\n".join(doc)
open(os.path.join(HERE, "docs", "DETAILED_REPORT.md"), "w").write(text)
try:
    import markdown
    body = markdown.markdown(text, extensions=["tables"])

    def embed(m):
        p = os.path.normpath(os.path.join(HERE, "docs", m.group(1)))
        return f'src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}"'

    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    css = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:900px;margin:28px auto;padding:0 24px;font-size:13px;line-height:1.5}
h1{font-size:22px;margin:34px 0 8px;padding-bottom:5px;border-bottom:2px solid #0b0b0b} h1:first-child{margin-top:0;border:none}
h2{font-size:16.5px;margin:28px 0 8px;padding-bottom:3px;border-bottom:1px solid #e6e5e0;page-break-after:avoid} h3{font-size:14px;margin:18px 0 6px}
p{margin:6px 0} ul,ol{margin:6px 0 6px 20px;padding:0} li{margin:3px 0}
table{border-collapse:collapse;width:100%;margin:8px 0 10px;font-size:11.5px} tr{page-break-inside:avoid} th,td{border:1px solid #e6e5e0;padding:4px 7px;text-align:left;vertical-align:top} th{background:#f5f5f2}
img{max-width:100%;display:block;margin:10px auto 2px;page-break-inside:avoid} em{color:#52514e;font-size:12px}
code{font-family:Menlo,monospace;font-size:11.5px;background:#f5f5f2;padding:1px 4px;border-radius:3px}
@page{size:A4;margin:14mm 12mm}"""
    open(os.path.join(HERE, "docs", "DETAILED_REPORT.html"), "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Detailed test report</title><style>{css}</style></head><body>{body}</body></html>")
    print("wrote DETAILED_REPORT.md, DETAILED_REPORT.html,", len(os.listdir(OUT)), "charts,", TOTAL_RUNS, "runs")
except ImportError:
    print("wrote DETAILED_REPORT.md (install `markdown` for the HTML)")
