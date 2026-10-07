"""Chart for the latency-optimisation rounds: per-token latency (× baseline) per prompt for the main configurations.

    python3 chart_opt.py results/results4.jsonl results/results5.jsonl results/results6.jsonl [results/results7.jsonl] -> results/optimisation.png
"""
import json, os, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
OUT = os.path.join(os.path.dirname(sys.argv[1]), "optimisation.png")

KEYS = ("approach", "G", "K", "SLOTS", "LOAD_MODE", "GAMMA", "TREE", "RESIDENT", "PROMPT_FILE", "DRAFT", "DRAFT_LAYERS", "n_prompt")
rows = defaultdict(list)
for path in sys.argv[1:]:
    for l in open(path):
        r = json.loads(l)
        rows[(os.path.basename(path),) + tuple(r.get(k) for k in KEYS)].append(r)


def per_token(rs):
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


# series to show, in order: (label, predicate on (file, dict))
def is_(d, **kw):
    return all(d.get(k) == v for k, v in kw.items())


SERIES = [
    ("chain γ=5, 2 pods", lambda f, d: f == "results4.jsonl" and is_(d, approach="spec", K=2, LOAD_MODE=None), INK2),
    ("chain γ=5, 1 pod 4 slots", lambda f, d: f == "results5.jsonl" and is_(d, approach="spec", SLOTS="4"), BLUE),
    ("tree 23 nodes, HF drafter", lambda f, d: f == "results5.jsonl" and is_(d, approach="tree", TREE="4,6,6,4,2"), ORANGE),
    ("tree 65 nodes, HF drafter", lambda f, d: f == "results5.jsonl" and is_(d, approach="tree", TREE="8,16,16,12,8,4"), YELLOW),
    ("tree 65 nodes, lean drafter", lambda f, d: f == "results6.jsonl" and is_(d, approach="tree", TREE="8,16,16,12,8,4", SLOTS="4", LOAD_MODE=None), AQUA),
    ("tree 65 nodes, lean, 5 slots", lambda f, d: f == "results6.jsonl" and is_(d, approach="tree", TREE="8,16,16,12,8,4", SLOTS="5"), MAGENTA),
    ("tree 65 nodes, lean, direct I/O", lambda f, d: f == "results7.jsonl" and is_(d, approach="tree", TREE="8,16,16,12,8,4", LOAD_MODE="direct"), "#008300"),
    ("tree 129 nodes, lean", lambda f, d: f == "results7.jsonl" and is_(d, approach="tree", TREE="16,32,32,24,16,8"), "#4a3aa7"),
]
PROMPTS = [("prompt_long.txt", "prose, 512 tok"), ("prompts/chat_explain.txt", "chat: explain"),
           ("prompts/chat_summary.txt", "chat: summary"), ("prompts/chat_code.txt", "chat: code")]

# baseline per (file, prompt): use the same file's baseline (same-time conditions)
base = {}
for (f, *k), rs in rows.items():
    d = dict(zip(KEYS, k))
    if d["approach"] == "baseline":
        base[(f, d["PROMPT_FILE"])] = per_token(rs)

vals = {}  # (series idx, prompt) -> (ratio, ms, ram)
for (f, *k), rs in rows.items():
    d = dict(zip(KEYS, k))
    for i, (lab, pred, c) in enumerate(SERIES):
        if pred(f, d) and (f, d["PROMPT_FILE"]) in base:
            s0 = next(r for r in rs if r["stats"])
            ram = (sum(x["VmHWM"] for x in s0["stats"]["stages"]) + s0["stats"]["coordinator"]["VmHWM"]) / 1e9
            vals[(i, d["PROMPT_FILE"])] = (per_token(rs) / base[(f, d["PROMPT_FILE"])], per_token(rs), ram)

fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURF)
ax.set_facecolor(SURF)
n = len(SERIES)
w = 0.8 / n
for i, (lab, pred, c) in enumerate(SERIES):
    xs, ys = [], []
    for j, (pf, _) in enumerate(PROMPTS):
        if (i, pf) in vals:
            xs.append(j + (i - n / 2 + 0.5) * w)
            ys.append(vals[(i, pf)][0])
            ax.annotate(f"{vals[(i, pf)][1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color=INK2)
    ax.bar(xs, ys, w * 0.92, color=c, label=lab, edgecolor=SURF, linewidth=1)
ax.axhline(1.0, color=INK2, lw=1.2, ls="--")
ax.text(len(PROMPTS) - 0.55, 1.03, "baseline (all in RAM)", color=INK2, fontsize=8, ha="right")
ax.set_xticks(range(len(PROMPTS)))
ax.set_xticklabels([p[1] for p in PROMPTS])
ax.set_ylabel("per-token latency ÷ baseline", color=INK2, fontsize=9)
ax.set_title("Latency optimisation, 3B, ≤ 3.5 GB: per-token latency by configuration (labels: ms)", color=INK, fontsize=11, loc="left")
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
for (i, pf), (ratio, ms, ram) in sorted(vals.items()):
    print(f"{SERIES[i][0]:30s} {pf:28s} {ms:5.0f} ms  {ratio:4.2f}x  {ram:4.2f} GB")
