"""Cross-model comparison: baseline vs single-pod streaming vs streaming + tree speculation vs kernel paging.

    python3 scripts/analyze_models.py results/six-models/*.jsonl [--chart charts/models.png]
Per-token = total decode time / tokens for speculative runs, median of tpot_ms otherwise; RAM = all pods (decimal GB).
"""
import json, os, statistics as st, sys
from collections import defaultdict

files = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
chart = sys.argv[sys.argv.index("--chart") + 1] if "--chart" in sys.argv else None
KEYS = ("model", "approach", "SLOTS", "TREE", "MEM_LIMIT", "PROMPT_FILE", "n_prompt", "cache")
groups = defaultdict(list)
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r.get("recheck"):       # diagnostic rows (recheck_paging.sh) are reported separately
            continue
        groups[tuple(r.get(k) for k in KEYS)].append(r)


def per_token(rs):
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


rows = []
for k, rs in groups.items():
    d = dict(zip(KEYS, k))
    s0 = next(r for r in rs if r["stats"])
    ram = (sum(x["VmHWM"] for x in s0["stats"]["stages"]) + s0["stats"]["coordinator"]["VmHWM"]) / 1e9
    rec = dict(d, tpot=per_token(rs), ttft=st.median([r["ttft_ms"] for r in rs]), ram=ram, reps=len(rs), tokens=[r["tokens"] for r in rs])
    if rs[0].get("spec"):
        rec["tpc"] = st.mean([x for r in rs for x in r["spec"]["accepted"]]) + 1
    rows.append(rec)

MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "gpt-oss-20b", "qwen3-30b-a3b"]
PROMPTS = {"prompt_long.txt": "prose 512", "prompts/chat_summary.txt": "chat: summary", "prompts/chat_code.txt": "chat: code"}
summary = defaultdict(dict)  # (model, prompt) -> {approach-label: (tpot, ratio, ram, ram_saved, identical, tpc)}
for m in sorted({r["model"] for r in rows}, key=lambda x: MODELS.index(x) if x in MODELS else 99):
    for pf, pl in PROMPTS.items():
        base = next((r for r in rows if r["model"] == m and r["PROMPT_FILE"] == pf and r["approach"] == "baseline"), None)
        if not base:
            continue
        bt = base["tokens"][0]
        for r in rows:
            if r["model"] != m or r["PROMPT_FILE"] != pf or r["approach"] == "baseline":
                continue
            lab = {"seq1pod": "streaming (1 pod, 4 slots)", "tree": "streaming + tree speculation", "mmap": f"kernel paging, limit {r['MEM_LIMIT']}"}.get(r["approach"], r["approach"])
            if r["approach"] == "mmap" and r.get("cache") == "evicted":
                lab += " (replication, host cache cleared)"
            ident = sum(1 for t in r["tokens"] if t[:len(bt)] == bt[:len(t)])
            summary[(m, pl)][lab] = (r["tpot"], r["tpot"] / base["tpot"], r["ram"], 1 - r["ram"] / base["ram"], f"{ident}/{len(r['tokens'])}", r.get("tpc"))
        summary[(m, pl)]["baseline"] = (base["tpot"], 1.0, base["ram"], 0.0, "—", None)

print("| model | prompt | approach | RAM | RAM saved | per token | vs baseline | tokens/read | output = baseline |")
print("|---|---|---|---|---|---|---|---|---|")
for (m, pl), d in summary.items():
    for lab in ["baseline", "streaming (1 pod, 4 slots)", "streaming + tree speculation"] + sorted((k for k in d if k.startswith("kernel")), key=lambda k: ("replication" in k, k)):
        if lab not in d:
            continue
        tpot, ratio, ram, saved, ident, tpc = d[lab]
        print(f"| {m} | {pl} | {lab} | {ram:.1f} GB | {saved * 100:.0f}% | {tpot:.0f} ms | {ratio:.2f}× | {f'{tpc:.1f}' if tpc else ''} | {ident} |")

if chart:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    NAMES = {"qwen2.5-1.5b": "Qwen2.5-1.5B (dense)", "qwen2.5-3b": "Qwen2.5-3B (dense)", "qwen2.5-7b": "Qwen2.5-7B (dense)",
             "qwen2.5-14b": "Qwen2.5-14B (dense)", "gpt-oss-20b": "gpt-oss-20b (MoE, 4 of 32 experts)", "qwen3-30b-a3b": "Qwen3-30B-A3B (MoE, 8 of 128 experts)"}
    ms = [m for m in MODELS if any(k[0] == m for k in summary)]
    cols = 3
    rws = (len(ms) + cols - 1) // cols
    fig, axs = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.6 * rws), facecolor=SURF, squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, m in zip(axs.flat, ms):
        ax.set_visible(True)
        ax.set_facecolor(SURF)
        d = summary.get((m, "prose 512"), {})
        kp = 0
        for lab, v in sorted(d.items(), key=lambda kv: kv[1][3]):
            tpot, ratio, ram, saved, ident, tpc = v
            if lab == "baseline" or "replication" in lab:
                continue
            c, mk = (BLUE, "o") if lab.startswith("streaming (") else (AQUA, "D") if lab.startswith("streaming +") else (YELLOW, "s")
            ax.scatter([saved * 100], [ratio], s=70, color=c, marker=mk, zorder=3, edgecolor=SURF)
            txt = f"{ratio:.1f}×" + (f" ({lab.split('limit ')[1]})" if lab.startswith("kernel") else "")
            off, ha = (6, -3), "left"
            if lab.startswith("kernel"):      # alternate above/below so neighbouring limits don't overlap
                off, ha = ((0, 8), "center") if kp % 2 == 0 else ((0, -13), "center")
                kp += 1
            ax.annotate(txt, (saved * 100, ratio), textcoords="offset points", xytext=off, ha=ha, fontsize=7.5, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 60)
        ax.set_xlim(0, 100)
        ax.axhline(1, color=INK2, ls="--", lw=0.8)
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×", "50×"])
        ax.minorticks_off()
        ax.set_title(NAMES.get(m, m), color=INK, fontsize=9.5, loc="left")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("RAM saved vs baseline (%)", color=INK2, fontsize=8.5)
    for r_ in axs:
        r_[0].set_ylabel("per token ÷ baseline", color=INK2, fontsize=8.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=BLUE, label="streaming (1 pod, 4 slots)"),
         Line2D([], [], marker="D", ls="", color=AQUA, label="streaming + tree speculation"),
         Line2D([], [], marker="s", ls="", color=YELLOW, label="kernel paging (memory limit in label)")]
    fig.legend(handles=h, loc="lower center", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Memory saved vs. slowdown, per model (512-token prose prompt, 32 tokens, 3 runs). Lower-right is better.", color=INK, fontsize=10.5, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    fig.savefig(chart, dpi=150)
    print("wrote", chart)
