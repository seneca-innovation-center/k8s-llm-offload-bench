"""results/results.jsonl -> results/summary.csv, results/*.png, REPORT tables (stdout, markdown)."""
import csv, json, os, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("R", "/exp/results")
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
L = int(os.environ.get("L", "36"))
MODEL_GB = float(os.environ.get("MODEL_GB", "5.55"))
DISK = json.load(open(f"{R}/diskbw.json"))["aggregate_GBps_by_concurrency"]
DISK_MAX = max(DISK.values())

recs = [json.loads(l) for l in open(f"{R}/results.jsonl")]
cfgs = defaultdict(list)
for r in recs:
    cfgs[(r["G"], r["K"], r["mode"], r["n_prompt"])].append(r)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


rows = []
for (G, K, mode, n), rs in cfgs.items():
    ttft = [r["ttft_ms"] for r in rs]
    tpot = [x for r in rs for x in r["tpot_ms"]]
    s0 = next(r for r in rs if r["stats"])
    stats, tr = s0["stats"], s0["traces"][1:]
    comp = [sum(t["end"] - t["start"] for t in tok) * 1e3 for tok in tr]
    wait = [sum(t["start"] - t["arr"] for t in tok) * 1e3 for tok in tr]
    hop = [sum(b["arr"] - a["end"] for a, b in zip(tok, tok[1:])) * 1e3 for tok in tr]
    stages = stats["stages"]
    rows.append(dict(
        G=G, K=K, mode=mode, n_prompt=n, reps=len(rs),
        resident_layers=K * L // G if mode == "cyclic" else L,
        ttft_med=st.median(ttft), ttft_p95=pct(ttft, .95),
        tpot_med=st.median(tpot), tpot_p95=pct(tpot, .95),
        dec_compute=st.median(comp), dec_wait=st.median(wait), dec_hop=st.median(hop),
        weights_peak_mb=sum(s["peak"] for s in stages) / 2**20,
        stage_rss_mb=sum(s["VmHWM"] for s in stages) / 2**20,
        kv_mb=sum(s["kv_bytes"] for s in stages) / 2**20,
        coord_rss_mb=stats["coordinator"]["VmHWM"] / 2**20,
        tokens=rs[0]["tokens"], all_reps_tokens=[r["tokens"] for r in rs]))

# baseline: pool both baseline runs (start + end of sweep) to bound drift
base = {r["n_prompt"]: r for r in rows if r["G"] == 1}
for r in rows:
    b = base[r["n_prompt"]]
    r["ttft_x"] = r["ttft_med"] / b["ttft_med"]
    r["tpot_x"] = r["tpot_med"] / b["tpot_med"]
    r["mem_frac"] = r["weights_peak_mb"] / b["weights_peak_mb"]
    r["tokens_match"] = all(t == b["tokens"] for t in r["all_reps_tokens"])
rows.sort(key=lambda r: (r["n_prompt"], r["mode"] != "resident" or r["G"] != 1, r["mode"], -r["G"], r["K"]))

keys = [k for k in rows[0] if k not in ("tokens", "all_reps_tokens")]
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


_GS = sorted({r["G"] for r in rows if r["mode"] == "cyclic"}, reverse=True)[:3]
SERIES = [(G, f"{L // G} layer{'s' if L // G > 1 else ''}/group (G={G})", c) for G, c in zip(_GS, (S1, S2, S3))]


def cyc(n, G):
    return sorted((r for r in rows if r["mode"] == "cyclic" and r["n_prompt"] == n and r["G"] == G),
                  key=lambda r: r["K"])


# 1) decode latency vs resident layers
fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURF)
for G, lab, c in SERIES:
    pts = cyc(512, G)
    ax.plot([p["K"] for p in pts], [p["tpot_med"] for p in pts], "-o", color=c, lw=2, ms=6,
            mec=SURF, mew=2, label=lab)
b = base[512]["tpot_med"]
yt = ax.get_yaxis_transform()  # x in axes fraction, y in data units: keeps labels inside the axes
ax.axhline(b, color=INK2, lw=1.2, ls="--")
ax.text(0.98, b, f"baseline, all {L} layers resident: {b:.0f} ms", color=INK2, fontsize=8, ha="right",
        va="bottom", transform=yt)
floor = MODEL_GB * 1e3 / DISK_MAX
ax.axhline(floor, color=INK2, lw=1.2, ls=":")
ax.text(0.98, floor, f"NVMe floor: {MODEL_GB:.2f} GB / {DISK_MAX:.1f} GB/s = {floor:.0f} ms per token", color=INK2,
        fontsize=8, ha="right", va="top", transform=yt)
ax.set_xlim(0, max(20, L * 0.55))
ax.set_ylim(0, None)
style(ax, "Decode: per-token latency vs number of pods (prompt 512)", "pods (K)",
      "per-token latency (ms, median)")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
fig.tight_layout()
fig.savefig(f"{R}/decode_latency.png", dpi=150)

# 2) TTFT ratio small multiples
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), facecolor=SURF, sharey=True)
for ax, n in zip(axs, (128, 512, 2048)):
    for G, lab, c in SERIES:
        pts = cyc(n, G)
        ax.plot([p["K"] for p in pts], [p["ttft_x"] for p in pts], "-o", color=c, lw=2, ms=6,
                mec=SURF, mew=2, label=lab)
    ax.axhline(1.0, color=INK2, lw=1.2, ls="--")
    ax.axhspan(1.0, 1.05, color=GRID, lw=0)
    style(ax, f"prompt {n} tokens (baseline {base[n]['ttft_med']:.0f} ms)", "pods (K)",
          "TTFT / baseline TTFT" if n == 128 else "")
axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle("Prefill: time-to-first-token relative to all-resident baseline (shaded band = within 5%)",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{R}/ttft_ratio.png", dpi=150)

# 3) decode time breakdown, G=36
pts = [r for r in rows if r["n_prompt"] == 512 and r["G"] == L] + [base[512]]
pts.sort(key=lambda r: (r["mode"] == "resident", r["K"]))
labels = [("baseline" if r["G"] == 1 else f"{r['mode'][:3]} K={r['K']}") for r in pts]
fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURF)
bottom = [0] * len(pts)
for key, lab, c in (("dec_compute", "compute", S1), ("dec_hop", "pod-to-pod hops", S2),
                    ("dec_wait", "waiting for weights (load not hidden)", S3)):
    v = [r[key] for r in pts]
    ax.bar(labels, v, bottom=bottom, color=c, label=lab, width=0.6, edgecolor=SURF, linewidth=2)
    bottom = [a + b for a, b in zip(bottom, v)]
style(ax, f"Where decode time goes (G={L}, prompt 512, per token, median)", "", "ms per token")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(f"{R}/decode_breakdown.png", dpi=150)

# 4) memory vs pods (G=36, prompt 128 = closest to steady state; longer prompts add transient activations)
pts = cyc(128, L)
bm = base[128]
fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURF)
ks = [p["K"] for p in pts]
wg = [p["weights_peak_mb"] * 2**20 / 1e9 for p in pts]
rg = [p["stage_rss_mb"] * 2**20 / 1e9 for p in pts]
ax.plot(ks, wg, "-o", color=S1, lw=2, ms=6, mec=SURF, mew=2, label="weights in memory, all pods together")
ax.plot(ks, rg, "-o", color=S2, lw=2, ms=6, mec=SURF, mew=2, label="total stage-pod RAM (weights + per-pod runtime + KV cache)")
for k, w in zip(ks, wg):
    ax.annotate(f"−{100 * (1 - w / (bm['weights_peak_mb'] * 2**20 / 1e9)):.0f}%", (k, w), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=7.5, color=INK2)
yt = ax.get_yaxis_transform()
for val, lab, va in ((bm["weights_peak_mb"], "baseline weights, one pod", "top"), (bm["stage_rss_mb"], "baseline stage-pod RAM", "bottom")):
    ax.axhline(val * 2**20 / 1e9, color=INK2, lw=1.2, ls="--")
    ax.text(0.98, val * 2**20 / 1e9, f"{lab}: {val * 2**20 / 1e9:.2f} GB", color=INK2, fontsize=8, ha="right", va=va,
            transform=yt)
ax.set_xlim(0, 20)
ax.set_ylim(0, None)
ax.set_xticks(ks)
ax.set_ylim(0, max(max(rg), bm["stage_rss_mb"] * 2**20 / 1e9) * 1.15)
style(ax, f"Memory vs number of pods (1 layer per pod in memory, G={L}; coordinator pod excluded)", "pods (K)", "GB (decimal)")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")
fig.tight_layout()
fig.savefig(f"{R}/memory.png", dpi=150)

# markdown tables
print("| G | K | mode | layers resident | weights GB | stage RSS GB | prompt | TTFT ms | TTFT x | "
      "tok ms | tok x | compute | hops | load-wait | tokens = baseline |")
print("|" + "---|" * 15)
for r in rows:
    print(f"| {r['G']} | {r['K']} | {r['mode']} | {r['resident_layers']} | {r['weights_peak_mb'] * 2**20 / 1e9:.2f} | "
          f"{r['stage_rss_mb'] * 2**20 / 1e9:.2f} | {r['n_prompt']} | {r['ttft_med']:.0f} | {r['ttft_x']:.2f} | "
          f"{r['tpot_med']:.0f} | {r['tpot_x']:.2f} | {r['dec_compute']:.0f} | {r['dec_hop']:.0f} | "
          f"{r['dec_wait']:.0f} | {'yes' if r['tokens_match'] else 'NO'} |")
