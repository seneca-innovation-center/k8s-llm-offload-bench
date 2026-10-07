"""results/results2.jsonl (+ cyclic rows of results.jsonl) -> comparison tables (markdown, stdout) and charts."""
import json, os, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("R", "/exp/results")
GB = lambda b: b / 1e9
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
L = int(os.environ.get("L", "36"))
EXTRA_KEYS = ("RESIDENT", "SLOTS", "LOAD_MODE", "MEM_LIMIT", "DRAFT", "GAMMA", "TREE", "BATCH", "MAXLEN", "PROMPT_FILE", "CHAT")

recs = [json.loads(l) for l in open(f"{R}/results2.jsonl")]
if os.path.exists(f"{R}/results3.jsonl"):
    recs += [json.loads(l) for l in open(f"{R}/results3.jsonl") if json.loads(l)["approach"] != "baseline"]
for l in open(f"{R}/results.jsonl"):  # first sweep: plain cyclic (G=36) rows, default prompt
    r = json.loads(l)
    if r["mode"] == "cyclic" and r["G"] == L:
        r["approach"] = "cyclic"
        recs.append(r)


def key(r):
    return (r["approach"], r["G"], r["K"], r["mode"]) + tuple(r.get(k) for k in EXTRA_KEYS) + (r["n_prompt"],)


groups = defaultdict(list)
for r in recs:
    groups[key(r)].append(r)


def describe(r):
    """(label, short tag) for a config."""
    a = r["approach"]
    if a == "baseline":
        return "baseline: all 36 layers in one pod", "baseline"
    if a == "cyclic":
        return f"cyclic: {r['K']} pods, 1 layer each", f"K={r['K']}"
    if a == "seq1pod":
        return f"single pod: all {L} layers streamed in order through {r['SLOTS']} slot{'s' if int(r['SLOTS']) > 1 else ''}", f"S={r['SLOTS']}"
    if a == "partial":
        hi = int(r["RESIDENT"].split("-")[1])
        streamed, slots = L - hi - 1, int(r["SLOTS"])
        return (f"partial: {hi + 1} layers pinned, {streamed} streamed through {slots} slots (−{streamed - slots} layers)",
                f"−{streamed - slots} layers")
    if a == "mmap":
        return f"mmap + cgroup limit {r['MEM_LIMIT']}", r["MEM_LIMIT"]
    if a in ("spec", "tree"):
        where = "all-resident pod" if r["G"] == 1 else (f"single pod, {r['SLOTS']} slots" if r["K"] == 1 else f"cyclic {r['K']} pods")
        if a == "tree":
            return f"tree speculation {r['TREE']} on {where}", f"tree{sum(int(w) for w in r['TREE'].split(','))}"
        return f"speculative γ={r['GAMMA']} on {where}", ("base" if r["G"] == 1 else f"K={r['K']}") + f" γ{r['GAMMA']}"
    if a == "batch":
        return f"batch {r['BATCH']}: " + ("baseline" if r["G"] == 1 else f"cyclic {r['K']} pods"), f"B={r['BATCH']}"
    return a, a


def per_token(rs):
    """Per-token latency in ms. Speculative runs: total decode time / tokens (a cycle emits several tokens)."""
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


rows = []
for k, rs in groups.items():
    r0 = rs[0]
    s0 = next(r for r in rs if r["stats"])
    stg, co = s0["stats"]["stages"], s0["stats"]["coordinator"]
    label, tag = describe(r0)
    natural = bool(r0.get("PROMPT_FILE"))
    if natural:
        label += " [non-repeating prompt]"
    mapped = r0.get("LOAD_MODE") == "mmap"
    # first-sweep records predate the head_bytes field: embed_head.safetensors is 622,334,152 bytes in every run
    weights = None if mapped else sum(s["peak"] for s in stg) + co.get("head_bytes", 622334152) + co.get("draft_bytes", 0)
    row = dict(approach=r0["approach"], G=r0["G"], K=r0["K"], n_prompt=r0["n_prompt"], natural=natural, label=label,
               tag=tag, reps=len(rs), batch=r0.get("batch", 1), ttft=st.median([r["ttft_ms"] for r in rs]),
               ttft_range=(min(r["ttft_ms"] for r in rs), max(r["ttft_ms"] for r in rs)), tpot=per_token(rs),
               stage_rss=GB(sum(s["VmHWM"] for s in stg)), coord_rss=GB(co["VmHWM"]),
               weights=GB(weights) if weights else None, tokens=r0["tokens"], all_tokens=[r["tokens"] for r in rs],
               cg_peak=GB(sum(s.get("cg_peak") or 0 for s in stg)) if mapped else None,
               cg_max=(stg[0].get("cg_max") if mapped else None), t=min(r["t"] for r in rs))
    row["total_rss"] = row["stage_rss"] + row["coord_rss"]
    if r0.get("spec"):
        acc = [x for r in rs for x in r["spec"]["accepted"]]
        row["accepted"] = st.mean(acc)
        row["cycle_ms"] = st.median([x for r in rs for x in r["spec"]["cycle_ms"]])
        row["draft_ms"] = st.median([x for r in rs for x in r["spec"]["draft_ms"]])
        row["verify_ms"] = st.median([x for r in rs for x in r["spec"]["verify_ms"]])
        row["tokens_per_cycle"] = row["accepted"] + 1  # each cycle emits the accepted drafts + one big-model token
    rows.append(row)

base = {(r["n_prompt"], r["natural"]): r for r in rows if r["approach"] == "baseline"}
for r in rows:
    b = base[(r["n_prompt"], r["natural"])]
    r["ttft_x"], r["tpot_x"] = r["ttft"] / b["ttft"], r["tpot"] / b["tpot"]
    r["mem_x"] = r["total_rss"] / b["total_rss"]
    bt = b["tokens"]
    n = min(len(bt), len(r["tokens"]))
    div = next((i for i in range(n) if bt[i] != r["tokens"][i]), None)
    r["match"] = "identical" if div is None and all(t[:n] == bt[:n] for t in r["all_tokens"]) else f"differs from token {div + 1}"
    if r["approach"] == "batch":  # batched baseline at the same B is the fair reference for per-user latency
        bb = next((x for x in rows if x["approach"] == "batch" and x["G"] == 1 and x["batch"] == r["batch"]), None)
        r["tpot_x_batched"] = r["tpot"] / bb["tpot"] if bb else None

order = {"baseline": 0, "cyclic": 1, "seq1pod": 2, "partial": 3, "mmap": 4, "spec": 5, "tree": 6, "batch": 7}
rows.sort(key=lambda r: (order[r["approach"]], r["natural"], r["n_prompt"], r["G"] != 1, r["K"], r["batch"], r["label"]))


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


def pick(approach, n, natural=False, **kw):
    return [r for r in rows if r["approach"] == approach and r["n_prompt"] == n and r["natural"] == natural
            and all(r.get(k) == v for k, v in kw.items())]


# 1) frontier: total RAM vs per-token latency, single user, 512-token prompt (default prompt)
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURF)
_ks = sorted({r["K"] for r in rows if r["approach"] == "cyclic"})
series = [("cyclic", f"cyclic streaming ({_ks[0]}–{_ks[-1]} pods)", S1, "o"), ("seq1pod", "single-pod sequential streaming", S5, "P"),
          ("partial", "partial residency (1 pod)", S2, "s"), ("mmap", "mmap + memory limit (1 pod)", S3, "D"),
          ("spec", "speculative decoding (γ=5)", S4, "^")]
b = base[(512, False)]
spec_natural = bool(pick("spec", 512, natural=True))  # prefer the non-repeating prompt for speculative points
for a, lab, c, mk in series:
    pts = pick(a, 512, natural=(a == "spec" and spec_natural))
    pts = [r for r in pts if not (a == "spec" and r["tag"].endswith("γ8"))]
    if a == "spec" and spec_natural:
        lab += ", non-repeating prompt"
    if not pts:
        continue
    ax.scatter([p["total_rss"] for p in pts], [p["tpot_x"] for p in pts], s=70, color=c, marker=mk, label=lab,
               edgecolors=SURF, linewidths=1.5, zorder=3)
    for p in pts:
        ax.annotate(p["tag"], (p["total_rss"], p["tpot_x"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK2)
ax.scatter([b["total_rss"]], [1.0], s=110, color=INK, marker="*", label="baseline", zorder=4)
ax.axhline(1.0, color=INK2, lw=1, ls="--")
ax.set_yscale("log")
_ymax = max(r["tpot_x"] for r in rows if r["n_prompt"] == 512 and r["approach"] != "batch")
_ticks = [t for t in (0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 30, 50, 80) if t <= _ymax * 1.25]
ax.set_yticks(_ticks)
ax.set_yticklabels([f"{t:g}×" for t in _ticks])
ax.minorticks_off()
ax.set_xlim(0, None)
style(ax, "Memory vs per-token latency, single user, 512-token prompt (all pods incl. coordinator)",
      "total pod RAM (GB, high-water mark)", "per-token latency ÷ baseline")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
fig.tight_layout()
fig.savefig(f"{R}/frontier.png", dpi=150)

# 2) batching: per-user latency and throughput vs batch size (128-token prompt)
fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
for (G, K), lab, c in (((1, 1), "baseline (all resident)", INK), ((L, 4), "cyclic, 4 pods", S1), ((L, 2), "cyclic, 2 pods", S2)):
    pts = sorted([r for r in rows if r["n_prompt"] == 128 and not r["natural"] and r["G"] == G and r["K"] == K
                  and r["approach"] in ("baseline", "cyclic", "batch")], key=lambda r: r["batch"])
    axs[0].plot([p["batch"] for p in pts], [p["tpot"] for p in pts], "-o", color=c, lw=2, ms=6, mec=SURF, mew=2, label=lab)
    axs[1].plot([p["batch"] for p in pts], [p["batch"] * 1000 / p["tpot"] for p in pts], "-o", color=c, lw=2, ms=6, mec=SURF, mew=2, label=lab)
for ax in axs:
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 16, 64, 256, 512])
    ax.set_xticklabels(["1", "16", "64", "256", "512"])
axs[0].set_yscale("log")
_bmax = max(r["tpot"] for r in rows if r["approach"] in ("batch", "baseline", "cyclic") and r["n_prompt"] == 128)
_bt = [t for t in (100, 200, 500, 1000, 2000, 4000, 8000, 16000) if t <= _bmax * 1.3]
axs[0].set_yticks(_bt)
axs[0].set_yticklabels([str(t) for t in _bt])
axs[0].minorticks_off()
style(axs[0], "Per-user latency per token", "concurrent requests (batch)", "ms per token, each user")
style(axs[1], "Throughput", "concurrent requests (batch)", "tokens per second, all users")
axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.tight_layout()
fig.savefig(f"{R}/batching.png", dpi=150)

# 3) speculative decoding: plain vs speculative, 128-token prompt (unique text) and 512-token non-repeating prompt
HAS_SPEC = any(r["approach"] == "spec" for r in rows)
panels = [(128, False, "128-token prompt"), (512, True, "512-token non-repeating prompt")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF, sharey=True)
for ax, (n, nat, title) in (zip(axs, panels) if HAS_SPEC else []):
    cfgs = [((1, 1), "all resident"), ((36, 2), "cyclic 2 pods"), ((36, 4), "cyclic 4 pods")]
    plain, specv, labs, notes = [], [], [], []
    for (G, K), lab in cfgs:
        p = next((r for r in rows if r["n_prompt"] == n and r["natural"] == nat and r["G"] == G and r["K"] == K
                  and r["approach"] in ("baseline", "cyclic")), None)
        s = next((r for r in rows if r["n_prompt"] == n and r["natural"] == nat and r["G"] == G and r["K"] == K
                  and r["approach"] == "spec" and r["tag"].endswith("γ5")), None)
        if p and s:
            plain.append(p["tpot"]); specv.append(s["tpot"]); labs.append(lab)
            notes.append(f"{s['tokens_per_cycle']:.1f} tok/cycle")
    xs = range(len(labs))
    ax.bar([x - 0.2 for x in xs], plain, 0.38, color=S1, label="plain greedy", edgecolor=SURF)
    ax.bar([x + 0.2 for x in xs], specv, 0.38, color=S4, label="speculative, γ=5, Qwen2.5-0.5B draft", edgecolor=SURF)
    for x, v, note in zip(xs, specv, notes):
        ax.annotate(note, (x + 0.2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labs)
    style(ax, title, "", "ms per token" if n == 128 else "")
if HAS_SPEC:
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle("Speculative decoding: per-token latency (single user)", color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout()
if HAS_SPEC:
    fig.savefig(f"{R}/speculative.png", dpi=150)

# ---- markdown tables ----
print("## Single user\n")
print("| approach / config | prompt | weights in RAM (GB) | stage-pod RAM (GB) | coordinator RAM (GB) | total RAM (GB) | vs baseline | TTFT (s) | TTFT × | per token (ms) | per token × | output |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["approach"] == "batch":
        continue
    w = f"{r['weights']:.2f}" if r["weights"] else f"mapped; cgroup peak {r['cg_peak']:.2f}"
    print(f"| {r['label']} | {r['n_prompt']} | {w} | {r['stage_rss']:.2f} | {r['coord_rss']:.2f} | {r['total_rss']:.2f} | "
          f"{100 * (r['mem_x'] - 1):+.0f}% | {r['ttft'] / 1e3:.2f} | {r['ttft_x']:.2f} | {r['tpot']:.0f} | {r['tpot_x']:.2f} | {r['match']} |")
print("\n## Speculative decoding details (medians per cycle)\n")
print("| config | prompt | accepted drafts / cycle | tokens / cycle | draft (ms) | verify (ms) | cycle (ms) | per token (ms) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["approach"] == "spec":
        print(f"| {r['label']} | {r['n_prompt']} | {r['accepted']:.2f} | {r['tokens_per_cycle']:.2f} | {r['draft_ms']:.0f} | "
              f"{r['verify_ms']:.0f} | {r['cycle_ms']:.0f} | {r['tpot']:.0f} |")
print("\n## Batching (128-token prompt)\n")
print("| config | batch | total RAM (GB) | TTFT (s) | per-user ms/token | vs baseline at same batch | tokens/s | output (row 0) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["approach"] == "batch" or (r["n_prompt"] == 128 and not r["natural"] and r["approach"] in ("baseline", "cyclic") and r["K"] in (1, 2, 4)):
        bx = r.get("tpot_x_batched") or r["tpot_x"]
        print(f"| {r['label']} | {r['batch']} | {r['total_rss']:.2f} | {r['ttft'] / 1e3:.2f} | {r['tpot']:.0f} | "
              f"{bx:.2f}× | {r['batch'] * 1000 / r['tpot']:.0f} | {r['match']} |")
json.dump([{k: v for k, v in r.items() if k not in ("tokens", "all_tokens")} for r in rows], open(f"{R}/summary2.json", "w"), indent=1)
