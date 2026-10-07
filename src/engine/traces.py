"""Where does the time go? Per-layer trace breakdown for prefill and decode.

    python traces.py /exp/results_gptoss/results.jsonl 24 1646375608
"""
import json, statistics as st, sys

path, L, layer_bytes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(path)]
cyc_ks = sorted({r["K"] for r in rows if r["mode"] == "cyclic" and r["G"] == L})


def tr(G, K, mode, n):
    return next(r for r in rows if (r["G"], r["K"], r["mode"], r["n_prompt"]) == (G, K, mode, n) and r["traces"])


print("=== prefill (first pass), 2048-token prompt: sums over layers, ms")
cfgs = [(1, 1, "resident")] + [(L, k, "cyclic") for k in cyc_ks] + [(L, k, "resident") for k in sorted({r["K"] for r in rows if r["mode"] == "resident" and r["G"] == L})]
for G, K, mode in cfgs:
    try:
        r = tr(G, K, mode, 2048)
    except StopIteration:
        continue
    p = r["traces"][0]
    wait = sum(t["start"] - t["arr"] for t in p) * 1e3
    comp = sum(t["end"] - t["start"] for t in p) * 1e3
    hops = sum(b["arr"] - a["end"] for a, b in zip(p, p[1:])) * 1e3
    loads = [t["load_ms"] for t in p if t["load_ms"]]
    print(f"G={G:<2} K={K:<2} {mode:8s}: ttft={r['ttft_ms']:6.0f}  wait={wait:6.0f}  compute={comp:6.0f} ({comp / len(p):.0f}/layer)  "
          f"hops={hops:4.0f}  load/layer={st.median(loads) if loads else 0:5.0f}")
print("=== decode, 512-token prompt: per token = wait + compute + hops (medians over tokens); load per layer; implied disk rate")
for K in cyc_ks:
    r = tr(L, K, "cyclic", 512)
    dec = r["traces"][1:]
    w = st.median([sum(t["start"] - t["arr"] for t in tok) * 1e3 for tok in dec])
    c = st.median([sum(t["end"] - t["start"] for t in tok) * 1e3 for tok in dec])
    h = st.median([sum(b["arr"] - a["end"] for a, b in zip(tok, tok[1:])) * 1e3 for tok in dec])
    ld = st.median([t["load_ms"] for tok in dec for t in tok if t["load_ms"]])
    tp = st.median(r["tpot_ms"])
    print(f"K={K:<2}: per token {tp:6.0f} ms = wait {w:6.0f} + compute {c:4.0f} + hops {h:3.0f};  load/layer {ld:5.0f} ms;  "
          f"{L * layer_bytes / (tp / 1e3) / 1e9:.1f} GB/s")
b = tr(1, 1, "resident", 512)
print("baseline: compute per token", round(st.median([sum(t["end"] - t["start"] for t in tok) * 1e3 for tok in b["traces"][1:]])),
      "ms; per token", round(st.median(b["tpot_ms"])), "ms")
