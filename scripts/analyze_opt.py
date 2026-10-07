"""Optimisation rounds: per-prompt comparison of baseline / speculative / tree / direct-I/O configurations.

    python3 analyze_opt.py results/results4.jsonl results/results5.jsonl ...   (markdown to stdout)
"""
import json, statistics as st, sys
from collections import defaultdict

KEYS = ("approach", "G", "K", "mode", "SLOTS", "LOAD_MODE", "GAMMA", "TREE", "RESIDENT", "PROMPT_FILE", "CHAT", "n_prompt")
GB = lambda b: b / 1e9

rows = defaultdict(list)
for path in sys.argv[1:]:
    for l in open(path):
        r = json.loads(l)
        rows[tuple(r.get(k) for k in KEYS)].append(r)


def per_token(rs):
    if rs[0].get("spec"):
        return sum(r["spec"]["decode_ms_total"] for r in rs) / sum(r["spec"]["tokens_out"] - 1 for r in rs)
    return st.median([x for r in rs for x in r["tpot_ms"]])


def label(k):
    d = dict(zip(KEYS, k))
    a = d["approach"]
    where = "all resident" if d["G"] == 1 else (f"single pod, {d['SLOTS']} slots" if d["K"] == 1 else f"{d['K']} pods")
    if d.get("RESIDENT"):
        where = f"pinned {d['RESIDENT']}, {d['SLOTS']} slots"
    io = " · direct I/O" if d.get("LOAD_MODE") == "direct" else ""
    if a == "spec":
        return f"chain γ={d['GAMMA']} · {where}{io}"
    if a == "tree":
        return f"tree {d['TREE']} ({sum(int(w) for w in d['TREE'].split(','))} nodes) · {where}{io}"
    return f"{a} · {where}{io}"


out = defaultdict(list)
for k, rs in rows.items():
    d = dict(zip(KEYS, k))
    prompt = (d.get("PROMPT_FILE") or "prompt.txt").split("/")[-1] + (f" ({d['n_prompt']} tok)" if d["n_prompt"] else " (full)")
    s0 = next(r for r in rs if r["stats"])
    stg, co = s0["stats"]["stages"], s0["stats"]["coordinator"]
    rec = dict(label=label(k), approach=d["approach"], reps=len(rs), ttft=st.median([r["ttft_ms"] for r in rs]), tpot=per_token(rs),
               ram=GB(sum(x["VmHWM"] for x in stg) + co["VmHWM"]), tokens=rs[0]["tokens"], all_tokens=[r["tokens"] for r in rs])
    if rs[0].get("spec"):
        sp = [r["spec"] for r in rs]
        acc = [x for s_ in sp for x in s_["accepted"]]
        rec.update(accepted=st.mean(acc), tpc=st.mean(acc) + 1, draft=st.median([x for s_ in sp for x in s_["draft_ms"]]),
                   verify=st.median([x for s_ in sp for x in s_["verify_ms"]]), cycle=st.median([x for s_ in sp for x in s_["cycle_ms"]]),
                   nodes=st.mean([x for s_ in sp for x in s_.get("nodes", [s_["gamma"] + 1])]))
    out[prompt].append(rec)

for prompt, recs in out.items():
    base = next((r for r in recs if r["approach"] == "baseline"), None)
    print(f"\n### {prompt}\n")
    print("| configuration | total RAM | per token | vs baseline | first token | tokens/cycle | nodes | draft ms | verify ms | cycle ms | output |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    recs.sort(key=lambda r: (r["approach"] != "baseline", r["tpot"]))
    for r in recs:
        if base:
            bt = base["tokens"]
            ok = sum(1 for t in r["all_tokens"] if t[:len(bt)] == bt[:len(t)])
            same = "identical" if ok == len(r["all_tokens"]) else f"differs ({len(r['all_tokens']) - ok}/{len(r['all_tokens'])} reps)"
        else:
            same = "—"
        x = f"{r['tpot'] / base['tpot']:.2f}×" if base else ""
        tt = f"{r['ttft'] / 1e3:.2f} s" + (f" ({r['ttft'] / base['ttft']:.2f}×)" if base else "")
        sp = (f"{r['tpc']:.2f} | {r['nodes']:.0f} | {r['draft']:.0f} | {r['verify']:.0f} | {r['cycle']:.0f}" if "tpc" in r else " | | | | ")
        print(f"| {r['label']} | {r['ram']:.2f} GB | {r['tpot']:.0f} ms | {x} | {tt} | {sp} | {same} |")
