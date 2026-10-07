"""Check that every run produced the same tokens as the baseline of its round (greedy decoding, bf16).

    python3 scripts/validate_output_identity.py

For each data file, every non-baseline run's generated tokens are compared with the first baseline run for the same
prompt. A run counts as identical when the shorter sequence is a prefix of the longer one (some runs generate fewer
tokens). Divergences are listed with the generated-token position where they start. Standard library only.

Expected result (docs/REPORT.md §7.1): every streaming, partial-residency, kernel-paging and control run is identical;
the only divergences are speculative or batched runs, at eight near-tie positions.
"""
import json, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)


def jl(path):
    return [json.loads(l) for l in open(path)]


def toks(r):
    t = r["tokens"]
    return t[0] if t and isinstance(t[0], list) else t   # batched runs store one list per request; request 0 = single-user prompt


def check(rows, is_base, key):
    base, same, where = {}, 0, Counter()
    for r in rows:
        if is_base(r) and key(r) not in base:
            base[key(r)] = toks(r)
    n_other = 0
    for r in rows:
        if is_base(r):
            continue
        n_other += 1
        b, t = base[key(r)], toks(r)
        n = min(len(b), len(t))
        if t[:n] == b[:n]:
            same += 1
        else:
            d = next(i for i in range(n) if t[i] != b[i])
            where[(r.get("approach") or "cyclic", str(key(r)[1] or key(r)[0]).split("/")[-1], d + 1)] += 1
    return same, n_other, where


exp_key = lambda r: (str(r.get("n_prompt")), r.get("PROMPT_FILE"))
e1_base = lambda r: r["G"] == 1 and r["mode"] == "resident"
appr_base = lambda r: r["approach"] == "baseline"
CHECKS = [
    ("3B experiment 1: cyclic streaming + controls", R("qwen2.5-3b", "results.jsonl"), e1_base, exp_key),
    ("3B experiment 2: alternatives", R("qwen2.5-3b", "results2.jsonl"), appr_base, exp_key),
    ("3B single-pod streaming", R("qwen2.5-3b", "results3.jsonl"), appr_base, exp_key),
    ("gpt-oss experiment 1", R("gpt-oss-20b", "results.jsonl"), e1_base, exp_key),
    ("gpt-oss experiment 2", R("gpt-oss-20b", "results2.jsonl"), appr_base, exp_key),
] + [(f"optimisation round {i}", R("qwen2.5-3b", f"results{n}.jsonl"), appr_base, exp_key) for i, n in ((1, 4), (2, 5), (3, 6), (4, 7))] + \
    [(f"six models: {m}", R("six-models", f"{m}.jsonl"), appr_base, exp_key)
     for m in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "gpt-oss-20b", "qwen3-30b-a3b")] + \
    [("degenerate-prompt recheck", R("six-models", "gptoss_paging_recheck.jsonl"), appr_base, lambda r: ("512", None))]

total_same = total = 0
print("| experiment | identical | runs | divergences (approach, prompt, generated token: count) |")
print("|---|---|---|---|")
for name, path, is_base, key in CHECKS:
    same, n, where = check(jl(path), is_base, key)
    total_same += same
    total += n
    print(f"| {name} | {same} | {n} | {'; '.join(f'{a}, {p}, token {d}: {c}' for (a, p, d), c in sorted(where.items())) or '—'} |")

# kernel-paging replication: compared with the six-model baselines of the same model and prompt
base = {}
for m in ("gpt-oss-20b", "qwen3-30b-a3b"):
    for r in jl(R("six-models", f"{m}.jsonl")):
        if r["approach"] == "baseline":
            base.setdefault((m, r["PROMPT_FILE"]), toks(r))
same = n = 0
for f in ("gpt-oss-20b_paging2.jsonl", "qwen3-30b-a3b_paging2.jsonl"):
    for r in jl(R("six-models", f)):
        b, t = base[(r["model"], r["PROMPT_FILE"])], toks(r)
        k = min(len(b), len(t))
        n += 1
        same += t[:k] == b[:k]
total_same += same
total += n
print(f"| kernel-paging replication | {same} | {n} | {'—' if same == n else 'see data'} |")
print(f"\nTotal: {total_same} of {total} non-baseline runs identical to their baseline; {total - total_same} diverge.")
