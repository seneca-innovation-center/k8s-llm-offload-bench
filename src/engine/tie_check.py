"""fp32 logit margin at the first token where a run diverged from the baseline (is it a bf16 near-tie?).

    python tie_check.py <hf model dir> <results.jsonl> <PROMPT_FILE value> [CHAT]
Finds the baseline tokens for that prompt, the first divergent position among the other runs, and prints the
fp32 top-3 logits of the full model at that position.
"""
import json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16)
src, results, pf = sys.argv[1], sys.argv[2], sys.argv[3]
chat = len(sys.argv) > 4 and sys.argv[4] == "CHAT"
W = os.environ.get("WEIGHTS", "/exp/weights")
tok = AutoTokenizer.from_pretrained(W)
text = open(os.path.join("/exp", pf)).read()
ids = (tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True) if chat
       else tok(text)["input_ids"])
rows = [json.loads(l) for l in open(results) if json.loads(l).get("PROMPT_FILE") == pf]
base = next(r for r in rows if r["approach"] == "baseline")["tokens"]
divs = {}
for r in rows:
    if r["approach"] == "baseline":
        continue
    t = r["tokens"]
    d = next((i for i in range(min(len(base), len(t))) if base[i] != t[i]), None)
    if d is not None:
        divs.setdefault(d, set()).add((r["approach"], r.get("TREE") or r.get("GAMMA")))
if not divs:
    print("no divergence from the baseline in", results, "for", pf)
    sys.exit()
d = min(divs)
print(f"first divergence at generated token {d + 1}; configs: {sorted(divs[d])}")
n = int(rows[0]["n_prompt"]) if rows[0]["n_prompt"] else len(ids)
prefix = ids[:n] + base[:d]
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
with torch.inference_mode():
    lg = m(torch.tensor([prefix])).logits[0, -1]
top = torch.topk(lg, 3)
print("fp32 top-3 ids", top.indices.tolist(), "logits", [round(v, 3) for v in top.values.tolist()],
      "tokens", [repr(tok.decode([i])) for i in top.indices.tolist()])
print(f"margin top1-top2 = {(top.values[0] - top.values[1]).item():.4f} ({100 * (top.values[0] - top.values[1]).item() / top.values[0].item():.2f}% of top1)")
print("baseline picked", base[d], repr(tok.decode([base[d]])), "; divergent runs picked",
      sorted({r["tokens"][d] for r in rows if r["approach"] != "baseline" and len(r["tokens"]) > d and r["tokens"][d] != base[d]}))
