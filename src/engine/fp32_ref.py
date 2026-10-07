"""Low-memory fp32 reference at every position where a run diverged from the baseline.

    python fp32_ref.py <weights dir> <results.jsonl> <PROMPT_FILE> [CHAT]

Streams the model layer by layer in fp32 through our own layer code (verified against transformers by
layer_check.py), so a 14B or 30B model needs a few GB, not the full fp32 copy. Prints the fp32 top-3 logits
and margin at each divergence position, and which choice (baseline or divergent run) the fp32 model agrees with.
"""
import json, os, sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from common import Cfg, layer_forward, load_cold, rms, rope

torch.set_num_threads(int(os.environ.get("THREADS", "16")))
W, results, pf = sys.argv[1], sys.argv[2], sys.argv[3]
chat = len(sys.argv) > 4 and sys.argv[4] == "CHAT"
cfg = Cfg(f"{W}/config.json")
tok = AutoTokenizer.from_pretrained(W)
text = open(os.path.join(os.environ.get("PROMPT_ROOT", "/exp"), pf)).read()
ids = (tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True) if chat
       else tok(text)["input_ids"])
rows = [r for r in (json.loads(l) for l in open(results)) if r.get("PROMPT_FILE") == pf]
base = next(r for r in rows if r["approach"] == "baseline")["tokens"]
divs = {}   # position -> {(approach, detail, picked token)}
for r in rows:
    if r["approach"] == "baseline":
        continue
    t = r["tokens"]
    d = next((i for i in range(min(len(base), len(t))) if base[i] != t[i]), None)
    if d is not None:
        divs.setdefault(d, set()).add((r["approach"], r.get("TREE") or r.get("GAMMA") or r.get("MEM_LIMIT") or "", t[d]))
if not divs:
    print(f"{os.path.basename(W)} {pf}: no divergence from the baseline")
    sys.exit()
n = int(rows[0]["n_prompt"]) if rows[0]["n_prompt"] else len(ids)
D = max(divs)
prefix = ids[:n] + base[:D]
S = len(prefix)
EVICT = os.environ.get("EVICT", "0") == "1"   # EVICT=1: fadvise DONTNEED, so the check leaves no trace in the page cache
head, _ = load_cold(f"{W}/embed_head.safetensors", evict=EVICT)
x = head["model.embed_tokens.weight"][torch.tensor([prefix])].float()
cos, sin = rope(cfg, 0, S, torch.float32)
with torch.inference_mode():
    for l in range(cfg.L):
        Wl, _ = load_cold(f"{W}/layer_{l:02d}.safetensors", evict=EVICT)
        Wf = {k: v.float() for k, v in Wl.items()}
        del Wl
        kv = (torch.zeros(1, cfg.nkv, S, cfg.hd), torch.zeros(1, cfg.nkv, S, cfg.hd))
        x = layer_forward(cfg, Wf, x, cos, sin, kv, 0, l)
        del Wf, kv
    h = rms(cfg, x[0], head["model.norm.weight"].float())
    LM = head.get("lm_head.weight", head["model.embed_tokens.weight"])
    want = sorted(divs)
    pos = torch.tensor([n + d - 1 for d in want])
    logits = F.linear(h[pos], LM.float())
for i, d in enumerate(want):
    top = torch.topk(logits[i], 3)
    ti, tv = top.indices.tolist(), top.values.tolist()
    margin = tv[0] - tv[1]
    others = sorted({p for _, _, p in divs[d]})
    agree = "baseline" if ti[0] == base[d] else ("divergent run" if ti[0] in others else "neither")
    print(f"{os.path.basename(W)} {pf}: generated token {d + 1}; runs {sorted((a, s) for a, s, _ in divs[d])}")
    print(f"  fp32 top-3 {ti} logits {[round(v, 3) for v in tv]} tokens {[repr(tok.decode([t])) for t in ti]}")
    print(f"  margin top1-top2 = {margin:.4f} ({100 * margin / abs(tv[0]):.2f}% of top1); baseline picked {base[d]} {tok.decode([base[d]])!r}, "
          f"divergent picked {others} {[tok.decode([o]) for o in others]}; fp32 agrees with: {agree}")
