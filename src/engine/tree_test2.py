"""Exactness test of tree speculation with the lean DraftModel (fp32 target via our layer code, draft = same 0.5B
through DraftModel with logit noise to force rejections). python3 tree_test2.py <layer files of Qwen2.5-0.5B-Instruct>"""
import sys, time, torch
import torch.nn.functional as F
from common import Cfg, ancestor_mask, layer_forward, load_cold, rms, rope, rope_at
from draft_model import DraftModel
from transformers import AutoTokenizer

torch.manual_seed(0)
W = sys.argv[1]
cfg = Cfg(f"{W}/config.json"); tok = AutoTokenizer.from_pretrained(W)
head, _ = load_cold(f"{W}/embed_head.safetensors", evict=False)
EMB = head["model.embed_tokens.weight"].float(); LM = head.get("lm_head.weight", EMB).float(); NORM = head["model.norm.weight"].float()
LAYERS = [{k: v.float() for k, v in load_cold(f"{W}/layer_{l:02d}.safetensors", evict=False)[0].items()} for l in range(cfg.L)]
ids = tok(open("prompt_long.txt").read())["input_ids"][:96]

class Pipeline:
    def __init__(self): self.kv = {l: (torch.zeros(1, cfg.nkv, 1024, cfg.hd), torch.zeros(1, cfg.nkv, 1024, cfg.hd)) for l in range(cfg.L)}
    def step(self, x, pos, positions=None, parents=None, compact=None):
        if compact:
            f, idx = compact["from"], compact["idx"]; src = [f + i for i in idx]
            for K_, V_ in self.kv.values(): K_[:, :, f:f+len(idx)] = K_[:, :, src]; V_[:, :, f:f+len(idx)] = V_[:, :, src]
        pt = torch.tensor(positions) if positions else None; tree = ancestor_mask(parents) if parents else None
        cos, sin = rope_at(cfg, pt, x.dtype) if pt is not None else rope(cfg, pos, x.shape[1], x.dtype)
        for l in range(cfg.L): x = layer_forward(cfg, LAYERS[l], x, cos, sin, self.kv[l], pos, l, tree, pt)
        return x
def logits_of(h): return F.linear(rms(cfg, h, NORM), LM)

def plain_greedy(n):
    P = Pipeline(); out = []
    with torch.inference_mode():
        x = EMB[torch.tensor([ids])]; pos = 0
        for _ in range(n):
            nid = int(logits_of(P.step(x, pos)[:, -1:]).argmax()); out.append(nid); pos += x.shape[1]; x = EMB[torch.tensor([[nid]])]
    return out

NOISE = 3.0
class NoisyDraft(DraftModel):
    def _forward(self, toks, positions, mask=None):
        return super()._forward(toks, positions, mask) + NOISE * torch.randn(len(toks), self.cfg.vocab)

def tree_greedy(n, widths):
    P = Pipeline(); dm = NoisyDraft(W, 1024, torch.float32); out, acc = [], []
    with torch.inference_mode():
        h = P.step(EMB[torch.tensor([ids])], 0); cur = int(logits_of(h[:, -1:]).argmax()); out.append(cur); pos = len(ids)
        dm.prefill(ids); compact = None
        while len(out) < n:
            nodes = dm.build_tree(cur, widths)
            toks = [nd["tok"] for nd in nodes]; parents = [nd["parent"] for nd in nodes]; positions = [pos + nd["depth"] for nd in nodes]
            h = P.step(EMB[torch.tensor([toks])], pos, positions, parents, compact)
            children = {}
            for i, p in enumerate(parents): children.setdefault(p, []).append(i)
            node, accepted = 0, []
            while True:
                tgt = int(logits_of(h[:, node:node+1]).argmax()); nxt = next((c for c in children.get(node, []) if toks[c] == tgt), None)
                if nxt is None: bonus = tgt; break
                accepted.append(nxt); node = nxt
            out.extend([toks[c] for c in accepted] + [bonus]); acc.append(len(accepted))
            compact = {"from": pos, "idx": [0] + accepted}; dm.commit(accepted); pos += 1 + len(accepted); cur = bonus
    return out[:n], acc

N = 40; t = time.time(); ref = plain_greedy(N); print(f"plain greedy {time.time()-t:.0f} s")
for widths in ([1, 1, 1, 1, 1], [4, 6, 6, 4, 2], [8, 16, 16, 12, 8, 4]):
    t = time.time(); out, acc = tree_greedy(N, widths)
    print(f"lean-draft tree {widths} ({sum(widths)+1} nodes): identical={out == ref} accepted/cycle={sum(acc)/len(acc):.2f} cycles={len(acc)} {time.time()-t:.0f} s")
    if out != ref: print(" ref ", ref); print(" tree", out)
NOISE = 0.0
t = time.time(); out, acc = tree_greedy(N, [4, 6, 6, 4, 2]); print(f"noise-free draft (same model): identical={out == ref} accepted/cycle={sum(acc)/len(acc):.2f} {time.time()-t:.0f} s")
