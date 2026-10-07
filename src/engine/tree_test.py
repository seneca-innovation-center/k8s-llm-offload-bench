"""Local correctness test for tree speculation, in fp32 so results are exact.

Target = our layer code over the local 0.5B layer files (all resident, in-process "pipeline" with the same
header semantics as the stages: positions, parents, compaction). Draft = the HF 0.5B model with noise added
to its logits so that guesses are often wrong (exercises rejection, bonus tokens and KV compaction).
Checks: (1) tree-speculative greedy output == plain greedy output; (2) chain (gamma) mode == plain greedy.

    python3 tree_test.py <layer files of Qwen2.5-0.5B-Instruct>
"""
import copy, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from common import Cfg, ancestor_mask, layer_forward, load_cold, rms, rope, rope_at

torch.manual_seed(0)
W = sys.argv[1]
cfg = Cfg(f"{W}/config.json")
tok = AutoTokenizer.from_pretrained(W)
head, _ = load_cold(f"{W}/embed_head.safetensors", evict=False)
EMB = head["model.embed_tokens.weight"].float()
LM = head.get("lm_head.weight", head["model.embed_tokens.weight"]).float()
NORM = head["model.norm.weight"].float()
LAYERS = [{k: v.float() for k, v in load_cold(f"{W}/layer_{l:02d}.safetensors", evict=False)[0].items()} for l in range(cfg.L)]
MAXLEN = 1024
ids = tok(open("prompt_long.txt").read())["input_ids"][:96]


class Pipeline:
    """All layers in one process, same semantics as the stages."""
    def __init__(self):
        self.kv = {l: (torch.zeros(1, cfg.nkv, MAXLEN, cfg.hd), torch.zeros(1, cfg.nkv, MAXLEN, cfg.hd)) for l in range(cfg.L)}

    def step(self, x, pos, positions=None, parents=None, compact=None):
        if compact:
            f, idx = compact["from"], compact["idx"]
            src = [f + i for i in idx]
            for K_, V_ in self.kv.values():
                K_[:, :, f:f + len(idx)] = K_[:, :, src]
                V_[:, :, f:f + len(idx)] = V_[:, :, src]
        pt = torch.tensor(positions) if positions else None
        tree = ancestor_mask(parents) if parents else None
        cos, sin = rope_at(cfg, pt, x.dtype) if pt is not None else rope(cfg, pos, x.shape[1], x.dtype)
        for l in range(cfg.L):
            x = layer_forward(cfg, LAYERS[l], x, cos, sin, self.kv[l], pos, l, tree, pt)
        return x


def logits_of(h):
    return torch.nn.functional.linear(rms(cfg, h, NORM), LM)


def plain_greedy(max_new):
    P = Pipeline()
    out = []
    with torch.inference_mode():
        x = EMB[torch.tensor([ids])]
        pos = 0
        for _ in range(max_new):
            h = P.step(x, pos)
            nid = int(logits_of(h[:, -1:]).argmax())
            out.append(nid)
            pos += x.shape[1]
            x = EMB[torch.tensor([[nid]])]
    return out


draft = AutoModelForCausalLM.from_pretrained(sys.argv[2], dtype=torch.float32).eval()
NOISE = 3.0  # logit noise: makes the draft wrong often enough to exercise rejections


def noisy(logits):
    return logits + NOISE * torch.randn_like(logits)


def build_tree(root_tok, root_logits, cache, widths):
    nodes = [{"tok": root_tok, "parent": -1, "depth": 0, "cum": 1.0}]
    level, logits = [0], root_logits
    for d, width in enumerate(widths):
        probs = torch.softmax(logits.float(), -1)
        topv, topi = probs.topk(min(width, probs.shape[-1]), -1)
        cands = sorted(((nodes[n]["cum"] * topv[r, c].item(), n, r, topi[r, c].item())
                        for r, n in enumerate(level) for c in range(topv.shape[1])), reverse=True)[:width]
        rows, toks, new_level = [], [], []
        for score, n, r, tk in cands:
            nodes.append({"tok": tk, "parent": n, "depth": d + 1, "cum": score})
            new_level.append(len(nodes) - 1)
            rows.append(r)
            toks.append(tk)
        if d == len(widths) - 1:
            break
        cache.batch_select_indices(torch.tensor(rows))
        o = draft(input_ids=torch.tensor(toks)[:, None], past_key_values=cache, use_cache=True)
        cache, logits, level = o.past_key_values, noisy(o.logits[:, -1]), new_level
    return nodes


def tree_greedy(max_new, widths):
    P = Pipeline()
    out, acc, nn_ = [], [], []
    with torch.inference_mode():
        h = P.step(EMB[torch.tensor([ids])], 0)
        cur = int(logits_of(h[:, -1:]).argmax())
        out.append(cur)
        pos = len(ids)
        committed, pending, compact = DynamicCache(), ids + [cur], None
        while len(out) < max_new:
            o = draft(input_ids=torch.tensor([pending]), past_key_values=committed, use_cache=True)
            committed = o.past_key_values
            nodes = build_tree(cur, noisy(o.logits[:, -1]), copy.deepcopy(committed), widths)
            toks = [nd["tok"] for nd in nodes]
            parents = [nd["parent"] for nd in nodes]
            positions = [pos + nd["depth"] for nd in nodes]
            h = P.step(EMB[torch.tensor([toks])], pos, positions, parents, compact)
            preds = logits_of(h).argmax(-1)[0].tolist()
            children = {}
            for i, p in enumerate(parents):
                children.setdefault(p, []).append(i)
            node, accepted = 0, []
            while True:
                tgt = preds[node]
                nxt = next((c for c in children.get(node, []) if toks[c] == tgt), None)
                if nxt is None:
                    bonus = tgt
                    break
                accepted.append(nxt)
                node = nxt
            new = [toks[c] for c in accepted] + [bonus]
            out.extend(new)
            acc.append(len(accepted))
            nn_.append(len(nodes))
            compact = {"from": pos, "idx": [0] + accepted}
            pos += 1 + len(accepted)
            cur = bonus
            pending = new
    return out[:max_new], acc, nn_


N = 40
t = time.time()
ref = plain_greedy(N)
print(f"plain greedy: {N} tokens in {time.time() - t:.0f} s")
for widths in ([1, 1, 1, 1, 1], [4, 6, 6, 4, 2], [8, 16, 16, 12, 8, 4]):
    t = time.time()
    out, acc, nn_ = tree_greedy(N, widths)
    print(f"tree {widths} ({sum(widths) + 1} nodes): identical={out == ref}  accepted/cycle={sum(acc) / len(acc):.2f}  "
          f"cycles={len(acc)}  {time.time() - t:.0f} s")
    if out != ref:
        print("  ref :", ref)
        print("  tree:", out)
print("ref text:", repr(tok.decode(ref))[:200])
