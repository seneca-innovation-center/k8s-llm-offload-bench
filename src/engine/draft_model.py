"""Lean draft model for tree speculation: a small Qwen2 run through our own layer code, all layers resident.

The draft keeps one KV cache. Tree nodes are written into a scratch region after the committed prefix, level by
level, each level attending to the prefix plus its ancestors (tree mask). After the big model has verified the
tree, the same compaction as in the stages keeps only the accepted path, so the committed prefix is always exact.
No cache copies, no per-node loops: one forward per level.
"""
import torch
import torch.nn.functional as F

from common import Cfg, layer_forward, load_cold, rms, rope_at


class DraftModel:
    def __init__(self, wdir, maxlen=4096, dtype=torch.bfloat16):
        self.cfg = Cfg(f"{wdir}/config.json")
        self.dtype = dtype
        head, nb = load_cold(f"{wdir}/embed_head.safetensors", evict=False)
        self.emb = head["model.embed_tokens.weight"].to(dtype)
        self.lm = head.get("lm_head.weight", head["model.embed_tokens.weight"]).to(dtype)
        self.norm = head["model.norm.weight"].to(dtype)
        self.layers = []
        self.bytes = nb
        for l in range(self.cfg.L):
            w, n = load_cold(f"{wdir}/layer_{l:02d}.safetensors", evict=False)
            self.layers.append({k: v.to(dtype) for k, v in w.items()})
            self.bytes += n
        c = self.cfg
        self.kv = {l: (torch.zeros(1, c.nkv, maxlen, c.hd, dtype=dtype), torch.zeros(1, c.nkv, maxlen, c.hd, dtype=dtype))
                   for l in range(c.L)}
        self.pos = 0  # committed prefix length

    def _forward(self, toks, positions, mask=None):
        """toks: list[int]; positions: list[int]; mask: bool [s, start+s] or None (causal continuation).
        Writes K/V at [start, start+s) where start = self.pos + (already written scratch)."""
        x = self.emb[torch.tensor([toks])]
        pt = torch.tensor(positions)
        cos, sin = rope_at(self.cfg, pt, x.dtype)
        start = self._scratch_start
        for l in range(self.cfg.L):
            x = layer_forward(self.cfg, self.layers[l], x, cos, sin, self.kv[l], start, l, mask, pt)
        self._scratch_start += len(toks)
        return F.linear(rms(self.cfg, x[0], self.norm), self.lm)  # [s, V]

    def prefill(self, ids):
        """Run the prompt (and anything committed so far) through the draft. Returns logits of the last token."""
        self._scratch_start = self.pos
        s = len(ids)
        logits = self._forward(ids, list(range(self.pos, self.pos + s)),
                               torch.ones(s, self.pos + s, dtype=torch.bool).tril(diagonal=self.pos))
        self.pos += s
        return logits[-1]

    def build_tree(self, root_tok, widths):
        """Best-first tree (SpecExec-style): at depth d keep the `widths[d]` children with the highest cumulative
        draft probability over all nodes of depth d-1. Node 0 = root (root_tok, at position self.pos).
        Returns nodes [{tok, parent, depth}]; the nodes' K/V sit at [self.pos, self.pos + len(nodes))."""
        self._scratch_start = self.pos
        nodes = [{"tok": root_tok, "parent": -1, "depth": 0, "cum": 1.0}]
        root_mask = torch.ones(1, self.pos + 1, dtype=torch.bool)
        logits = self._forward([root_tok], [self.pos], root_mask)
        level = [0]
        for d, width in enumerate(widths):
            probs = torch.softmax(logits.float(), -1)
            topv, topi = probs.topk(min(width, probs.shape[-1]), -1)
            cands = sorted(((nodes[n]["cum"] * topv[r, c].item(), n, topi[r, c].item())
                            for r, n in enumerate(level) for c in range(topv.shape[1])), reverse=True)[:width]
            new_level = []
            for score, n, tok in cands:
                nodes.append({"tok": tok, "parent": n, "depth": d + 1, "cum": score})
                new_level.append(len(nodes) - 1)
            if d == len(widths) - 1:
                break
            # mask for the new level: prefix (all), earlier nodes (ancestors only), new nodes (self only)
            n_prev, s = len(nodes) - len(new_level), len(new_level)
            m = torch.zeros(s, self.pos + n_prev + s, dtype=torch.bool)
            m[:, :self.pos] = True
            for j, ni in enumerate(new_level):
                a = nodes[ni]["parent"]
                while a >= 0:
                    m[j, self.pos + a] = True
                    a = nodes[a]["parent"]
                m[j, self.pos + n_prev + j] = True
            logits = self._forward([nodes[i]["tok"] for i in new_level], [self.pos + d + 1] * s, m)
            level = new_level
        return nodes

    def commit(self, accepted_nodes):
        """Keep root + accepted nodes (indices into the tree) as the new committed prefix."""
        idx = [0] + list(accepted_nodes)
        src = [self.pos + i for i in idx]
        for K_, V_ in self.kv.values():
            K_[:, :, self.pos:self.pos + len(idx)] = K_[:, :, src]
            V_[:, :, self.pos:self.pos + len(idx)] = V_[:, :, src]
        self.pos += len(idx)
