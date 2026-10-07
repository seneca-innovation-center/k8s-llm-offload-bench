"""fp32 per-layer check of our layer code against the transformers decoder layer for any supported architecture.

    python layer_check.py <weights dir> [layers e.g. 0,1,27]
"""
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

from common import Cfg, layer_forward, load_cold, rope

torch.set_num_threads(16)
W = sys.argv[1]
cfg = Cfg(f"{W}/config.json")
hc = AutoConfig.from_pretrained(W)
hc._attn_implementation = "eager"
LAYERS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, cfg.L - 1]
S = 96
if cfg.arch == "qwen3":
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as Layer, Qwen3RotaryEmbedding as Rot
elif cfg.arch == "qwen3_moe":
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer as Layer, Qwen3MoeRotaryEmbedding as Rot
elif cfg.arch == "qwen2":
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as Layer, Qwen2RotaryEmbedding as Rot
else:
    raise SystemExit("use gptoss_check.py for gpt-oss")
tok = AutoTokenizer.from_pretrained(W)
ids = tok(open("prompt_long.txt").read())["input_ids"][:S + 1]
head, _ = load_cold(f"{W}/embed_head.safetensors", evict=False)
x0 = head["model.embed_tokens.weight"][torch.tensor([ids])].float()
rot = Rot(hc)
pos_ids = torch.arange(S + 1)[None]
cos_h, sin_h = rot(x0, pos_ids)
cos_m, sin_m = rope(cfg, 0, S + 1, torch.float32)
print(f"{cfg.arch}: inv_freq max|Δ| {(rot.inv_freq - cfg.inv_freq).abs().max().item():.2e}; cos max|Δ| {(cos_h[0] - cos_m).abs().max().item():.2e}")
mask = torch.zeros(1, 1, S, S).masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), torch.finfo(torch.float32).min)
for l in LAYERS:
    Wl, _ = load_cold(f"{W}/layer_{l:02d}.safetensors", evict=False)
    Wf = {k: v.float() for k, v in Wl.items()}
    layer = Layer(hc, l).float()
    missing = layer.load_state_dict(Wf, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print("  key mismatch:", missing.missing_keys[:3], missing.unexpected_keys[:3])
    layer.eval()
    with torch.inference_mode():
        ref = layer(x0[:, :S], attention_mask=mask, position_ids=pos_ids[:, :S], cache_position=torch.arange(S), position_embeddings=(cos_h[:, :S], sin_h[:, :S]))
        if isinstance(ref, tuple):
            ref = ref[0]
        kv = (torch.zeros(1, cfg.nkv, S + 8, cfg.hd), torch.zeros(1, cfg.nkv, S + 8, cfg.hd))
        mine = layer_forward(cfg, Wf, x0[:, :S], cos_m[:S], sin_m[:S], kv, 0, l)
        kv2 = (torch.zeros(1, cfg.nkv, S + 8, cfg.hd), torch.zeros(1, cfg.nkv, S + 8, cfg.hd))
        full = layer_forward(cfg, Wf, x0, cos_m, sin_m, kv2, 0, l)
        step = layer_forward(cfg, Wf, x0[:, S:S + 1], cos_m[S:S + 1], sin_m[S:S + 1], kv, S, l)
    print(f"layer {l:2d}: max|Δ| vs HF fp32 = {(mine - ref).abs().max().item():.2e} (scale {ref.abs().max().item():.1f}); "
          f"decode step vs prefill = {(step - full[:, S:S + 1]).abs().max().item():.2e}", flush=True)
