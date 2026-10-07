"""gpt-oss: check our layer implementation against transformers' GptOssDecoderLayer in fp32,
layer by layer (whole-model fp32 would need ~85 GB), and check our decode path against our prefill path.

    python gptoss_check.py /exp/weights_gptoss
"""
import sys

import torch
from transformers import AutoTokenizer, GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

from common import Cfg, layer_forward, load_cold, rope

torch.set_num_threads(16)
W = sys.argv[1]
S = 256                                  # longer than the 128-token sliding window
LAYERS = [0, 1, 22, 23]                  # sliding, full, sliding, full

cfg = Cfg(f"{W}/config.json")
hc = GptOssConfig.from_pretrained(W)
hc._attn_implementation = "eager"
tok = AutoTokenizer.from_pretrained(W)
ids = tok(open("/exp/prompt_long.txt").read())["input_ids"][:S + 1]
head, _ = load_cold(f"{W}/embed_head.safetensors", evict=False)
x0 = head["model.embed_tokens.weight"][torch.tensor([ids])].float()          # [1, S+1, H]

rot = GptOssRotaryEmbedding(hc)
pos_ids = torch.arange(S + 1)[None]
cos_h, sin_h = rot(x0, pos_ids)                                               # [1, S+1, hd/2]
cos_m, sin_m = rope(cfg, 0, S + 1, torch.float32)                            # [S+1, hd]
print(f"inv_freq max|Δ| {(rot.inv_freq - cfg.inv_freq).abs().max().item():.3e}; "
      f"attention scaling HF {rot.attention_scaling:.6f} ours {cfg.attn_scaling:.6f}; "
      f"cos max|Δ| {(cos_h[0] - cos_m[:, :cfg.hd // 2]).abs().max().item():.3e}")

qi, kj = torch.arange(S)[:, None], torch.arange(S)[None]
for l in LAYERS:
    Wl, _ = load_cold(f"{W}/layer_{l:02d}.safetensors", evict=False)
    Wf = {k: v.float() for k, v in Wl.items()}
    layer = GptOssDecoderLayer(hc, l).float()
    layer.load_state_dict(Wf, strict=True)
    layer.eval()
    allowed = kj <= qi
    if hc.layer_types[l] == "sliding_attention":
        allowed &= kj > qi - hc.sliding_window
    mask = torch.zeros(1, 1, S, S).masked_fill(~allowed, torch.finfo(torch.float32).min)
    with torch.inference_mode():
        ref = layer(x0[:, :S], attention_mask=mask, position_ids=pos_ids[:, :S], cache_position=torch.arange(S),
                    position_embeddings=(cos_h[:, :S], sin_h[:, :S]))
        if isinstance(ref, tuple):
            ref = ref[0]
        kv = (torch.zeros(1, cfg.nkv, S + 8, cfg.hd), torch.zeros(1, cfg.nkv, S + 8, cfg.hd))
        mine = layer_forward(cfg, Wf, x0[:, :S], cos_m[:S], sin_m[:S], kv, 0, l)
        d = (mine - ref).abs().max().item()
        # decode consistency: prefill S tokens then one step, vs one prefill of S+1 tokens
        kv2 = (torch.zeros(1, cfg.nkv, S + 8, cfg.hd), torch.zeros(1, cfg.nkv, S + 8, cfg.hd))
        full = layer_forward(cfg, Wf, x0, cos_m, sin_m, kv2, 0, l)
        step = layer_forward(cfg, Wf, x0[:, S:S + 1], cos_m[S:S + 1], sin_m[S:S + 1], kv, S, l)
        d2 = (step - full[:, S:S + 1]).abs().max().item()
    print(f"layer {l:2d} ({hc.layer_types[l]:17s}): max|Δ| vs HF fp32 = {d:.3e} (output scale {ref.abs().max().item():.1f}); "
          f"decode step vs prefill = {d2:.3e}", flush=True)
    del Wl, Wf, layer
