"""Validate our Qwen3-MoE layer against transformers' Qwen3MoeDecoderLayer on a tiny random-weight config (fp32)."""
import json, os, tempfile, torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
from common import Cfg, layer_forward, rope
torch.manual_seed(0)
hc = Qwen3MoeConfig(hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], rms_norm_eps=1e-6,
                    rope_theta=10000.0, vocab_size=100, attention_bias=False)
hc._attn_implementation = "eager"
d = tempfile.mkdtemp(); json.dump(hc.to_dict(), open(f"{d}/config.json", "w"))
cfg = Cfg(f"{d}/config.json")
layer = Qwen3MoeDecoderLayer(hc, 0).float().eval()
for p in layer.parameters(): torch.nn.init.normal_(p, std=0.2)
W = {k: v.detach().clone() for k, v in layer.state_dict().items()}
S = 12; x = torch.randn(1, S + 1, 64)
rot = Qwen3MoeRotaryEmbedding(hc); pos = torch.arange(S + 1)[None]; cos_h, sin_h = rot(x, pos); cos_m, sin_m = rope(cfg, 0, S + 1, torch.float32)
mask = torch.zeros(1, 1, S, S).masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), torch.finfo(torch.float32).min)
with torch.inference_mode():
    ref = layer(x[:, :S], attention_mask=mask, position_ids=pos[:, :S], cache_position=torch.arange(S), position_embeddings=(cos_h[:, :S], sin_h[:, :S]))
    ref = ref[0] if isinstance(ref, tuple) else ref
    kv = (torch.zeros(1, 2, 32, 16), torch.zeros(1, 2, 32, 16)); mine = layer_forward(cfg, W, x[:, :S], cos_m[:S], sin_m[:S], kv, 0, 0)
    kv2 = (torch.zeros(1, 2, 32, 16), torch.zeros(1, 2, 32, 16)); full = layer_forward(cfg, W, x, cos_m, sin_m, kv2, 0, 0)
    step = layer_forward(cfg, W, x[:, S:S + 1], cos_m[S:S + 1], sin_m[S:S + 1], kv, S, 0)
print(f"qwen3_moe tiny: max|Δ| vs HF = {(mine - ref).abs().max().item():.2e} (scale {ref.abs().max().item():.2f}); decode step vs prefill = {(step - full[:, S:S + 1]).abs().max().item():.2e}")
print("experts used per token:", cfg.top_k, "of", cfg.n_experts, "| norm_topk:", cfg.norm_topk)
