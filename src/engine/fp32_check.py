"""Compare our layer implementation with HF Qwen2 in fp32 (logits of full prompt)."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import Cfg, layer_forward, load_cold, rms, rope
torch.set_num_threads(16)
src = sys.argv[1]; cfg = Cfg("/exp/weights/config.json")
tok = AutoTokenizer.from_pretrained("/exp/weights")
ids = tok(open("/exp/prompt.txt").read())["input_ids"][:128]
m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
with torch.inference_mode():
    ref = m(torch.tensor([ids])).logits[0]
    head, _ = load_cold("/exp/weights/embed_head.safetensors", evict=False)
    E = head["model.embed_tokens.weight"].float(); N = head["model.norm.weight"].float()
    x = E[torch.tensor([ids])]; cos, sin = rope(cfg, 0, len(ids), torch.float32)
    for l in range(cfg.L):
        W = {k: v.float() for k, v in load_cold(f"/exp/weights/layer_{l:02d}.safetensors", evict=False)[0].items()}
        kv = (torch.zeros(1, cfg.nkv, 256, cfg.hd), torch.zeros(1, cfg.nkv, 256, cfg.hd))
        x = layer_forward(cfg, W, x, cos, sin, kv, 0, l)
    ours = (rms(cfg, x, N) @ E.T)[0]
print("max|dlogit|", (ours - ref).abs().max().item(), "ref logit scale", ref.abs().max().item(),
      "argmax agree", (ours.argmax(-1) == ref.argmax(-1)).float().mean().item())
