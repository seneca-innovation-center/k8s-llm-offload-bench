"""gpt-oss-20b: dequantize the native MXFP4 expert weights to bf16 (transformers does this on CPU)
and write one safetensors file per decoder layer plus embed_head (embeddings, final norm, lm_head).

    python split_gptoss.py /exp/gptoss /exp/weights_gptoss
"""
import json, os, shutil, sys, time

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

SRC, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16)   # MXFP4 -> bf16 on CPU (no GPU kernels)
sd = m.state_dict()
print(f"loaded in {time.time() - t0:.0f} s, {len(sd)} tensors", flush=True)

cfg = json.load(open(f"{SRC}/config.json"))
L, E, H, I = cfg["num_hidden_layers"], cfg["num_local_experts"], cfg["hidden_size"], cfg["intermediate_size"]
assert tuple(sd["model.layers.0.mlp.experts.gate_up_proj"].shape) == (E, H, 2 * I), sd["model.layers.0.mlp.experts.gate_up_proj"].shape
assert tuple(sd["model.layers.0.mlp.experts.down_proj"].shape) == (E, I, H), sd["model.layers.0.mlp.experts.down_proj"].shape
assert "lm_head.weight" in sd, "expected an untied lm_head"

os.makedirs(OUT, exist_ok=True)
for i in range(L):
    p = f"model.layers.{i}."
    save_file({k[len(p):]: v.to(torch.bfloat16).contiguous() for k, v in sd.items() if k.startswith(p)},
              f"{OUT}/layer_{i:02d}.safetensors")
    print("layer", i, flush=True)
head = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items() if not k.startswith("model.layers.")}
save_file(head, f"{OUT}/embed_head.safetensors")

c = dict(cfg)
c.pop("quantization_config", None)   # the files we wrote are plain bf16
json.dump(c, open(f"{OUT}/config.json", "w"), indent=1)
for f in os.listdir(SRC):
    if f.endswith((".json", ".txt", ".jinja")) and f != "config.json" and "index" not in f:
        shutil.copy(f"{SRC}/{f}", OUT)

print("layers", L, "written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(f"{OUT}/{f}"))
