"""Download Qwen2.5-3B-Instruct and write one safetensors file per decoder layer."""
import json, os, shutil, sys
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/weights"
src = MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL, cache_dir=os.environ.get("HF_CACHE", "/hf"), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
cfg = json.load(open(f"{src}/config.json"))
L = cfg["num_hidden_layers"]
tensors = {}
for f in sorted(os.listdir(src)):
    if f.endswith(".safetensors"):
        with safe_open(f"{src}/{f}", "pt") as sf:
            for k in sf.keys():
                tensors[k] = sf.get_tensor(k).to(torch.bfloat16).contiguous()
os.makedirs(OUT, exist_ok=True)
for i in range(L):
    p = f"model.layers.{i}."
    save_file({k[len(p):]: v for k, v in tensors.items() if k.startswith(p)}, f"{OUT}/layer_{i:02d}.safetensors")
head = {k: v for k, v in tensors.items() if not k.startswith("model.layers.")}
if "lm_head.weight" not in head:
    head["lm_head.weight"] = head["model.embed_tokens.weight"]
    head = {k: v.clone() for k, v in head.items()}
    head.pop("lm_head.weight")  # tied: coordinator reuses embeddings
save_file(head, f"{OUT}/embed_head.safetensors")
for f in os.listdir(src):
    if f.endswith((".json", ".txt", ".model")) and "index" not in f:
        shutil.copy(f"{src}/{f}", OUT)
print("layers", L, "files written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(f"{OUT}/{f}"))
