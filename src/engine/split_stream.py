"""Split a Hugging Face checkpoint into one safetensors file per decoder layer, one layer at a time (low RAM).

    python split_stream.py <model dir> <out dir>
Handles single-file and sharded checkpoints (model.safetensors.index.json). Writes embed_head.safetensors with
embeddings, final norm and lm_head (if untied), and copies config/tokenizer files.
"""
import json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, OUT = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{SRC}/config.json"))
L = cfg["num_hidden_layers"]
idx_path = f"{SRC}/model.safetensors.index.json"
if os.path.exists(idx_path):
    key2file = json.load(open(idx_path))["weight_map"]
else:
    key2file = {}
    with safe_open(f"{SRC}/model.safetensors", "pt") as f:
        for k in f.keys():
            key2file[k] = "model.safetensors"
handles = {}


def get(k):
    fn = key2file[k]
    if fn not in handles:
        handles[fn] = safe_open(f"{SRC}/{fn}", "pt")
    return handles[fn].get_tensor(k).to(torch.bfloat16).contiguous()


os.makedirs(OUT, exist_ok=True)
by_layer = {}
others = []
for k in key2file:
    if k.startswith("model.layers."):
        by_layer.setdefault(int(k.split(".")[2]), []).append(k)
    else:
        others.append(k)
for i in range(L):
    p = f"model.layers.{i}."
    save_file({k[len(p):]: get(k) for k in by_layer[i]}, f"{OUT}/layer_{i:02d}.safetensors")
    print("layer", i, flush=True)
head = {k: get(k) for k in others}
save_file(head, f"{OUT}/embed_head.safetensors")
for f in os.listdir(SRC):
    if f.endswith((".json", ".txt", ".jinja", ".model")) and "index" not in f and f != "tree.json":
        shutil.copy(f"{SRC}/{f}", OUT)
print("layers", L, "written to", OUT, "| head keys:", sorted(head))
for f in sorted(os.listdir(OUT))[:3]:
    print(f, os.path.getsize(f"{OUT}/{f}"))
