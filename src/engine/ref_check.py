"""Reference: HF transformers greedy generation on the same prompt (bf16, CPU).

    WEIGHTS=/exp/weights ATTN=sdpa python ref_check.py <hf model dir> <n_prompt> <max_new>
"""
import json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16)
W = os.environ.get("WEIGHTS", "/exp/weights")
PROMPT = os.environ.get("PROMPT_FILE", "/exp/prompt.txt")
src, n, new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(W)
ids = tok(open(PROMPT).read())["input_ids"]
ids = (ids * (n // len(ids) + 1))[:n]
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, attn_implementation=os.environ.get("ATTN", "sdpa"))
out = m.generate(torch.tensor([ids]), max_new_tokens=new, do_sample=False, min_new_tokens=new)
print(json.dumps({"tokens": out[0, n:].tolist(), "text": tok.decode(out[0, n:])}))
