"""Per-layer cold-load vs compute timings (same code paths the pods use).

    WEIGHTS=/exp/weights_gptoss python microbench.py [out.json]
"""
import json, os, statistics as st, sys, time

import torch

from common import Cfg, layer_forward, load_cold, rope

torch.set_num_threads(16)
W = os.environ.get("WEIGHTS", "/exp/weights")
cfg = Cfg(f"{W}/config.json")
lt, size, keep = [], 0, {}
for l in range(cfg.L):
    t = time.perf_counter()
    w, n = load_cold(f"{W}/layer_{l:02d}.safetensors")
    lt.append((time.perf_counter() - t) * 1e3)
    size = n
    if l >= cfg.L - 2:
        keep[l] = w      # keep the last two layers (gpt-oss: one sliding, one full)
    del w
warm = []
for _ in range(5):
    t = time.perf_counter()
    load_cold(f"{W}/layer_00.safetensors", evict=False)
    warm.append((time.perf_counter() - t) * 1e3)
res = {"model": cfg.arch, "layers": cfg.L, "layer_bytes": size, "model_bytes": size * cfg.L,
       "cold_load_ms_med": st.median(lt), "cold_load_ms_p90": sorted(lt)[int(0.9 * len(lt))],
       "cold_GBps": size / 1e9 / (st.median(lt) / 1e3), "warm_load_ms_med": st.median(warm)}
with torch.inference_mode():
    for s in (1, 128, 512, 2048):
        pos = 0 if s > 1 else 512
        x = torch.randn(1, s, cfg.H, dtype=torch.bfloat16)
        cos, sin = rope(cfg, pos, s, x.dtype)
        ts = []
        for l, w in keep.items():
            kv = (torch.zeros(1, cfg.nkv, 4096, cfg.hd, dtype=torch.bfloat16),
                  torch.zeros(1, cfg.nkv, 4096, cfg.hd, dtype=torch.bfloat16))
            for _ in range(3):
                layer_forward(cfg, w, x, cos, sin, kv, pos, l)
            for _ in range(10 if s < 2048 else 4):
                t = time.perf_counter()
                layer_forward(cfg, w, x, cos, sin, kv, pos, l)
                ts.append((time.perf_counter() - t) * 1e3)
        res[f"compute_ms_s{s}"] = st.median(ts)
res["K_needed_decode"] = 1 + res["cold_load_ms_med"] / res["compute_ms_s1"]
res["floor_s_per_token_at_7.1GBps"] = res["model_bytes"] / 7.1e9
out = json.dumps(res, indent=1)
print(out)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(out)
