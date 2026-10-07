"""Does loading slow the computing? Decode compute per layer while 4 threads stream layers from NVMe with
buffered reads vs O_DIRECT, plus aggregate read bandwidth for 1/2/4 readers in both modes.

    WEIGHTS=/exp/weights python contention.py [out.json]
"""
import json, os, statistics as st, sys, threading, time

import torch

from common import Cfg, layer_forward, load_cold, load_direct, rope

torch.set_num_threads(16)
W = os.environ.get("WEIGHTS", "/exp/weights")
cfg = Cfg(f"{W}/config.json")
files = [f"{W}/layer_{l:02d}.safetensors" for l in range(cfg.L)]
w0, _ = load_cold(files[cfg.L - 1], evict=False)
LOADERS = {"buffered": lambda p: load_cold(p, evict=True), "direct": load_direct}


def bandwidth(loader, nthreads, dur=6.0):
    stop, total = time.time() + dur, [0] * nthreads

    def run(i):
        k = i
        while time.time() < stop:
            _, n = loader(files[k % cfg.L])
            total[i] += n
            k += nthreads
    th = [threading.Thread(target=run, args=(i,)) for i in range(nthreads)]
    t0 = time.time()
    for x in th:
        x.start()
    for x in th:
        x.join()
    return sum(total) / (time.time() - t0) / 1e9


def compute_ms(dur=6.0, s=1):
    kv = (torch.zeros(1, cfg.nkv, 4096, cfg.hd, dtype=torch.bfloat16), torch.zeros(1, cfg.nkv, 4096, cfg.hd, dtype=torch.bfloat16))
    x = torch.randn(1, s, cfg.H, dtype=torch.bfloat16)
    pos = 512 if s == 1 else 0
    cos, sin = rope(cfg, pos, s, x.dtype)
    ts, stop = [], time.time() + dur
    with torch.inference_mode():
        while time.time() < stop:
            t = time.perf_counter()
            layer_forward(cfg, w0, x, cos, sin, kv, pos, cfg.L - 1)
            ts.append((time.perf_counter() - t) * 1e3)
    return st.median(ts)


res = {"model": cfg.arch, "compute_ms_idle_s1": compute_ms(), "compute_ms_idle_s512": compute_ms(s=512)}
for name, loader in LOADERS.items():
    for n in (1, 2, 4):
        res[f"bw_GBps_{name}_{n}readers"] = round(bandwidth(loader, n), 2)
    stop = [False]

    def bg(i):
        k = i
        while not stop[0]:
            loader(files[k % cfg.L])
            k += 4
    th = [threading.Thread(target=bg, args=(i,)) for i in range(4)]
    for x in th:
        x.start()
    time.sleep(1.0)
    res[f"compute_ms_s1_during_{name}_loads"] = compute_ms()
    res[f"compute_ms_s512_during_{name}_loads"] = compute_ms(s=512)
    stop[0] = True
    for x in th:
        x.join()
for name in LOADERS:
    res[f"slowdown_s1_{name}"] = round(res[f"compute_ms_s1_during_{name}_loads"] / res["compute_ms_idle_s1"], 2)
    res[f"slowdown_s512_{name}"] = round(res[f"compute_ms_s512_during_{name}_loads"] / res["compute_ms_idle_s512"], 2)
out = json.dumps(res, indent=1)
print(out)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(out)
