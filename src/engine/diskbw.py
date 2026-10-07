"""Aggregate cold-read bandwidth with N concurrent loaders (distinct layer files)."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from common import load_cold
out = {}
for n in (1, 2, 4, 8, 12, 18):
    files = [f"/exp/weights/layer_{l:02d}.safetensors" for l in range(36)]
    t = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        tot = sum(sz for _, sz in ex.map(load_cold, files))
    out[n] = round(tot / (time.perf_counter() - t) / 1e9, 2)
print(json.dumps({"aggregate_GBps_by_concurrency": out}))
