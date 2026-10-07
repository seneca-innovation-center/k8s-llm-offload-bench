"""Run inside the coordinator pod: benchmark one deployed config, append JSON lines to the results file."""
import argparse, json, time, urllib.request


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--prompts", default="128,512,2048")
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default="/exp/results/results.jsonl")
a = ap.parse_args()

meta = dict(json.loads(a.tag))
with open(a.out, "a") as f:
    for n in map(int, a.prompts.split(",")):
        call("/generate", {"n_prompt": n, "max_new": 4})  # warmup
        call("/reset", {})
        runs = [call("/generate", {"n_prompt": n, "max_new": a.max_new}) for _ in range(a.reps)]
        st = call("/stats")
        for i, r in enumerate(runs):
            rec = dict(meta, n_prompt=n, rep=i, ttft_ms=r["ttft_ms"], tpot_ms=r["tpot_ms"], tokens=r["tokens"],
                       batch=r.get("batch", 1), spec=r.get("spec"), traces=r["traces"] if i == 0 else None,
                       stats=st if i == 0 else None, t=time.time())
            f.write(json.dumps(rec) + "\n")
        tp = sorted(x for r in runs for x in r["tpot_ms"])
        line = dict(meta, n_prompt=n, ttft_med=sorted(r["ttft_ms"] for r in runs)[len(runs) // 2],
                    tpot_med=tp[len(tp) // 2], text=runs[0]["text"][:60])
        if runs[0].get("spec"):
            acc = [k for r in runs for k in r["spec"]["accepted"]]
            line["accepted_mean"] = sum(acc) / len(acc)
        print(json.dumps(line), flush=True)
