"""Quick check of the deployed config: python smoke.py N_PROMPT MAX_NEW. Compares tokens with the sweep baseline."""
import json, os, sys, urllib.request


def call(p, b=None):
    r = urllib.request.Request("http://127.0.0.1:8000" + p, data=json.dumps(b).encode() if b is not None else None)
    return json.loads(urllib.request.urlopen(r, timeout=7200).read())


n, new = int(sys.argv[1]), int(sys.argv[2])
call("/reset", {})
r = call("/generate", {"n_prompt": n, "max_new": new})
st = call("/stats")
base = None
for l in open(os.environ.get("RESULTS", "/exp/results/results.jsonl")):
    x = json.loads(l)
    if x["G"] == 1 and x["n_prompt"] == n:
        base = x["tokens"]
        break
print("tokens", r["tokens"])
print("match baseline:", base is not None and r["tokens"][:len(base)] == base[:len(r["tokens"])])
tp = sorted(r["tpot_ms"])
print("ttft", round(r["ttft_ms"]), "tpot_med", round(tp[len(tp) // 2], 1) if tp else None, "batch", r.get("batch"))
if r.get("spec"):
    s = r["spec"]
    print("spec: cycles", len(s["accepted"]), "accepted/cycle", round(sum(s["accepted"]) / len(s["accepted"]), 2),
          "draft ms/cycle", round(sum(s["draft_ms"]) / len(s["draft_ms"])), "verify ms/cycle",
          round(sum(s["verify_ms"]) / len(s["verify_ms"])), "decode ms/token", round(s["decode_ms_total"] / (s["tokens_out"] - 1)))
tr = r["traces"][-1]
print("last-token per-group (wait, compute) ms", [(round((t["start"] - t["arr"]) * 1e3, 1), round((t["end"] - t["start"]) * 1e3, 1)) for t in tr][:8])
for s in st["stages"]:
    print("pod", s["pod"], "pinned", s["pinned"], "streamed", s["streamed"], "slots", s["slots"], "resident_now",
          s["resident_groups"], "w_peak MB", s["peak"] // 10**6, "loads", s["loads"], "HWM MB", s["VmHWM"] // 10**6,
          "cg_peak MB", (s.get("cg_peak") or 0) // 10**6, "cg_max", s.get("cg_max"))
c = st["coordinator"]
print("coord HWM MB", c["VmHWM"] // 10**6, "draft MB", c["draft_bytes"] // 10**6)
