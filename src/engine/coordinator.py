"""Coordinator pod: tokenizer, embeddings, final norm + lm_head, greedy token loop.

Sends hidden states for group 0 to stage-0 and waits for group G-1's output to come back.
  BATCH=B      run B independent prompts in lockstep (per-user latency = step time)
  DRAFT=<dir>  speculative decoding: a small HF model drafts GAMMA tokens, the pipeline verifies them in one pass
HTTP API on :8000 -> POST /generate {"n_prompt": int, "max_new": int}, GET /stats, POST /reset.
"""
import copy, faulthandler, json, os, queue, signal, socket, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 1 -> stack dump of every thread
STEP_TIMEOUT = float(os.environ.get("STEP_TIMEOUT", 900))
import torch.nn.functional as F
from transformers import AutoTokenizer

from common import (Cfg, connect, from_bytes, load_cold, mem_stats, recv_msg, reset_hwm, rms,
                    send_msg, to_bytes)

W_DIR = os.environ.get("WEIGHTS", "/exp/weights")
K = int(os.environ["K"])
PORT = int(os.environ.get("PORT", 9000))
STAGE_FMT = os.environ.get("STAGE_FMT", "stage-{}.stage.default.svc.cluster.local")
BATCH = int(os.environ.get("BATCH", 1))
DRAFT = os.environ.get("DRAFT", "")
GAMMA = int(os.environ.get("GAMMA", 5))
MAXLEN = int(os.environ.get("MAXLEN", 4096))
TREE = [int(w) for w in os.environ.get("TREE", "").split(",") if w]   # tree widths per depth, e.g. "6,8,8,6,4"
torch.set_num_threads(int(os.environ.get("THREADS", 16)))

cfg = Cfg(f"{W_DIR}/config.json")
tok = AutoTokenizer.from_pretrained(W_DIR)
head, _ = load_cold(f"{W_DIR}/embed_head.safetensors", evict=False)
EMB = head["model.embed_tokens.weight"]
LM = head.get("lm_head.weight", EMB)
NORM = head["model.norm.weight"]

TEXT = open(os.path.join(os.path.dirname(__file__), os.environ.get("PROMPT_FILE", "prompt.txt"))).read()
if os.environ.get("CHAT") == "1":   # instruction prompts: wrap in the model's chat template, use the full length
    BASE_IDS = tok.apply_chat_template([{"role": "user", "content": TEXT.strip()}], add_generation_prompt=True)
else:
    BASE_IDS = tok(TEXT)["input_ids"]

draft, DRAFT_BYTES, dm = None, 0, None
DRAFT_LAYERS = os.environ.get("DRAFT_LAYERS", "")      # lean draft: our layer code over per-layer files
if DRAFT_LAYERS:
    assert BATCH == 1, "speculative decoding is implemented for BATCH=1"
    from draft_model import DraftModel
    dm = DraftModel(DRAFT_LAYERS, MAXLEN)
    DRAFT_BYTES = dm.bytes
elif DRAFT:
    assert BATCH == 1, "speculative decoding is implemented for BATCH=1"
    from transformers import AutoModelForCausalLM, DynamicCache
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, torch_dtype=torch.bfloat16).eval()
    DRAFT_BYTES = sum(p.numel() * p.element_size() for p in draft.parameters())

results = queue.Queue()
stage0 = None
gen_lock = threading.Lock()


def return_server():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(16)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=lambda c=c: _drain(c), daemon=True).start()


def _drain(c):
    try:
        while True:
            results.put(recv_msg(c))
    except ConnectionError:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        c.close()


def prompt_ids(n, b):
    """Prompt b is the base text rotated by b tokens, so batch rows differ; row 0 equals the single-user prompt.
    n <= 0 means: the whole prompt as is (chat prompts have their natural length)."""
    if n <= 0:
        return list(BASE_IDS)
    r = b % len(BASE_IDS)
    base = BASE_IDS[r:] + BASE_IDS[:r]
    return (base * (n // len(base) + 1))[:n]


def step(x, pos, positions=None, parents=None, compact=None):
    """x: [B, s, H] -> hidden states after the last layer, same shape.
    positions/parents describe a speculation tree over the s new tokens; compact = {"from", "idx"} tells the
    stages which of the previous pass's scratch KV entries to keep (accepted nodes, in order)."""
    global stage0
    if stage0 is None:
        stage0 = connect(STAGE_FMT.format(0), PORT)
    hdr = {"op": "fwd", "g": 0, "pos": pos, "shape": list(x.shape[:2]), "trace": []}
    if positions is not None:
        hdr["positions"], hdr["parents"] = positions, parents
    if compact is not None:
        hdr["compact"] = compact
    send_msg(stage0, hdr, to_bytes(x))
    try:
        hdr, payload = results.get(timeout=STEP_TIMEOUT)
    except queue.Empty:
        raise RuntimeError(f"no reply from the pipeline within {STEP_TIMEOUT:.0f} s (pos={pos}, shape={list(x.shape[:2])})")
    return hdr, from_bytes(payload, x.shape)


def logits_of(h):
    return F.linear(rms(cfg, h, NORM), LM)


def generate(n_prompt, max_new):
    if (draft is not None or dm is not None) and TREE:
        return generate_tree(n_prompt, max_new)
    if draft is not None:
        return generate_spec(n_prompt, max_new)
    ids = torch.tensor([prompt_ids(n_prompt, b) for b in range(BATCH)])
    out, tpot, traces = [[] for _ in range(BATCH)], [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = EMB[ids]
        pos = 0
        for i in range(max_new):
            ts = time.perf_counter()
            hdr, h = step(x, pos)
            nid = logits_of(h[:, -1:]).argmax(-1)[:, 0]
            te = time.perf_counter()
            if i == 0:
                ttft = (te - t0) * 1e3
            else:
                tpot.append((te - ts) * 1e3)
            traces.append(hdr["trace"])
            for b in range(BATCH):
                out[b].append(int(nid[b]))
            pos += x.shape[1]
            x = EMB[nid][:, None, :]
    return {"n_prompt": n_prompt, "max_new": max_new, "batch": BATCH, "tokens": out[0], "ttft_ms": ttft,
            "tpot_ms": tpot, "traces": traces, "text": tok.decode(out[0])}


def generate_spec(n_prompt, max_new):
    """Greedy speculative decoding. Output equals plain greedy decoding of the big model (up to bf16 rounding)."""
    ids = prompt_ids(n_prompt, 0)
    n = len(ids)
    out, traces = [], []
    spec = {"gamma": GAMMA, "accepted": [], "draft_ms": [], "verify_ms": [], "cycle_ms": []}
    t0 = time.perf_counter()
    with torch.inference_mode():
        hdr, h = step(EMB[torch.tensor([ids])], 0)
        cur = int(logits_of(h[:, -1:]).argmax())          # first token, confirmed by the big model
        ttft = (time.perf_counter() - t0) * 1e3
        out.append(cur)
        traces.append(hdr["trace"])
        pos = n                                            # big-model KV cache holds positions < pos
        confirmed = ids + [cur]
        cache = DynamicCache()
        pending = confirmed                                # tokens the draft model has not seen yet
        tpot = []
        t_dec = time.perf_counter()
        while len(out) < max_new:
            tc = time.perf_counter()
            drafts, inp = [], torch.tensor([pending])
            for _ in range(GAMMA):
                o = draft(input_ids=inp, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                d = int(o.logits[0, -1].argmax())
                drafts.append(d)
                inp = torch.tensor([[d]])
            td = time.perf_counter()
            hdr, h = step(EMB[torch.tensor([[cur] + drafts])], pos)   # verify cur + GAMMA drafts in one pass
            preds = logits_of(h).argmax(-1)[0].tolist()                # big model's choice after each input token
            k = 0
            while k < GAMMA and preds[k] == drafts[k]:
                k += 1
            new = drafts[:k] + [preds[k]]                              # accepted drafts + one corrected/bonus token
            tv = time.perf_counter()
            out.extend(new)
            traces.append(hdr["trace"])
            spec["accepted"].append(k)
            spec["draft_ms"].append((td - tc) * 1e3)
            spec["verify_ms"].append((tv - td) * 1e3)
            spec["cycle_ms"].append((tv - tc) * 1e3)
            tpot.extend([(tv - tc) * 1e3 / len(new)] * len(new))
            pos += 1 + k                                               # cur and k drafts are now in the big KV
            cur = preds[k]
            confirmed = confirmed + new
            keep = min(cache.get_seq_length(), pos)                    # draft cache prefix that is still valid
            cache.crop(keep)
            pending = confirmed[keep:]
        spec["decode_ms_total"] = (time.perf_counter() - t_dec) * 1e3
        spec["tokens_out"] = len(out)
    out = out[:max_new]
    return {"n_prompt": n_prompt, "max_new": max_new, "batch": 1, "tokens": out, "ttft_ms": ttft,
            "tpot_ms": tpot[:max_new - 1], "traces": traces, "spec": spec, "text": tok.decode(out)}


def build_tree(root_tok, root_logits, cache, widths):
    """Best-first speculation tree from the draft model (SpecExec-style): at each depth keep the `width`
    children with the highest cumulative draft probability over all nodes of the previous depth.
    Returns nodes [{tok, parent, depth}] with node 0 = root (the current token); cache is consumed."""
    nodes = [{"tok": root_tok, "parent": -1, "depth": 0, "cum": 1.0}]
    level, logits = [0], root_logits
    for d, width in enumerate(widths):
        probs = torch.softmax(logits.float(), -1)
        topv, topi = probs.topk(min(width, probs.shape[-1]), -1)
        cands = sorted(((nodes[n]["cum"] * topv[r, c].item(), n, r, topi[r, c].item())
                        for r, n in enumerate(level) for c in range(topv.shape[1])), reverse=True)[:width]
        rows, toks, new_level = [], [], []
        for score, n, r, tok in cands:
            nodes.append({"tok": tok, "parent": n, "depth": d + 1, "cum": score})
            new_level.append(len(nodes) - 1)
            rows.append(r)
            toks.append(tok)
        if d == len(widths) - 1:
            break
        cache.batch_select_indices(torch.tensor(rows))      # one cache row per child, copied from its parent
        out = draft(input_ids=torch.tensor(toks)[:, None], past_key_values=cache, use_cache=True)
        cache, logits, level = out.past_key_values, out.logits[:, -1], new_level
    return nodes


def generate_tree(n_prompt, max_new):
    """Greedy tree speculation: the draft proposes a tree, one streamed pass verifies all nodes, the longest
    root path the big model agrees with is kept plus one token of its own. Output = plain greedy output."""
    ids = prompt_ids(n_prompt, 0)
    n = len(ids)
    out, traces = [], []
    spec = {"gamma": sum(TREE), "tree": TREE, "accepted": [], "draft_ms": [], "verify_ms": [], "cycle_ms": [], "nodes": []}
    t0 = time.perf_counter()
    with torch.inference_mode():
        hdr, h = step(EMB[torch.tensor([ids])], 0)
        cur = int(logits_of(h[:, -1:]).argmax())
        ttft = (time.perf_counter() - t0) * 1e3
        out.append(cur)
        traces.append(hdr["trace"])
        pos = n
        confirmed = ids + [cur]
        if dm is not None:
            dm.pos = 0
            dm.prefill(ids)                                    # draft sees the prompt; root token handled in build_tree
        else:
            committed = DynamicCache()
        pending = confirmed                                    # tokens the HF draft cache does not hold yet
        compact = None
        tpot = []
        t_dec = time.perf_counter()
        while len(out) < max_new:
            tc = time.perf_counter()
            if dm is not None:
                nodes = dm.build_tree(cur, TREE)
            else:
                o = draft(input_ids=torch.tensor([pending]), past_key_values=committed, use_cache=True)
                committed = o.past_key_values
                nodes = build_tree(cur, o.logits[:, -1], copy.deepcopy(committed), TREE)
            toks = [nd["tok"] for nd in nodes]
            parents = [nd["parent"] for nd in nodes]
            positions = [pos + nd["depth"] for nd in nodes]
            td = time.perf_counter()
            hdr, h = step(EMB[torch.tensor([toks])], pos, positions, parents, compact)
            children = {}
            for i, p in enumerate(parents):
                children.setdefault(p, []).append(i)
            node, accepted = 0, []
            while True:
                tgt = int(logits_of(h[:, node:node + 1]).argmax())   # output layer only along the accepted path
                nxt = next((c for c in children.get(node, []) if toks[c] == tgt), None)
                if nxt is None:
                    bonus = tgt
                    break
                accepted.append(nxt)
                node = nxt
            new = [toks[c] for c in accepted] + [bonus]
            tv = time.perf_counter()
            out.extend(new)
            traces.append(hdr["trace"])
            spec["accepted"].append(len(accepted))
            spec["nodes"].append(len(nodes))
            spec["draft_ms"].append((td - tc) * 1e3)
            spec["verify_ms"].append((tv - td) * 1e3)
            spec["cycle_ms"].append((tv - tc) * 1e3)
            tpot.extend([(tv - tc) * 1e3 / len(new)] * len(new))
            compact = {"from": pos, "idx": [0] + accepted}      # root + accepted nodes become the next KV prefix
            if dm is not None:
                dm.commit(accepted)
            pos += 1 + len(accepted)
            cur = bonus
            confirmed = confirmed + new
            pending = new                                      # committed cache holds everything up to the old cur
        spec["decode_ms_total"] = (time.perf_counter() - t_dec) * 1e3
        spec["tokens_out"] = len(out)
    out = out[:max_new]
    return {"n_prompt": n_prompt, "max_new": max_new, "batch": 1, "tokens": out, "ttft_ms": ttft,
            "tpot_ms": tpot[:max_new - 1], "traces": traces, "spec": spec, "text": tok.decode(out)}


def stage_call(i, op):
    s = connect(STAGE_FMT.format(i), PORT, timeout=30)
    send_msg(s, {"op": op})
    hdr, _ = recv_msg(s)
    s.close()
    return hdr


class H(BaseHTTPRequestHandler):
    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
            co = mem_stats()
            co["draft_bytes"] = DRAFT_BYTES
            co["head_bytes"] = sum(t.numel() * t.element_size() for t in head.values())
            self._json({"coordinator": co, "stages": [stage_call(i, "stats") for i in range(K)]})
        else:
            self._json({"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/generate":
            with gen_lock:
                self._json(generate(int(body.get("n_prompt", 128)), int(body.get("max_new", 32))))
        elif self.path == "/reset":
            reset_hwm()
            self._json([stage_call(i, "reset") for i in range(K)])

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=return_server, daemon=True).start()
    print(f"coordinator ready K={K} batch={BATCH} draft={DRAFT_LAYERS or DRAFT or None} gamma={GAMMA} tree={TREE or None} "
          f"base prompt tokens={len(BASE_IDS)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
