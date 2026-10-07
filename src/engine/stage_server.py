"""Pipeline stage pod.

Pod i owns layer groups {g : g % K == i}. Groups are either PINNED (kept in memory) or
STREAMED: a streamed group is loaded from NVMe, computed, freed, and the next streamed
group is prefetched. SLOTS streamed groups may be in memory at once.

  MODE=cyclic, RESIDENT=""        -> everything streamed, one group at a time (the cyclic experiment)
  MODE=cyclic, RESIDENT=0-32, SLOTS=3 -> layers 0-32 pinned, layers 33-35 streamed (partial residency)
  MODE=resident                   -> everything pinned (baseline / controls)
  LOAD_MODE=mmap                  -> weights memory-mapped; the kernel pages them in/out (use with a cgroup limit)
"""
import faulthandler, os, queue, signal, socket, threading, time, traceback

import torch

faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 1 -> stack dump of every thread

from common import (Cfg, ancestor_mask, connect, from_bytes, layer_forward, load_cold, load_direct, load_mmap, mem_stats,
                    recv_msg, reset_hwm, rope, rope_at, send_msg, to_bytes)

W_DIR = os.environ.get("WEIGHTS", "/exp/weights")
IDX = int(os.environ["POD_NAME"].rsplit("-", 1)[1])
G, K = int(os.environ["G"]), int(os.environ["K"])
MODE = os.environ.get("MODE", "cyclic")
LOAD_MODE = os.environ.get("LOAD_MODE", "read")
SLOTS = int(os.environ.get("SLOTS", 1))
BATCH = int(os.environ.get("BATCH", 1))
PORT = int(os.environ.get("PORT", 9000))
MAXLEN = int(os.environ.get("MAXLEN", 4096))
STAGE_FMT = os.environ.get("STAGE_FMT", "stage-{}.stage.default.svc.cluster.local")
COORD = os.environ.get("COORD", "coordinator.default.svc.cluster.local")
torch.set_num_threads(int(os.environ.get("THREADS", 16)))
torch.set_num_interop_threads(1)

cfg = Cfg(f"{W_DIR}/config.json")
assert cfg.L % G == 0 and G % K == 0, "need G | L and K | G"
LPG = cfg.L // G
MY = [g for g in range(G) if g % K == IDX]


def parse_ranges(s):
    out = set()
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


RESIDENT = parse_ranges(os.environ.get("RESIDENT", ""))
PINNED = [g for g in MY if MODE == "resident" or all(l in RESIDENT for l in range(g * LPG, (g + 1) * LPG))]
STREAMED = [g for g in MY if g not in PINNED]
STREAMED_SET = set(STREAMED)
EFF_SLOTS = min(SLOTS, len(STREAMED))
NLOADERS = max(1, min(EFF_SLOTS, 4))

kv = {l: (torch.zeros(BATCH, cfg.nkv, MAXLEN, cfg.hd, dtype=torch.bfloat16),
          torch.zeros(BATCH, cfg.nkv, MAXLEN, cfg.hd, dtype=torch.bfloat16))
      for g in MY for l in range(g * LPG, (g + 1) * LPG)}

cv = threading.Condition()
slots = {}            # group -> list of per-layer weight dicts
load_ms = {}          # group -> last load duration
gbytes = {}           # group -> bytes read
w_bytes = {"cur": 0, "peak": 0, "loads": 0, "load_bytes": 0}
loadq = queue.Queue()


def load_group(g):
    t0 = time.perf_counter()
    ws, nb = [], 0
    for l in range(g * LPG, (g + 1) * LPG):
        path = f"{W_DIR}/layer_{l:02d}.safetensors"
        if LOAD_MODE == "mmap":
            w, n = load_mmap(path)
        elif LOAD_MODE == "direct":
            w, n = load_direct(path)
        else:
            w, n = load_cold(path, evict=True)
        ws.append(w)
        nb += n
    dt = (time.perf_counter() - t0) * 1e3
    with cv:
        slots[g] = ws
        load_ms[g] = dt
        gbytes[g] = nb
        w_bytes["cur"] += nb
        w_bytes["peak"] = max(w_bytes["peak"], w_bytes["cur"])
        w_bytes["loads"] += 1
        w_bytes["load_bytes"] += nb
        cv.notify_all()


def loader():
    while True:
        g = loadq.get()
        try:
            load_group(g)
        except Exception:
            traceback.print_exc()


def evict(g):
    with cv:
        ws = slots.pop(g)
        w_bytes["cur"] -= gbytes[g]
    del ws  # last references -> buffer freed (munmap)


out_socks, out_lock = {}, threading.Lock()


def send_to(host, hdr, payload):
    with out_lock:
        s = out_socks.get(host) or out_socks.setdefault(host, connect(host, PORT))
        send_msg(s, hdr, payload)


compute_lock = threading.Lock()


def forward(hdr, payload):
    """Run group hdr["g"] and every following group that lives on this pod, then hand off.

    Consecutive groups owned by the same pod (always the case for K=1) are chained in-process;
    sending them to ourselves over TCP would add a hop per group and deadlocks once the
    hidden state exceeds the socket buffer (2048-token prefill = 8 MB).
    """
    g, pos = hdr["g"], hdr["pos"]
    B, s = hdr["shape"]
    x = from_bytes(payload, (B, s, cfg.H))
    # KV compaction after a speculative pass: keep only the accepted nodes, in order (once per pod per pass)
    comp = hdr.get("compact")
    if comp and g == IDX:
        f, idx = comp["from"], comp["idx"]
        src = [f + i for i in idx]
        for K_, V_ in kv.values():
            K_[:, :, f:f + len(idx)] = K_[:, :, src]
            V_[:, :, f:f + len(idx)] = V_[:, :, src]
    positions = torch.tensor(hdr["positions"]) if hdr.get("positions") else None
    tree = ancestor_mask(hdr["parents"]) if hdr.get("parents") else None
    cos, sin = rope_at(cfg, positions, x.dtype) if positions is not None else rope(cfg, pos, s, x.dtype)
    while True:
        t_arr = time.time()
        with cv:
            while g not in slots:
                if not cv.wait(30):
                    print(f"WAIT g={g} resident={sorted(slots)} queue={loadq.qsize()}", flush=True)
            ws = slots[g]
        t_start = time.time()
        with compute_lock, torch.inference_mode():
            for i, W in enumerate(ws):
                l = g * LPG + i
                x = layer_forward(cfg, W, x, cos, sin, kv[l], pos, l, tree, positions)
            del W
        del ws
        t_end = time.time()
        if g in STREAMED_SET:
            evict(g)
            loadq.put(STREAMED[(STREAMED.index(g) + EFF_SLOTS) % len(STREAMED)])
        hdr["trace"].append({"g": g, "pod": IDX, "arr": t_arr, "start": t_start, "end": t_end,
                             "load_ms": load_ms.get(g)})
        if g == G - 1:
            send_to(COORD, hdr, to_bytes(x))
            return
        g += 1
        if g % K != IDX:
            hdr["g"] = g
            send_to(STAGE_FMT.format(g % K), hdr, to_bytes(x))
            return


def stats():
    with cv:
        st = dict(w_bytes)
        st["resident_groups"] = sorted(slots)
    st.update(mem_stats())
    st["kv_bytes"] = sum(a.numel() * a.element_size() * 2 for a, _ in kv.values())
    st.update(pod=IDX, groups=MY, pinned=PINNED, streamed=STREAMED, slots=EFF_SLOTS, load_mode=LOAD_MODE,
              cyclic=bool(STREAMED), load_ms=dict(load_ms))
    return st


def handle(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            hdr, payload = recv_msg(conn)
            op = hdr.get("op")
            if op == "fwd":
                forward(hdr, payload)
            elif op == "stats":
                send_msg(conn, stats())
            elif op == "reset":
                with cv:
                    w_bytes.update(peak=w_bytes["cur"], loads=0, load_bytes=0)
                reset_hwm()
                send_msg(conn, {"ok": True})
    except ConnectionError:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()


def main():
    for g in PINNED + STREAMED[:EFF_SLOTS]:  # preload before accepting traffic
        load_group(g)
    for _ in range(NLOADERS):
        threading.Thread(target=loader, daemon=True).start()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(64)
    print(f"stage {IDX}: G={G} K={K} mode={MODE} load={LOAD_MODE} batch={BATCH} pinned={PINNED} "
          f"streamed={STREAMED} slots={EFF_SLOTS} loaders={NLOADERS} resident_now={sorted(slots)}", flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
