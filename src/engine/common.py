"""Shared pieces: safetensors loaders (cold read / mmap), decoder layers (Qwen2, gpt-oss), framed TCP messages.

The layer maths replicate the Hugging Face reference implementations operation for operation
(checked in fp32 by fp32_check.py for Qwen2 and gptoss_check.py for gpt-oss).
"""
import json, math, os, socket, struct, time, warnings

import numpy as np
import torch
import torch.nn.functional as F

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _views(buf, t):
    """Split a safetensors buffer (numpy uint8) into tensors viewing the same memory."""
    hlen = struct.unpack("<Q", buf[:8].tobytes())[0]
    hdr = json.loads(buf[8:8 + hlen].tobytes())
    base, out = 8 + hlen, {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = t[base + s:base + e].view(DT[v["dtype"]]).view(v["shape"])
    return out


def load_cold(path, evict=True):
    """Read a safetensors file from disk into a private anonymous buffer.

    posix_fadvise(DONTNEED) before and after the read keeps the file out of the page
    cache, so every reload is a real NVMe read and the only copy lives in this process.
    Returned tensors are zero-copy views of the buffer; dropping them frees the memory.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        if evict:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        size = os.fstat(fd).st_size
        buf = np.empty(size, dtype=np.uint8)
        mv, off = memoryview(buf), 0
        while off < size:
            off += os.readv(fd, [mv[off:]])
        if evict:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return _views(buf, torch.from_numpy(buf)), size


def load_direct(path, align=4096):
    """Read a safetensors file with O_DIRECT: the drive DMAs straight into our buffer with no page-cache copy.
    btrfs silently falls back to buffered I/O in some cases (e.g. a direct read that crosses end-of-file), so the
    aligned part is read with O_DIRECT, the unaligned tail with a normal read, and whatever still landed in the
    page cache is evicted afterwards (checked with fincore.py / iotest.py)."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    try:
        size = os.fstat(fd).st_size
        aligned = size // align * align
        raw = np.empty(size + align, dtype=np.uint8)
        off = (-raw.ctypes.data) % align
        buf = raw[off:off + size]
        mv, pos, chunk = memoryview(buf), 0, 64 << 20
        while pos < aligned:
            n = os.readv(fd, [mv[pos:pos + min(chunk, aligned - pos)]])
            if n <= 0:
                break
            pos += n
    finally:
        os.close(fd)
    if aligned < size:  # unaligned tail: plain read
        fd = os.open(path, os.O_RDONLY)
        try:
            os.preadv(fd, [mv[aligned:size]], aligned)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return _views(buf, torch.from_numpy(buf)), size


def load_mmap(path):
    """Map a safetensors file; the kernel pages weights in on first touch and may evict
    them under memory pressure (cgroup limit). Nothing is copied into process memory."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # read-only buffer warning from torch.from_numpy
        return _views(mm, torch.from_numpy(np.asarray(mm))), mm.size


def yarn_inv_freq(c):
    """YaRN inverse frequencies + attention scaling, as transformers' _compute_yarn_parameters."""
    rs, base = c["rope_scaling"], c["rope_theta"]
    dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    factor = rs["factor"]
    orig = rs.get("original_max_position_embeddings") or c["max_position_embeddings"]
    attn = rs.get("attention_factor")
    if attn is None:
        mscale, mad = rs.get("mscale"), rs.get("mscale_all_dim")
        g = lambda s, m=1: 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
        attn = float(g(factor, mscale) / g(factor, mad)) if (mscale and mad) else g(factor)
    bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1
    corr = lambda nrot: (dim * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(base))
    low, high = corr(bf), corr(bs)
    if rs.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float32) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    ext = 1 - ramp
    return inter * (1 - ext) + extra * ext, attn


class Cfg:
    def __init__(self, path):
        c = json.load(open(path))
        self.arch = c.get("model_type", "qwen2")
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.nkv = c["num_key_value_heads"]
        self.hd = c.get("head_dim") or self.H // self.nh
        self.eps = c["rms_norm_eps"]
        self.theta = c.get("rope_theta", 10000.0)
        self.L = c["num_hidden_layers"]
        self.vocab = c["vocab_size"]
        self.scaling = self.hd ** -0.5
        rs = c.get("rope_scaling") or {}
        if rs.get("rope_type", rs.get("type")) == "yarn":
            self.inv_freq, self.attn_scaling = yarn_inv_freq(c)
        else:
            self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
            self.attn_scaling = 1.0
        # mixture-of-experts / gpt-oss specifics (None for dense models)
        self.n_experts = c.get("num_local_experts") or c.get("num_experts")
        self.top_k = c.get("num_experts_per_tok")
        self.norm_topk = bool(c.get("norm_topk_prob", False))
        self.window = c.get("sliding_window")
        self.layer_types = c.get("layer_types") or ["full_attention"] * self.L
        self.swiglu_limit = c.get("swiglu_limit", 7.0)
        self.swiglu_alpha = 1.702


def rms(cfg, x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.eps)
    if cfg.arch == "gpt_oss":      # GptOssRMSNorm: weight applied in fp32, then cast
        return (w * xf).to(x.dtype)
    return w * xf.to(x.dtype)      # Qwen2RMSNorm: cast, then weight


def rope_at(cfg, positions, dtype):
    """cos/sin for explicit positions (LongTensor [s]); tree nodes at the same depth share a position."""
    f = positions.to(torch.float32)[:, None] * cfg.inv_freq[None]
    emb = torch.cat([f, f], -1)
    return (emb.cos() * cfg.attn_scaling).to(dtype), (emb.sin() * cfg.attn_scaling).to(dtype)


def rope(cfg, pos, s, dtype):
    return rope_at(cfg, torch.arange(pos, pos + s), dtype)


def ancestor_mask(parents):
    """parents[j] = index of node j's parent (-1 for the root). mask[j, i] = node j may attend to node i
    (i is j itself or one of its ancestors)."""
    n = len(parents)
    m = torch.zeros(n, n, dtype=torch.bool)
    for j in range(n):
        i = j
        while i >= 0:
            m[j, i] = True
            i = parents[i]
    return m


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def _qwen2_layer(cfg, W, x, cos, sin, kv, pos, tree=None):
    """One Qwen2 decoder layer. x: [B, s, H]; kv: (K, V) each [B, nkv, maxT, hd]; cos/sin: [s, hd].

    Works for prefill (pos=0, s>1), decode (s=1), chain verify (pos>0, s>1) and tree verify
    (tree: bool [s, s], node j attends to the cache plus the new tokens flagged in tree[j]).
    """
    B, s, _ = x.shape
    h = rms(cfg, x, W["input_layernorm.weight"])
    q = F.linear(h, W["self_attn.q_proj.weight"], W["self_attn.q_proj.bias"]).view(B, s, cfg.nh, cfg.hd).transpose(1, 2)
    k = F.linear(h, W["self_attn.k_proj.weight"], W["self_attn.k_proj.bias"]).view(B, s, cfg.nkv, cfg.hd).transpose(1, 2)
    v = F.linear(h, W["self_attn.v_proj.weight"], W["self_attn.v_proj.bias"]).view(B, s, cfg.nkv, cfg.hd).transpose(1, 2)
    q = q * cos + _rot(q) * sin
    k = k * cos + _rot(k) * sin
    K, V = kv
    K[:, :, pos:pos + s] = k
    V[:, :, pos:pos + s] = v
    if tree is not None:  # tree verify: every node sees the whole cache plus its own ancestors
        # tree is [s, s] (ancestors among the new tokens) or a full [s, pos+s] mask (draft tree levels)
        mask = tree if tree.shape[1] == pos + s else torch.cat([torch.ones(s, pos, dtype=torch.bool), tree], dim=1)
        causal = False
    elif s == 1:
        mask, causal = None, False
    elif pos == 0:
        mask, causal = None, True
    else:  # new tokens attend to the whole cache plus earlier new tokens
        mask, causal = torch.ones(s, pos + s, dtype=torch.bool).tril(diagonal=pos), False
    o = F.scaled_dot_product_attention(q, K[:, :, :pos + s], V[:, :, :pos + s], attn_mask=mask,
                                       is_causal=causal, enable_gqa=True)
    x = x + F.linear(o.transpose(1, 2).reshape(B, s, cfg.H), W["self_attn.o_proj.weight"])
    h = rms(cfg, x, W["post_attention_layernorm.weight"])
    g = F.linear(h, W["mlp.gate_proj.weight"])
    u = F.linear(h, W["mlp.up_proj.weight"])
    return x + F.linear(F.silu(g) * u, W["mlp.down_proj.weight"])


def _gptoss_layer(cfg, W, x, cos, sin, kv, pos, l, tree=None, positions=None):
    """One gpt-oss decoder layer: attention with learned sinks (sliding window on even layers),
    then a top-k mixture of experts. Mirrors transformers' eager GptOss implementation."""
    B, s, _ = x.shape
    h = rms(cfg, x, W["input_layernorm.weight"])
    q = F.linear(h, W["self_attn.q_proj.weight"], W["self_attn.q_proj.bias"]).view(B, s, cfg.nh, cfg.hd).transpose(1, 2)
    k = F.linear(h, W["self_attn.k_proj.weight"], W["self_attn.k_proj.bias"]).view(B, s, cfg.nkv, cfg.hd).transpose(1, 2)
    v = F.linear(h, W["self_attn.v_proj.weight"], W["self_attn.v_proj.bias"]).view(B, s, cfg.nkv, cfg.hd).transpose(1, 2)
    q = q * cos + _rot(q) * sin
    k = k * cos + _rot(k) * sin
    K, V = kv
    K[:, :, pos:pos + s] = k
    V[:, :, pos:pos + s] = v
    T = pos + s
    sliding = cfg.layer_types[l] == "sliding_attention"
    lo = max(0, T - cfg.window) if (sliding and s == 1) else 0   # decode on a sliding layer: only the last `window` keys
    rep = cfg.nh // cfg.nkv
    Ks = K[:, :, lo:T].repeat_interleave(rep, dim=1)
    Vs = V[:, :, lo:T].repeat_interleave(rep, dim=1)
    att = torch.matmul(q, Ks.transpose(2, 3)) * cfg.scaling          # [B, nh, s, T-lo]
    if tree is not None:
        qi = positions[:, None]
        kc = torch.arange(lo, pos)[None]                      # cache keys, by position
        allowed_c = kc <= qi
        allowed_n = tree.clone()                               # new tokens: ancestors only
        if sliding:
            allowed_c &= kc > qi - cfg.window
            allowed_n &= positions[None] > qi - cfg.window
        att = att.masked_fill(~torch.cat([allowed_c, allowed_n], dim=1), torch.finfo(att.dtype).min)
    elif s > 1:
        qi = torch.arange(pos, T)[:, None]
        kj = torch.arange(lo, T)[None]
        allowed = kj <= qi
        if sliding:
            allowed &= kj > qi - cfg.window
        att = att.masked_fill(~allowed, torch.finfo(att.dtype).min)
    sinks = W["self_attn.sinks"].to(att.dtype).reshape(1, -1, 1, 1).expand(B, -1, s, -1)
    logits = torch.cat([att, sinks], dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)[..., :-1]  # the sink absorbs mass and is dropped
    o = torch.matmul(probs, Vs).transpose(1, 2).reshape(B, s, cfg.nh * cfg.hd)
    x = x + F.linear(o, W["self_attn.o_proj.weight"], W["self_attn.o_proj.bias"])
    # mixture of experts: router picks top_k of n_experts per token, softmax over the picked logits
    h = rms(cfg, x, W["post_attention_layernorm.weight"]).reshape(-1, cfg.H)
    rl = F.linear(h, W["mlp.router.weight"], W["mlp.router.bias"])
    topv, topi = torch.topk(rl, cfg.top_k, dim=-1)
    topv = F.softmax(topv, dim=1, dtype=topv.dtype)
    scores = torch.zeros_like(rl).scatter_(1, topi, topv)
    out = torch.zeros_like(h)
    GU, GUb = W["mlp.experts.gate_up_proj"], W["mlp.experts.gate_up_proj_bias"]
    D, Db = W["mlp.experts.down_proj"], W["mlp.experts.down_proj_bias"]
    for e in torch.unique(topi).tolist():
        tok = (topi == e).any(-1).nonzero(as_tuple=True)[0]
        gu = h[tok] @ GU[e] + GUb[e]
        gate, up = gu[..., ::2], gu[..., 1::2]
        gate = gate.clamp(max=cfg.swiglu_limit)
        up = up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
        glu = gate * torch.sigmoid(gate * cfg.swiglu_alpha)
        oe = ((up + 1) * glu) @ D[e] + Db[e]
        out.index_add_(0, tok, (oe * scores[tok, e, None]).to(h.dtype))
    return x + out.view(B, s, cfg.H)


def _qwen3_attn(cfg, W, h, cos, sin, kv, pos, tree):
    """Qwen3 attention: per-head RMSNorm on q and k before rope, no biases. Mask logic as in Qwen2."""
    B, s, _ = h.shape
    q = rms(cfg, F.linear(h, W["self_attn.q_proj.weight"]).view(B, s, cfg.nh, cfg.hd), W["self_attn.q_norm.weight"]).transpose(1, 2)
    k = rms(cfg, F.linear(h, W["self_attn.k_proj.weight"]).view(B, s, cfg.nkv, cfg.hd), W["self_attn.k_norm.weight"]).transpose(1, 2)
    v = F.linear(h, W["self_attn.v_proj.weight"]).view(B, s, cfg.nkv, cfg.hd).transpose(1, 2)
    q = q * cos + _rot(q) * sin
    k = k * cos + _rot(k) * sin
    K, V = kv
    K[:, :, pos:pos + s] = k
    V[:, :, pos:pos + s] = v
    if tree is not None:
        mask = tree if tree.shape[1] == pos + s else torch.cat([torch.ones(s, pos, dtype=torch.bool), tree], dim=1)
        causal = False
    elif s == 1:
        mask, causal = None, False
    elif pos == 0:
        mask, causal = None, True
    else:
        mask, causal = torch.ones(s, pos + s, dtype=torch.bool).tril(diagonal=pos), False
    o = F.scaled_dot_product_attention(q, K[:, :, :pos + s], V[:, :, :pos + s], attn_mask=mask, is_causal=causal, enable_gqa=True)
    return F.linear(o.transpose(1, 2).reshape(B, s, cfg.nh * cfg.hd), W["self_attn.o_proj.weight"])


def _qwen3_layer(cfg, W, x, cos, sin, kv, pos, tree=None):
    x = x + _qwen3_attn(cfg, W, rms(cfg, x, W["input_layernorm.weight"]), cos, sin, kv, pos, tree)
    h = rms(cfg, x, W["post_attention_layernorm.weight"])
    return x + F.linear(F.silu(F.linear(h, W["mlp.gate_proj.weight"])) * F.linear(h, W["mlp.up_proj.weight"]), W["mlp.down_proj.weight"])


def _qwen3moe_layer(cfg, W, x, cos, sin, kv, pos, tree=None):
    """Qwen3-MoE: Qwen3 attention + sparse MoE (softmax over all experts, top-k, renormalised, SwiGLU experts)."""
    x = x + _qwen3_attn(cfg, W, rms(cfg, x, W["input_layernorm.weight"]), cos, sin, kv, pos, tree)
    B, s, _ = x.shape
    h = rms(cfg, x, W["post_attention_layernorm.weight"]).reshape(-1, cfg.H)
    probs = F.softmax(F.linear(h, W["mlp.gate.weight"]), dim=1, dtype=torch.float32)
    topv, topi = torch.topk(probs, cfg.top_k, dim=-1)
    if cfg.norm_topk:
        topv = topv / topv.sum(-1, keepdim=True)
    topv = topv.to(h.dtype)
    out = torch.zeros_like(h)
    for e in torch.unique(topi).tolist():
        rows, slot = (topi == e).nonzero(as_tuple=True)
        xe = h[rows]
        y = F.linear(F.silu(F.linear(xe, W[f"mlp.experts.{e}.gate_proj.weight"])) * F.linear(xe, W[f"mlp.experts.{e}.up_proj.weight"]),
                     W[f"mlp.experts.{e}.down_proj.weight"])
        out.index_add_(0, rows, (y * topv[rows, slot, None]).to(h.dtype))
    return x + out.view(B, s, cfg.H)


def layer_forward(cfg, W, x, cos, sin, kv, pos, l=0, tree=None, positions=None):
    if cfg.arch == "gpt_oss":
        return _gptoss_layer(cfg, W, x, cos, sin, kv, pos, l, tree, positions)
    if cfg.arch == "qwen3":
        return _qwen3_layer(cfg, W, x, cos, sin, kv, pos, tree)
    if cfg.arch == "qwen3_moe":
        return _qwen3moe_layer(cfg, W, x, cos, sin, kv, pos, tree)
    return _qwen2_layer(cfg, W, x, cos, sin, kv, pos, tree)


# ---- framed messages: [u32 header len][header json][u64 payload len][payload] ----

def send_msg(sock, hdr, payload=b""):
    h = json.dumps(hdr).encode()
    sock.sendall(struct.pack("<I", len(h)) + h + struct.pack("<Q", len(payload)))
    if payload:
        sock.sendall(payload)


def _recv_exact(sock, n):
    buf = bytearray(n)
    mv, off = memoryview(buf), 0
    while off < n:
        r = sock.recv_into(mv[off:])
        if r == 0:
            raise ConnectionError("peer closed")
        off += r
    return buf


def recv_msg(sock):
    (hl,) = struct.unpack("<I", _recv_exact(sock, 4))
    hdr = json.loads(bytes(_recv_exact(sock, hl)))
    (pl,) = struct.unpack("<Q", _recv_exact(sock, 8))
    return hdr, (_recv_exact(sock, pl) if pl else b"")


def connect(host, port, timeout=300):
    deadline = time.time() + timeout
    while True:
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.5)


def to_bytes(x):
    return x.contiguous().view(torch.int16).numpy().tobytes()


def from_bytes(b, shape):
    return torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).view(*shape)


def mem_stats():
    st = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM"):
            st[k] = int(v.split()[0]) * 1024
    for f in ("memory.current", "memory.peak", "memory.max"):
        try:
            val = open(f"/sys/fs/cgroup/{f}").read().split()[0]
            st["cg_" + f.split(".")[1]] = val if val == "max" else int(val)
        except (OSError, ValueError):
            pass
    return st


def reset_hwm():
    try:
        open("/proc/self/clear_refs", "w").write("5")
    except OSError:
        pass
