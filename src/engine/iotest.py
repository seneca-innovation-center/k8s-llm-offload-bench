"""Cold-cache check of the two loaders: bandwidth with 1/4 readers, and whether the read bytes end up in the
page cache (Cached in /proc/meminfo must not grow for a real O_DIRECT path).

    WEIGHTS=/exp/weights python iotest.py
"""
import os, threading, time

from common import load_cold, load_direct

W = os.environ.get("WEIGHTS", "/exp/weights")
files = sorted(f"{W}/{f}" for f in os.listdir(W) if f.startswith("layer_"))


def cached_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1]) / 1e6


def drop():
    for p in files:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def run(loader, nthreads, nfiles=16):
    drop()
    c0 = cached_gb()
    subset = files[:nfiles]
    total = [0] * nthreads

    def work(i):
        for p in subset[i::nthreads]:
            _, n = loader(p)
            total[i] += n
    th = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.time() - t0
    return sum(total) / dt / 1e9, cached_gb() - c0


for name, loader in (("buffered (fadvise)", lambda p: load_cold(p, evict=True)), ("O_DIRECT", load_direct)):
    for n in (1, 4):
        bw, dc = run(loader, n)
        print(f"{name:20s} {n} reader(s): {bw:5.2f} GB/s   page cache change {dc:+.2f} GB")
    bw2, dc2 = run(loader, 4)
    print(f"{name:20s} repeat 4 readers: {bw2:5.2f} GB/s   page cache change {dc2:+.2f} GB")
