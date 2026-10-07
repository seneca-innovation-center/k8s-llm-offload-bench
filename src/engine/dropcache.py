"""Evict the weight files from the Linux page cache (no root needed: posix_fadvise DONTNEED on each file).

Run on the host before an mmap experiment, so the pod has to fault the weights in itself and the
pages are charged to its own cgroup. Otherwise pages cached by an earlier pod are served for free
and a memory limit measures nothing.
"""
import glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
n = 0
for p in glob.glob(f"{d}/*.safetensors"):
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    n += 1
print(f"dropped page cache for {n} files in {d}")
