"""Page-cache residency of files (mincore). Usage: python3 fincore.py file..."""
import ctypes, ctypes.util, mmap, os, sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
tot = res = 0
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    size = os.fstat(fd).st_size
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    n = (size + 4095) // 4096
    vec = (ctypes.c_ubyte * n)()
    r = libc.mincore(addr, size, vec)
    libc.munmap(addr, size)
    os.close(fd)
    k = sum(v & 1 for v in vec)
    tot += n
    res += k
    print(f"{os.path.basename(p):28s} {100 * k / n:5.1f}% resident")
print(f"total: {100 * res / max(tot, 1):.1f}% of {tot * 4096 / 1e9:.2f} GB resident")
