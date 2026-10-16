"""The honest competitor: torch.compile (TorchInductor).

A fused Triton kernel is only interesting if it beats what the compiler already
does for free. Inductor does fuse pointwise chains, so this is the number that
decides whether the hand-written kernels are worth anything.

Run: PYTHONPATH=. /opt/venv-ml/bin/python compare_torch_compile.py
"""

import torch
import triton

from kernels import fused_rmsnorm as rms
from kernels import fused_swiglu as swi
from kernels import fused_rope as rope

DEV = "cuda"


def measure(fn, iters=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench(name, eager_fn, triton_fn, compiled_fn):
    t_eager = measure(eager_fn)
    t_tri = measure(triton_fn)
    t_comp = measure(compiled_fn)
    print(f"{name:<28} eager {t_eager*1000:8.1f}us   "
          f"torch.compile {t_comp*1000:8.1f}us   "
          f"triton {t_tri*1000:8.1f}us   "
          f"triton vs compile {t_comp/t_tri:5.2f}x")
    return t_eager, t_comp, t_tri


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"triton {triton.__version__}\n")

    x = torch.randn(8192, 4096, device=DEV, dtype=torch.float16)
    r = torch.randn(8192, 4096, device=DEV, dtype=torch.float16)
    w = torch.randn(4096, device=DEV, dtype=torch.float16)
    bench("add+RMSNorm (8192x4096)", lambda: rms.torch_add_rmsnorm_fp16(x, r, w),
          lambda: rms.fused_add_rmsnorm(x, r, w),
          torch.compile(lambda: rms.torch_add_rmsnorm_fp16(x, r, w)))

    g = torch.randn(16384, 8192, device=DEV, dtype=torch.float16)
    u = torch.randn(16384, 8192, device=DEV, dtype=torch.float16)
    bench("SwiGLU (16384x8192)", lambda: swi.torch_swiglu(g, u),
          lambda: swi.fused_swiglu(g, u),
          torch.compile(lambda: swi.torch_swiglu(g, u)))

    n_rows, head_dim = 32 * 4096, 128
    half = head_dim // 2
    xq = torch.randn(n_rows, head_dim, device=DEV, dtype=torch.float16)
    cos = torch.rand(n_rows, half, device=DEV, dtype=torch.float16) * 2 - 1
    sin = torch.rand(n_rows, half, device=DEV, dtype=torch.float16) * 2 - 1
    bench("RoPE (131072x128)", lambda: rope.torch_rope(xq, cos, sin),
          lambda: rope.fused_rope(xq, cos, sin),
          torch.compile(lambda: rope.torch_rope(xq, cos, sin)))


if __name__ == "__main__":
    main()
