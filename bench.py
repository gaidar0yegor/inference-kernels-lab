"""Benchmark runner: fused Triton kernels vs unfused PyTorch on the RTX 3060 Ti.

Every number here is measured in this process, on this GPU. Byte accounting is
exact for the fused kernels (we know precisely what the kernel reads and writes);
the unfused baselines are the real torch implementations, so their extra traffic
is whatever torch actually does — not an estimate we invent.

Writes results.md next to this file.

Run: /opt/venv-ml/bin/python bench.py
"""

import os
import time

import torch
import triton

from kernels import fused_rmsnorm as rms
from kernels import fused_swiglu as swi
from kernels import fused_rope as rope

DEV = "cuda"


def measure(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def copy_bandwidth_gbs(nbytes=256 << 20, iters=100):
    """Device-to-device copy = the realistic memory ceiling for streaming kernels."""
    a = torch.empty(nbytes // 4, device=DEV)
    b = torch.empty_like(a)
    ms = measure(lambda: b.copy_(a), iters=iters, warmup=25)
    return 2 * nbytes / (ms * 1e-3) / 1e9


def case(name, shape, dtype_str, unfused, fused, fused_bytes, err, peak):
    t_unf = measure(unfused)
    t_fus = measure(fused)
    bw = fused_bytes / (t_fus * 1e-3) / 1e9
    return {
        "kernel": name,
        "shape": shape,
        "dtype": dtype_str,
        "unfused_ms": t_unf,
        "fused_ms": t_fus,
        "speedup": t_unf / t_fus,
        "gbps": bw,
        "pct_peak": 100 * bw / peak,
        "max_err": err,
    }


def main():
    peak = copy_bandwidth_gbs()
    rows = []

    # ---------------- fused add + RMSNorm ----------------
    x = torch.randn(8192, 4096, device=DEV, dtype=torch.float16)
    r = torch.randn(8192, 4096, device=DEV, dtype=torch.float16)
    w = torch.randn(4096, device=DEV, dtype=torch.float16)
    ref = rms.torch_add_rmsnorm_fp16(x, r, w)
    out = rms.fused_add_rmsnorm(x, r, w)
    err = (ref.float() - out.float()).abs().max().item()
    rows.append(case(
        "fused add + RMSNorm",
        "8192x4096 (fp16)",
        "fp16",
        lambda: rms.torch_add_rmsnorm_fp16(x, r, w),
        lambda: rms.fused_add_rmsnorm(x, r, w),
        fused_bytes=8192 * 4096 * 2 * 3 + 4096 * 2,
        err=err, peak=peak,
    ))
    ref32 = rms.torch_add_rmsnorm(x, r, w)
    t_eager = measure(lambda: rms.torch_add_rmsnorm(x, r, w))
    eager_row = ("naive eager fp32 rmsnorm (5+ passes)", t_eager,
                 t_eager / rows[0]["fused_ms"])

    # ---------------- fused SwiGLU ----------------
    g = torch.randn(16384, 8192, device=DEV, dtype=torch.float16)
    u = torch.randn(16384, 8192, device=DEV, dtype=torch.float16)
    ref = swi.torch_swiglu(g, u)
    out = swi.fused_swiglu(g, u)
    err = (ref.float() - out.float()).abs().max().item()
    rows.append(case(
        "fused SwiGLU (silu(g)*u)",
        "16384x8192 (fp16)",
        "fp16",
        lambda: swi.torch_swiglu(g, u),
        lambda: swi.fused_swiglu(g, u),
        fused_bytes=16384 * 8192 * 2 * 3,
        err=err, peak=peak,
    ))

    # ---------------- fused RoPE ----------------
    n_rows, head_dim = 32 * 4096, 128
    half = head_dim // 2
    xq = torch.randn(n_rows, head_dim, device=DEV, dtype=torch.float16)
    cos = (torch.rand(n_rows, half, device=DEV, dtype=torch.float16) * 2 - 1)
    sin = (torch.rand(n_rows, half, device=DEV, dtype=torch.float16) * 2 - 1)
    ref = rope.torch_rope(xq, cos, sin)
    out = rope.fused_rope(xq, cos, sin)
    err = (ref.float() - out.float()).abs().max().item()
    rows.append(case(
        "fused RoPE",
        f"{n_rows}x{head_dim} (fp16)",
        "fp16",
        lambda: rope.torch_rope(xq, cos, sin),
        lambda: rope.fused_rope(xq, cos, sin),
        fused_bytes=n_rows * head_dim * 2 * 2 + n_rows * half * 2 * 2,
        err=err, peak=peak,
    ))

    # ---------------- report ----------------
    hdr = (f"{'kernel':<26} {'shape':<20} {'unfused':>12} {'fused':>10} "
           f"{'speedup':>8} {'GB/s':>7} {'%peak':>6} {'max_err':>9}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"measured copy bandwidth (roofline): {peak:.1f} GB/s "
          f"(theoretical GDDR6 ~448 GB/s)\n")
    print(hdr)
    print("-" * len(hdr))
    lines = []
    for r in rows:
        line = (f"{r['kernel']:<26} {r['shape']:<20} {r['unfused_ms']*1000:9.1f}us "
                f"{r['fused_ms']*1000:7.1f}us {r['speedup']:7.2f}x "
                f"{r['gbps']:6.0f} {r['pct_peak']:5.0f}% {r['max_err']:9.2e}")
        print(line)
        lines.append(line)
    naive = (f"{eager_row[0]:<26} {'8192x4096':<20} {eager_row[1]*1000:9.1f}us "
             f"{rows[0]['fused_ms']*1000:7.1f}us {eager_row[2]:7.2f}x")
    print(naive)
    lines.append(naive)

    if not os.environ.get("NO_WRITE"):
        with open("results.md", "w") as fh:
            fh.write(f"# Results — {torch.cuda.get_device_name(0)}\n\n")
            fh.write(f"- torch {torch.__version__}, triton {triton.__version__}\n")
            fh.write(f"- measured copy bandwidth (roofline): **{peak:.1f} GB/s** "
                     f"(theoretical GDDR6 ~448 GB/s)\n")
            fh.write(f"- generated {time.strftime('%Y-%m-%d %H:%M')} on the host below\n\n")
            fh.write("```\n" + "\n".join(lines) + "\n```\n\n")
            fh.write(f"Naive eager fp32 RMSNorm baseline: {eager_row[1]*1000:.1f} us "
                     f"({eager_row[2]:.2f}x slower than fused).\n")


if __name__ == "__main__":
    main()
