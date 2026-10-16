"""Fused residual-add + RMSNorm forward — Triton kernel vs PyTorch.

Why this kernel: RMSNorm is on the hot path of every modern LLM (Llama/Qwen/Mistral).
Unfused it is 3+ kernel launches and 3+ full passes over the row (x+r materialised,
rstd computed, then scaled) — pure memory-bandwidth waste. Fused: one pass in,
one pass out.

Measured on: RTX 3060 Ti (sm_86, 8 GB, GDDR6 ~448 GB/s theoretical).
Run: /opt/venv-ml/bin/python fused_rmsnorm.py
"""

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "cuda"
EPS = 1e-6


@triton.jit
def _fused_add_rmsnorm_fwd(
    x_ptr, r_ptr, w_ptr, y_ptr,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r                                     # fused residual add

    var = tl.sum(s * s, axis=0) / n_cols          # mean of squares, one reduction
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols,
             (s * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x, r, w, eps=EPS, block=None):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = block or triton.next_power_of_2(N)
    _fused_add_rmsnorm_fwd[(M,)](x, r, w, y, N, eps, BLOCK=BLOCK)
    return y


def torch_add_rmsnorm_fp16(x, r, w, eps=EPS):
    """Reference: unfused fp16 path — 3 full passes over the row, 3 kernel launches.
    This is the honest comparison: same dtype, same maths, just not fused."""
    s = x + r                                    # pass 1: read x, r -> write s
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)   # pass 2
    return (s * rstd * w).to(x.dtype)            # pass 3: read s, w -> write y


def torch_add_rmsnorm(x, r, w, eps=EPS):
    """Also shown: what naive eager code does — fp32 promotion, 5+ passes."""
    s = (x.float() + r.float())
    return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


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


def peak_bandwidth_gbs(nbytes=256 << 20, iters=100):
    """Measured device-to-device copy bandwidth = realistic memory ceiling."""
    a = torch.empty(nbytes // 4, device=DEV)
    b = torch.empty_like(a)
    ms = measure(lambda: b.copy_(a), iters=iters, warmup=25)
    return 2 * nbytes / (ms * 1e-3) / 1e9


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"torch {torch.__version__} | triton {triton.__version__}")

    copy_bw = peak_bandwidth_gbs()
    print(f"measured copy bandwidth: {copy_bw:.1f} GB/s (theo. spec ~448 GB/s)\n")

    shapes = [(4096, 4096), (8192, 4096), (16384, 4096), (8192, 8192)]
    print(f"{'shape':>16} {'max_err':>9} {'torch fp32':>18} {'torch fp16 unfused':>22} "
          f"{'triton fused':>22} {'vs fp16':>8}")
    for M, N in shapes:
        x = torch.randn(M, N, device=DEV, dtype=torch.float16)
        r = torch.randn(M, N, device=DEV, dtype=torch.float16)
        w = torch.randn(N, device=DEV, dtype=torch.float16)

        ref = torch_add_rmsnorm(x, r, w)
        out = fused_add_rmsnorm(x, r, w)
        err = (ref.float() - out.float()).abs().max().item()
        assert torch.allclose(ref.float(), out.float(), atol=2e-2, rtol=2e-2), f"mismatch {err}"

        # Ideal traffic for a fully fused kernel: read x, read r, write y (+ tiny w).
        ideal = M * N * 2 * 3 + N * 2

        t_fused = measure(lambda: fused_add_rmsnorm(x, r, w))
        t_fp16 = measure(lambda: torch_add_rmsnorm_fp16(x, r, w))
        t_fp32 = measure(lambda: torch_add_rmsnorm(x, r, w))

        def bw(t, passes):
            return passes * ideal / (t * 1e-3) / 1e9

        fused_bw = bw(t_fused, 1)
        fp16_bw = bw(t_fp16, 3)   # unfused fp16 = 3x the traffic
        fp32_bw = bw(t_fp32, 6)   # eager fp32 ≈ 6 passes

        print(f"M={M:>6} N={N:>5} {err:9.2e} "
              f"{t_fp32*1000:8.1f}us({fp32_bw:5.0f}GB/s) "
              f"{t_fp16*1000:9.1f}us({fp16_bw:5.0f}GB/s) "
              f"{t_fused*1000:9.1f}us({fused_bw:5.0f}GB/s {100*fused_bw/copy_bw:3.0f}%pk) "
              f"{t_fp16/t_fused:6.2f}x")

    print(f"\npeak = measured copy bandwidth {copy_bw:.0f} GB/s "
          f"(theoretical GDDR6 448 GB/s). Fused kernel sits at the roofline: "
          f"it is memory-bound, so ~100% of copy BW is the ceiling, not 9.8x more.")


if __name__ == "__main__":
    main()
