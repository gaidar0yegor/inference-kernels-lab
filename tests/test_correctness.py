"""Correctness tests for the fused kernels — every kernel is checked against a
plain PyTorch reference before a single benchmark number is allowed to stand.

Tolerances: fp16 accumulation across a row (RMSNorm reduction) plus fp32 rounding
on store. 2e-2 absolute / 2e-2 relative is the standard tolerance for fp16 fused
ops (same order as torch's own fp16 kernels vs an fp32 oracle).

Run directly:   /opt/venv-ml/bin/python tests/test_correctness.py
Run with pytest: /opt/venv-ml/bin/python -m pytest tests/ -v
"""

import torch

from kernels import fused_rmsnorm as rms
from kernels import fused_swiglu as swi
from kernels import fused_rope as rope

ATOL, RTOL = 2e-2, 2e-2


def _check(ref, out, name):
    assert out.shape == ref.shape, f"{name}: shape {out.shape} != {ref.shape}"
    err = (ref.float() - out.float()).abs().max().item()
    assert torch.allclose(ref.float(), out.float(), atol=ATOL, rtol=RTOL), \
        f"{name}: max_err {err:.3e} exceeds tolerance"
    return err


def test_add_rmsnorm_fp16():
    torch.manual_seed(0)
    for M, N in [(1, 4096), (64, 1024), (4096, 4096), (8192, 4096)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        r = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        ref = rms.torch_add_rmsnorm_fp16(x, r, w)
        out = rms.fused_add_rmsnorm(x, r, w)
        _check(ref, out, f"add_rmsnorm M={M} N={N}")


def test_add_rmsnorm_fp32_inputs():
    """Kernel must not assume fp16 — dtype-generic via out tensor."""
    torch.manual_seed(0)
    M, N = 512, 2048
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    r = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = rms.torch_add_rmsnorm_fp16(x, r, w)   # same maths, fp32 inputs
    out = rms.fused_add_rmsnorm(x, r, w)
    err = _check(ref, out, "add_rmsnorm fp32")
    assert err < 1e-4, f"fp32 path should be near-exact, got {err:.3e}"


def test_swiglu():
    torch.manual_seed(0)
    for rows, width in [(1, 1024), (128, 4096), (4096, 8192), (16384, 8192)]:
        g = torch.randn(rows, width, device="cuda", dtype=torch.float16)
        u = torch.randn(rows, width, device="cuda", dtype=torch.float16)
        _check(swi.torch_swiglu(g, u), swi.fused_swiglu(g, u),
               f"swiglu rows={rows} width={width}")


def test_swiglu_large_magnitudes():
    """silu saturates for |g| >> 1 — check both tails explicitly."""
    torch.manual_seed(0)
    g = torch.cat([torch.full((1024, 64), -30.0, device="cuda", dtype=torch.float16),
                   torch.full((1024, 64), 30.0, device="cuda", dtype=torch.float16)])
    u = torch.ones_like(g)
    out = swi.fused_swiglu(g, u)
    assert torch.isfinite(out).all(), "swiglu produced inf/nan on saturated input"
    assert (out[:1024].abs() < 1e-3).all(), "silu(-30) must round to ~0"


def test_rope():
    torch.manual_seed(0)
    for n_rows, head_dim in [(1, 64), (512, 128), (32 * 4096, 128)]:
        half = head_dim // 2
        x = torch.randn(n_rows, head_dim, device="cuda", dtype=torch.float16)
        cos = torch.rand(n_rows, half, device="cuda", dtype=torch.float16) * 2 - 1
        sin = torch.rand(n_rows, half, device="cuda", dtype=torch.float16) * 2 - 1
        _check(rope.torch_rope(x, cos, sin), rope.fused_rope(x, cos, sin),
               f"rope rows={n_rows} dim={head_dim}")


def test_rope_is_norm_preserving():
    """RoPE is a rotation: it must preserve the vector norm (fp16 tolerance)."""
    torch.manual_seed(0)
    n_rows, head_dim = 1024, 128
    half = head_dim // 2
    x = torch.randn(n_rows, head_dim, device="cuda", dtype=torch.float32)
    # cos/sin for a true rotation of position p: cos(p*theta), sin(p*theta)
    pos = torch.arange(n_rows, device="cuda", dtype=torch.float32).unsqueeze(1)
    theta = torch.pow(10000.0, -torch.arange(half, device="cuda", dtype=torch.float32) / half)
    cos, sin = torch.cos(pos * theta), torch.sin(pos * theta)
    out = rope.fused_rope(x, cos, sin)
    dn = (out.norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    assert dn < 1e-3, f"rotation changed vector norm by {dn:.3e}"


if __name__ == "__main__":
    torch.cuda.init()
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
