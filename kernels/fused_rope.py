"""Fused RoPE (rotary position embedding) — Triton kernel vs unfused PyTorch.

Why: RoPE is applied to q and k of every attention layer, every forward pass.
The unfused torch recipe is:
    rotate_half(x)  -> allocates a full-size tensor (slice + cat)     (streams)
    x*cos + rotate_half(x)*sin -> 2 muls + 1 add, more temporaries
= 5+ passes over q, and q is the tensor you least want to touch twice.

Fused: one pass — read x, read cos, read sin, write out. The rotation itself is
2 multiply-adds on the two halves of the row, so it is trivially compute-light:
again pure bandwidth.

Layout: x is [n_rows, head_dim]; the row is split into two halves
    out[:half] = x[:half]*cos - x[half:]*sin
    out[half:] = x[half:]*cos + x[:half]*sin
cos/sin are [n_rows, half] (already position-indexed; in production they broadcast
over heads — same kernel, cheaper loads).

Run: /opt/venv-ml/bin/python -m kernels.fused_rope
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, half, stride_row,
                 BLOCK_HALF: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half
    base = row * stride_row

    x1 = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + half + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + row * half + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + row * half + offs, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s

    tl.store(out_ptr + base + offs, o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + base + half + offs, o2.to(out_ptr.dtype.element_ty), mask=mask)


def fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [n_rows, head_dim]; cos/sin: [n_rows, head_dim//2]."""
    n_rows, head_dim = x.shape
    half = head_dim // 2
    out = torch.empty_like(x)
    BLOCK_HALF = triton.next_power_of_2(half)
    _rope_kernel[(n_rows,)](x, cos, sin, out, half, x.stride(0), BLOCK_HALF=BLOCK_HALF)
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def torch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Unfused reference — Llama/HF convention (half-split), matching the kernel.

    Important: HF's Llama builds cos/sin as `cat(freqs, freqs)`, i.e. the SAME
    half-frequency vector duplicated over both halves of the row, and rotate_half
    splits the row in halves. (GPT-NeoX instead interleaves — `repeat_interleave`
    — which is a different convention and needs a different kernel. Getting these
    two mixed up is a classic silent-correctness bug: the output shapes match and
    only a reference test catches it.)
    """
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return (x * cos) + (_rotate_half(x) * sin)


if __name__ == "__main__":
    torch.manual_seed(0)
    # e.g. Qwen3-1.7B: head_dim 128; 32 heads x 4096 tokens
    for n_rows, head_dim in [(32 * 512, 128), (32 * 4096, 128)]:
        half = head_dim // 2
        x = torch.randn(n_rows, head_dim, device="cuda", dtype=torch.float16)
        cos = torch.rand(n_rows, half, device="cuda", dtype=torch.float16) * 2 - 1
        sin = torch.rand(n_rows, half, device="cuda", dtype=torch.float16) * 2 - 1

        ref, out = torch_rope(x, cos, sin), fused_rope(x, cos, sin)
        err = (ref.float() - out.float()).abs().max().item()
        ok = torch.allclose(ref.float(), out.float(), atol=2e-2, rtol=2e-2)
        print(f"rows={n_rows:>7} head_dim={head_dim:>4} max_err={err:.2e} ok={ok}")
        assert ok, "rope mismatch"
