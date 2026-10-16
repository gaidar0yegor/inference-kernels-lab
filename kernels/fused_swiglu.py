"""Fused SwiGLU activation (silu(gate) * up) — Triton kernel vs unfused PyTorch.

Why: in a SwiGLU MLP block the intermediate tensor is [batch, seq, 4*hidden] —
by far the biggest tensor in the model. Unfused, torch does:
    act = silu(gate)      # read gate, write act        (2 streams)
    out = act * up        # read act, read up, write out (3 streams)
= 5 full passes over a tensor that dominates activation memory.

Fused: one pass — read gate, read up, write out (3 streams). This is a pure
elementwise, memory-bound op, so the win is bounded by bandwidth, not maths.

Run: /opt/venv-ml/bin/python -m kernels.fused_swiglu   (or via bench.py)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # silu(g) = g * sigmoid(g); computed in fp32, stored back in the tensor dtype
    silu = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (silu * u).to(out_ptr.dtype.element_ty), mask=mask)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _swiglu_kernel[grid](gate, up, out, n, BLOCK=BLOCK)
    return out


def torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Unfused reference: silu materialises a full-size intermediate, then multiplies."""
    return torch.nn.functional.silu(gate) * up


if __name__ == "__main__":
    torch.manual_seed(0)
    # [batch*seq, 4*hidden] — e.g. Qwen3-1.7B: hidden 2048 -> intermediate 8192
    for rows, width in [(4096, 8192), (16384, 8192)]:
        gate = torch.randn(rows, width, device="cuda", dtype=torch.float16)
        up = torch.randn(rows, width, device="cuda", dtype=torch.float16)
        ref, out = torch_swiglu(gate, up), fused_swiglu(gate, up)
        err = (ref.float() - out.float()).abs().max().item()
        ok = torch.allclose(ref.float(), out.float(), atol=2e-2, rtol=2e-2)
        print(f"rows={rows:>6} width={width:>5} max_err={err:.2e} ok={ok}")
        assert ok, "swiglu mismatch"
