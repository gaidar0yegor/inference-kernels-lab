# inference-kernels-lab

Hand-written Triton kernels for the memory-bound parts of a transformer block, measured
honestly against the two baselines that actually matter: unfused PyTorch **and** `torch.compile`.

Everything in this README was produced by the scripts in this repo, on the GPU named below.
No number here is theoretical.

```
GPU    NVIDIA GeForce RTX 3060 Ti (sm_86, 8 GB, GDDR6)
Stack  torch 2.6.0+cu124 · triton 3.2.0 · CUDA 12.4
Host   single-GPU, no NVLink, no multi-node — deliberately a *small* target
```

## What's here

| kernel | what it fuses | file |
|---|---|---|
| fused residual-add + RMSNorm | `y = rmsnorm(x + residual) * weight` in one pass | `kernels/fused_rmsnorm.py` |
| fused SwiGLU | `silu(gate) * up` — kills the 4·hidden intermediate | `kernels/fused_swiglu.py` |
| fused RoPE | Llama-convention rotary embedding, one pass over q/k | `kernels/fused_rope.py` |

These three sit around the matmuls in every Llama/Qwen/Mistral-style block, and all three are
pure **memory-bound** ops — so the only thing that matters is bytes moved. That is exactly why
they are the right place to start, and exactly why they are the hardest place to win.

## Method

- **Roofline measured, not assumed:** the ceiling used for every "% peak" column is a
  device-to-device copy benchmark in the same process (409.5 GB/s here), not the datasheet
  448 GB/s. A kernel can legitimately land just above the copy number; it cannot land at 134%.
- **Exact byte accounting** for the fused kernels (the kernel reads x, writes y — we know it),
  and the *real* unfused torch implementation as the baseline, so its extra traffic is whatever
  torch actually does rather than an estimate we invented.
- **Correctness first:** `tests/test_correctness.py` checks every kernel against a PyTorch
  reference (fp16 tolerance 2e-2), plus a rotation-invariant check for RoPE (a true rotation
  must preserve the row norm) and saturation checks for `silu` at ±30.

## Results vs unfused PyTorch

```
kernel                     shape                     unfused      fused  speedup    GB/s  %peak   max_err
---------------------------------------------------------------------------------------------------------
fused add + RMSNorm        8192x4096 (fp16)        1689.7us   488.0us    3.46x    413   101%  1.56e-02
fused SwiGLU (silu(g)*u)   16384x8192 (fp16)       3253.6us  1943.4us    1.67x    414   101%  7.81e-03
fused RoPE                 131072x128 (fp16)       1450.8us   255.2us    5.69x    395    96%  3.91e-03
naive eager fp32 rmsnorm (5+ passes) 8192x4096     4803.3us   488.0us    9.84x
```

All three fused kernels sit at ~100% of measured memory bandwidth. They are at the roofline:
**there is no more room on this GPU, and chasing "more speedup" from here would be dishonest.**

## The result that matters: `torch.compile` already does this

`compare_torch_compile.py` — because a fused kernel is only interesting if it beats what
Inductor gives you for free:

```
add+RMSNorm (8192x4096)   eager 1697.8us   torch.compile  495.9us   triton  488.2us   1.02x
SwiGLU (16384x8192)       eager 3252.6us   torch.compile 1943.2us   triton 1943.2us   1.00x
RoPE (131072x128)         eager 1449.9us   torch.compile  245.6us   triton  255.8us   0.96x
```

**Parity — and slightly *behind* on RoPE.** Inductor fuses pointwise chains just as well as a
hand-written kernel, because on a single GPU these ops are trivial to schedule: one load, one
store, done. Anyone claiming a large win over `torch.compile` on ops like these is either
measuring the eager baseline and calling it the competitor, or measuring wrong.

The honest conclusion, and the reason this repo exists: **hand-written kernels earn their keep
where the compiler cannot reach** —

- **dequant + matmul fusion (W4A16/W8A16)** — Inductor cannot fuse weight dequantization *into*
  a cuBLAS GEMM; a custom kernel can, and that is real bandwidth saved on every token;
- **paged / flash attention kernels** — block-table indirection, no materialized score matrix;
- **fused sampling** — top-k/top-p over a 128k vocab without materializing logits;
- **persistent kernels / occupancy tuning** for small batch sizes, where launch overhead and
  tail effects dominate;
- non-standard dtypes and layouts the compiler has no rule for.

That is what the next kernels here will be. The point of publishing the parity result first is
that it tells you which numbers in this repo you can trust.

## Bugs the tests caught (the reason to keep them)

- **RoPE convention mismatch, silently.** The first reference used the GPT-NeoX *interleaved*
  layout (`repeat_interleave`) while the kernel implemented the Llama *half-split* layout
  (`cat(freqs, freqs)`). Shapes match, values don't — 2.7 absolute error, no crash, no warning.
  Only a reference test against a real implementation finds this class of bug.
- **A wrong bandwidth number.** The first bench version counted 6 byte-streams for a kernel that
  moves 4 (it double-counted the weight vector). It reported "134% of peak" — which is
  physically impossible and was the tell. Fixed accounting now in `bench.py`.

## Reproduce

```bash
export PYTHONPATH=.
python tests/test_correctness.py       # 6 passed
python bench.py                        # writes results.md
python compare_torch_compile.py        # the torch.compile comparison
```

## What this is not

- Not a kernel expert's portfolio. It is a systems/inference engineer's first pass at the kernel
  layer, with the honest baselines attached.
- Not multi-GPU, not NVLink, not Hopper/Blackwell features (TMA, FP8/FP4 tensor cores, warp
  specialization), not CUDA-C. Those are the next steps, not claims.
- Not integrated into a serving stack. Standalone ops with measured numbers, nothing more.
