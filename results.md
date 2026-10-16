# Results — NVIDIA GeForce RTX 3060 Ti

- torch 2.6.0+cu124, triton 3.2.0
- measured copy bandwidth (roofline): **409.5 GB/s** (theoretical GDDR6 ~448 GB/s)
- generated 2026-10-16 21:32 on the host below

```
fused add + RMSNorm        8192x4096 (fp16)        1689.7us   488.0us    3.46x    413   101%  1.56e-02
fused SwiGLU (silu(g)*u)   16384x8192 (fp16)       3253.6us  1943.4us    1.67x    414   101%  7.81e-03
fused RoPE                 131072x128 (fp16)       1450.8us   255.2us    5.69x    395    96%  3.91e-03
naive eager fp32 rmsnorm (5+ passes) 8192x4096               4803.3us   488.0us    9.84x
```

Naive eager fp32 RMSNorm baseline: 4803.3 us (9.84x slower than fused).
