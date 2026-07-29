---
name: bitdecoding
description: Design knowledge for low-bit KV-cache decode kernels, from BitDecoding (HPCA 2026). Use when designing or reviewing quantized KV layouts, deciding how packed low-bit data must align with Tensor Core fragments, choosing warp partitioning for decode attention, or diagnosing why a low-bit decode kernel underutilizes Tensor Cores — including Chinese variants ("low-bit layout 怎么设计", "warp 怎么切", "为什么 Tensor Core 用不满").
---

# Skill: Low-Bit KV Decode Design (BitDecoding)

Source: Du, Cao, Cheng, Mai, Cao, Yang. *BitDecoding: Unlocking Tensor Cores for
Long-Context LLMs with Low-Bit KV Cache.* HPCA 2026. arXiv:2503.18773.
Code: <https://github.com/OpenBitSys/BitDecoding>.

**When to use:** any decision about how quantized K/V is laid out, how it reaches Tensor
Core registers, or how warps are partitioned in a decode attention kernel.

**The one sentence version:** low-bit KV shifts the bottleneck from memory to compute, so
the layout must be induced by the hardware instructions that will consume it, and the
warp partitioning inherited from FlashAttention is wrong for decode.

---

## Two ideas that matter

### 1. Induce the layout, do not transform it

Naive approach: quantize and pack KV however is convenient, then insert a layout
transformation before the MMA. This is what Ladder and Marlin do for static weights, and
it costs 4.8 ms to 58 ms at prefill — unusable for a cache that changes every token.

BitDecoding's insight: **`ldmatrix` already places data in the Tensor Core interleaved
fragment layout.** If each thread quantizes and packs its own registers *in place*, the
packed low-bit data implicitly inherits that interleaving. On unpack it is already
correct, so the remapping cost is zero.

The correctness invariant is **mirroring**: the producer (Residual Kernel, which fuses
compute + quantize + pack) and the consumer (Packing Kernel, which fuses dequantize +
compute) must use the same `ldmatrix` variant, the same `mma` variant, and the same
warp-tiling configuration. Break the mirror and you get silently wrong values, not a
crash.

Corollary for sizing: a fragment that is only partially filled wastes the Tensor Core.
The residual (high-precision tail) block size must therefore be

```
N_r = P_n x W_n x R
```

where `P_n` is elements per warp tile along N, `W_n` is warps along N, and `R = w / b` is
the packing ratio (word size over bit width). Details in `reference/01-layout-induction.md`.

### 2. FlashAttention's warp partitioning is wrong for low-bit decode

FlashAttention allocates warps along M. That is right for prefill, where M is long. In
decode the query length is 1, so M is almost entirely padding, and meanwhile
dequantization (or, on Blackwell, re-quantization of P) stalls each warp in turn.

BitDecoding instead **fixes `W_m = 1` and spends the warps on N**. More warps along N
give the SM warp scheduler independent work to hide the quantize/dequantize latency.
In code this is literally `Layout<Shape<Int<1>,_4,_1>>` where FlashAttention-2 would
write `Layout<Shape<Int<kNWarps>,_1,_1>>`.

This breaks register-level softmax, because a row is now spread across warps. The fix is
a **cooperative softmax** using two shared-memory buffers: `sTMP` for cross-warp rowmax
reduction, and `sAcc` to stage P and reload it via `ldmatrix` so it is aligned for the PV
MMA.

The ablation is the reason to care (Table III, 4-bit):

| `W_n` | Cooperative softmax | Latency | TC utilization | Correct |
|---:|---|---:|---:|---|
| 1 | no | 3.746 ms | 10.91% | yes |
| 4 | no | 0.610 ms | 19.71% | **no** |
| 4 | yes | 0.613 ms | 19.66% | yes |

Widening N gives **6.1x**; cooperative softmax costs **0.5%** and is what makes it
correct rather than merely fast. Details in `reference/02-warp-parallelism.md`.

---

## Applying this to an SM100 NVFP4 kernel

BitDecoding targets SM80/SM89/SM90 with `ldmatrix` + `mma.m16n8k16` + `lop3`
dequantization. An SM100 kernel using native NVFP4 `tcgen05` MMA inherits the
*principles* but not the *mechanisms*. `reference/03-porting-to-sm100.md` works through
which is which; the short version:

| BitDecoding technique | Transfers to SM100 NVFP4? |
|---|---|
| Query transformation `[1,(g_q,h_kv)] -> [g_q,h_kv]` | **Yes, directly** |
| `W_m = 1`, spend warps on N | **Yes as a principle**, mechanism differs under `tcgen05` |
| Cooperative softmax via `sTMP` / `sAcc` | **Yes**, required if N-parallelism widens |
| Residual block sized to fill TC fragments | **Yes**, the formula generalizes |
| P re-quantization is a new stall source | **Yes** — the paper names this as Blackwell-specific |
| `lop3` / `75316420` remapping | **No** — the paper itself bypasses it under native NVFP4 |
| `ldmatrix`-induced packing in a fused Residual Kernel | **Principle only** — SM100 uses TMA plus hardware-mandated scale-factor layouts |
| Hopper `STSM` + `wgmma_SS` workaround | **No** — Hopper-specific |

---

## Reference

| File | Contents |
|---|---|
| `reference/01-layout-induction.md` | The four layout methods, `N_r` formula, mirroring invariant |
| `reference/02-warp-parallelism.md` | `W_m=1`/`W_n` rationale, cooperative softmax, Algorithm 1 |
| `reference/03-porting-to-sm100.md` | What transfers to this repository's kernel, and where it currently disagrees |
