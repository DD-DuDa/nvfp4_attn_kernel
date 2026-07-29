# Warp Parallelism for Low-Bit Decode

## Why FlashAttention's partitioning is wrong here

Low-bit KV moves far fewer bytes, so the kernel stops being memory-bound and becomes
compute-bound. **The roofline moves, and the warp layout tuned against the old roofline is
no longer the right one.** This is the core claim, and it is why inheriting FlashAttention's
partitioning unchanged leaves large gains on the table.

FlashAttention assigns a single warp along N to do register-level softmax and the `PV`
matmul, with warps spread along M. Two things break in low-bit decode:

1. **M is empty.** Decode query length is 1 (typically `< 16` even with GQA packing), so
   warps allocated along M have nothing to work on.
2. **Every warp stalls in turn.** Small warp tiles of K or V must traverse N sequentially,
   and dequantization sits on the critical path of each step. Nsight Compute confirms the
   added DQ raises memory-access stalls and depresses both compute throughput and Tensor
   Core utilization (paper Fig. 4b).

On Blackwell the stall source changes but does not disappear. Native NVFP4 removes
dequantization, yet the second matmul needs P back in low precision:

```
P_f16 = softmax(Q_f4 K_f4^T)
O_f16 = Quant(P_f16) V_f4
```

That on-the-fly **re-quantization of P after softmax is a new serialization point**, and
the paper calls it out explicitly as the Blackwell-specific analogue of the dequantization
stall.

## The fix: W_m = 1, spend the warps on N

Constrain warps along M to one and reallocate everything to N. With several warps along N,
the SM warp scheduler has independent work available: while one warp is dequantizing (or
re-quantizing P), others issue MMA.

```cpp
// csrc/bit_decode/src/include/kernel_traits.h:117-125
using TiledMma = TiledMMA<
    typename Base::MMA_Atom_Arch,
    Layout<Shape<Int<1>,_4,_1>>,     // W_m = 1, W_n = 4, W_k = 1
    Tile<Int<16>, Int<128>, _16>>;

using TiledMmaKV_i4 = TiledMMA<
    typename Base::MMA_Atom_Arch,
    Layout<Shape<Int<1>,_4,_1>>,
    Tile<Int<16>, Int<32>, _16>>;
```

FlashAttention-2 would write `Layout<Shape<Int<kNWarps>,_1,_1>>` here. The whole
difference is which axis the warps live on.

Note this couples back to layout: `W_n` appears in `N_r = P_n x W_n x R`, so changing the
warp layout changes the required residual block size.

## The consequence: register-level softmax stops working

Once a row is spread across warps, two things break:

1. `rowmax` needs a value that no single warp owns.
2. The resulting P distribution does not match what the `PV` MMA expects for operand A.

Both are fixed by going through shared memory, with two buffers:

- `sTMP ∈ R^{W_n}` — cross-warp reduction for the row-wise maximum.
- `sAcc ∈ R^{T_m x T_n}` — stages P from Tensor Core registers, then reloads it via
  `ldmatrix` so it is correctly aligned for the subsequent MMA.

Since `W_n` is small, `sTMP` can share `sAcc`'s shared-memory allocation.

### Algorithm 1, cooperative softmax

```
1: S_i = Q_i K_j^T
2: m_i^new = max(m_i, rowmax(S_i, sTMP))     <- two-level: intra-warp then cross-warp
3: P_i = exp(S_i - m_i^new)
4: sAcc = tiled_copy_r2s(P_i)                <- registers to shared memory
5: P_i' = tiled_copy_s2r(sAcc)               <- back, now MMA-aligned
6: O_i^new = P_i' V_j + diag(e^{m_i - m_i^new}) O_i
```

Steps 4 and 5 look like a pointless round trip. They are not: they are a **layout
correction**, converting the accumulator distribution produced by the `QK` MMA into the
operand-A distribution required by the `PV` MMA.

The reduction in step 2 is two-level — intra-warp shuffle first, then shared memory:

```cpp
// csrc/bit_decode/src/include/softmax.h
float val = warp_reduce_acc(src(i), op);   // intra-warp, groups of 4 threads
if (lane_id % 4 == 0) reduce_tmp(row, warp_id) = val;
__syncthreads();
if ((lane_id % 4) == 0) {
    float group_val = reduce_tmp(row, 0);
    for (int w = 1; w < 4; w++) { /* combine across warps */ }
}
```

And the `sAcc` round trip:

```cpp
// csrc/bit_decode/src/flash_fwd_kernel.h:442-455
auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomAcc{}, tiled_mma);
auto tCsAcc_r2s = r2s_thr_copy_c.partition_D(sAcc_residual);
cute::copy(r2s_tiled_copy_c, tCrAcc_r2s, tCsAcc_r2s);
__syncthreads();
Tensor tSrAcc      = thr_mma_residual.partition_fragment_A(sAcc_residual);
Tensor tSsAcc_view = smem_thr_copy_Acc.partition_S(sAcc_residual);
```

On Hopper, WGMMA reads operands directly from shared memory, so step 5 disappears.

## The evidence

Table III, 4-bit:

| `W_n` | Cooperative softmax | Latency | TC utilization | Correct |
|---:|---|---:|---:|---|
| 1 | no | 3.746 ms | 10.91% | yes |
| 4 | no | 0.610 ms | 19.71% | **no** |
| 4 | yes | 0.613 ms | 19.66% | yes |

Three readings, in order of importance:

1. Widening N is worth **6.1x**. This is the single largest structural win in the paper.
2. Widening N *without* cooperative softmax is **incorrect**. The middle row is a trap:
   it looks like a clean win on the timer and produces wrong answers. Any experiment that
   widens N must carry a numerical gate, or it will report a fake success.
3. Cooperative softmax costs **0.5%**. Correctness here is nearly free, so there is no
   real trade-off to agonize over.

Even after the fix, Tensor Core utilization is only 19.66% — widening N is necessary but
far from sufficient, and the remaining headroom is what the pipeline work addresses.

## Pipelining, once the warp layout is right

With CUDA cores doing quantize/dequantize and Tensor Cores doing MMA, overlap them at
register level: while slice `i` is in the MMA, slice `i+1` is being loaded by `ldmatrix`
and dequantized. Combined with `cp.async` for global-to-shared movement, this keeps a
continuous producer-consumer flow.

Reported effect: dequantization drops from roughly half of kernel time (Atom, QServe) to
under 15% at 4-bit and 35% at 2-bit.
