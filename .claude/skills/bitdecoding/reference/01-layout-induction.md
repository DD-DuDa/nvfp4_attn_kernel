# Inducing a Valid, Efficient Low-Bit Layout

## The problem

Tensor Core instructions impose a strict value-to-thread mapping on register fragments,
and that mapping is interleaved. Quantization, by contrast, naturally packs values
*contiguously* per thread. The two disagree.

Concretely (paper Fig. 3): two FP16 values that thread T0 owns under `mma.m16n8k16` may be
quantized and packed as eight consecutive low-bit values. After unpacking they no longer
sit where the MMA expects them. The result is not a crash — it is **wrong numbers**.

Three compounding factors:

1. Fragment layouts differ across instructions and generations (`mma.m16n8k8` vs
   `mma.m16n8k16` vs Hopper `wgmma.m64n64k16` vs Blackwell `tcgen05`).
2. Lower bit widths make the mismatch worse, and native low-precision formats add
   block-scaling factors with their own mandated layouts.
3. Naive `static_cast` from low-bit to FP16 is slow, so even a *correct* layout can be an
   *inefficient* one.

Why the weight-quantization playbook does not apply: Ladder and Marlin fix this with a
separate layout-transformation kernel, which is fine for weights (static, transformed
once offline) and unusable for KV cache (dynamic, appended every decode step). Measured
cost of that approach:

| Phase | Marlin | Ladder | BitDecoding |
|---|---:|---:|---:|
| Prefill | 58.02 ms | 4.79 ms | **0.0599 ms** |
| Decode | 0.41 ms | 0.65 ms | **0.008 ms** |

## Method 1: induce the layout with the load instruction

`ldmatrix` deposits data into registers *already* in the Tensor Core interleaved layout.
So: load the high-precision KV with `ldmatrix`, do the Tensor Core work, and then have
**each thread quantize and pack its own registers in place**. The packed low-bit data now
implicitly carries the interleaving. When it is later loaded back with the same
`ldmatrix` variant and unpacked, the values land exactly where the MMA wants them.

No global reshape. No transformation kernel. The hardware instruction does the layout
design for you.

This is realized as two kernels:

- **Residual Kernel** — fuses compute, quantize, and pack for newly generated FP16 KV,
  writing layout-compatible low-bit data straight to global memory.
- **Packing Kernel** — fuses dequantize with compute, consuming that cache.

## The mirroring invariant

The Packing Kernel must mirror the Residual Kernel's instruction configuration:

1. the same `ldmatrix` variant,
2. the same `mma` variant,
3. the same warp-tiling configuration.

If any of the three drifts, the unpacked values silently misalign. **This is the single
most important correctness property of the whole design**, and it is a property of two
kernels *jointly*, so neither can be changed in isolation.

Generalized: whoever writes the quantized cache and whoever reads it are coupled through
the instruction configuration, not merely through a data format. A repository that
quantizes K/V in one kernel and consumes them in another must treat that pair as a single
unit under change.

## Method 2: size the residual block to fill fragments

Tensor Cores work on warp tiles. A partially populated tile wastes the unit. So the
high-precision residual buffer must be a whole number of fragments:

```
N_r = P_n x W_n x R
```

| Symbol | Meaning | Example |
|---|---|---|
| `P_n` | elements per warp tile along N | 8 under `mma.m16n8k16` |
| `W_n` | warps along N | 4 |
| `R = w / b` | packing ratio: word size over bit width | 16/4 = 4 for INT4 |

Worked: 4-bit gives `8 x 4 x 4 = 128`; 2-bit gives `8 x 4 x 8 = 256`. This matches the
implementation exactly:

```cpp
// csrc/bit_decode/src/include/kernel_traits.h:83-84
static constexpr int kBlockN_pack     = num_bits == 4 ? 128 : 256;
static constexpr int kBlockN_residual = kBlockN_pack;
```

The KV cache is then split `X = X_pack ∪ X_res`, with `X_res = X[L - N_r:]` held in half
precision. During decode, new K/V append to the residual buffer; when it reaches `N_r` it
is quantized and packed as one aligned unit.

A useful side effect: this partitioning makes **channel-wise quantization along seq_len
and tensor-wise along the hidden dimension** fall out naturally within a residual block,
which is what lets one implementation serve both granularities.

Overhead is small because `seq_len >> N_r` (typically 32K+ against `N_r <= 256`), and it
shrinks further as context grows.

## Method 3: remap for fast dequantization

Layout-compatible is not the same as dequantization-friendly. `static_cast` from INT4 to
FP16 is expensive. BitDecoding casts packed data to INT32 and maps to the interleaved
pattern `75316420`, which permits `lop3` bitwise conversion that simultaneously matches
the Tensor Core access pattern.

```cpp
// csrc/bit_decode/src/include/dequantize.h:59-72
__device__ inline FragA lop3_dequant(int q) {
    int lo_1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);        // 0,4
    int hi_1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);        // 1,5
    int lo_2 = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, LO, EX);  // 2,6
    int hi_2 = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, HI, EX);  // 3,7
```

**This method is bypassed on Blackwell**, where native micro-scaling formats consume
packed 4-bit directly and no dequantization step exists.

## Method 4: derive everything from one configuration

Order of derivation, which is also the order in which design decisions must be made:

1. GPU architecture fixes the `ldmatrix` and `mma` variants.
2. Those plus `W_n` and the bit width fix `N_r` via the formula above.
3. Residual and Packing kernels are both generated from that one configuration.

Because step 3 consumes step 1, changing the MMA variant silently changes the required
residual size and the packing layout. Configuration is not a tuning knob here; it is a
correctness contract.
