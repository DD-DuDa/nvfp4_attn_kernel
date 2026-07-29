---
name: blackwellGPU
description: Use when writing, debugging, or optimizing CUDA / CUTLASS / CuTeDSL kernels for NVIDIA B200 (Blackwell, SM100, sm_100). Triggers on tcgen05, UMMA, TMEM (Tensor Memory), FlashAttention-4 / FA4, NVFP4 / MXFP4 / MXFP8 block-scaling, CTA pairs / 2-CTA MMA, cluster shape (2,1), distributed shared memory (DSMEM), warp specialization on SM100, async pipelines (PipelineTmaUmma / PipelineUmmaAsync / PipelineTmaStore), tcgen05.alloc / tcgen05.cp / tcgen05.ld / tcgen05.commit, SBO/LBO SMEM descriptors, sub-byte GEMM, or any "how do I run this on B200" question. Read this BEFORE writing B200 kernel code.
---

# Blackwell B200 (SM100) Programming Reference

NVIDIA's data-center Blackwell (SM100, B200, GB200) breaks the Hopper programming model in five places that you must understand before writing or porting a kernel:

1. **TMEM** — a new on-chip memory tier the tensor cores read/write directly; accumulators no longer live in registers.
2. **UMMA (`tcgen05.mma`)** replaces WGMMA (`wgmma.mma_async`); only **one thread** issues it.
3. **CTA pairs** — two SMs in a cluster cooperate on one MMA tile (`cta_group::2`), splitting the M-dim accumulator and operand traffic.
4. **Three-level async pipeline** — TMA → UMMA → Epilogue → TMA-store, each with its own mbarrier ring.
5. **Asymmetric scaling** — tensor-core throughput jumped 2.25× from H100→B200, but the SFU (`MUFU.EX2`) and SMEM bandwidth did NOT. Softmax and SMEM traffic are now first-order bottlenecks.

If you skip any of these, your kernel either (a) won't compile, (b) will silently corrupt accumulators, or (c) will leave 50%+ of peak FLOPs on the floor.

> Note: SM120 (consumer Blackwell, RTX 50 / GeForce) is NOT the same as SM100. SM120 has **no Tensor Memory**. This skill targets SM100 only.

## When to use

**Use this skill when:**
- Writing/porting a GEMM, attention, or convolution kernel for B200
- Reading or modifying `cutlass/examples/python/CuTeDSL/blackwell/*` or `cutlass/examples/cute/tutorial/blackwell/*`
- Debugging an SM100 kernel (silent accumulator corruption, TMEM OOB, mbarrier deadlock, cluster sync hang, multicast mask wrong)
- Working with NVFP4 / MXFP4 / MXFP8 block-scaled MMA
- Porting an FA3 kernel to FA4
- Touching `tcgen05.*` PTX, `cute::TMEM::Allocator*`, `umma_arrive*`, `cluster_sync`, `Sm100MmaPeerBitMask`, `PipelineTmaUmma`, `PipelineUmmaAsync`

**Don't use for:**
- Hopper (SM90 / H100) — those kernels use WGMMA, accumulators in registers, no TMEM. Use the cutedsl skill + Hopper docs.
- Consumer Blackwell (SM120, RTX 50) — no TMEM, different programming model.
- Pure host-side / CUDA-runtime code that doesn't touch the SM100 ISA.

**Companion skill:** `cutedsl` (`.claude/skills/cutedsl/skill.md`) — covers CuTe layout algebra, the DSL surface, and the universal idioms (TMA atom construction, mbarrier ops). Read both.

---

## Architecture cheatsheet (B200, SM100)

| Resource | Per SM | Notes |
|---|---|---|
| SM count (B200) | 148 | per-die; GB200 is 2 dies |
| HBM | — | 183 GiB (B200), HBM3e |
| Tensor Cores (BF16) | **8192 ops/cycle** | 2.25× vs H100 |
| Tensor Cores (FP4) | **higher still** | NVFP4 / MXFP4 hardware paths |
| SFU (`MUFU.EX2`) | **16 ops/cycle** | unchanged vs H100 — softmax bottleneck |
| SMEM bandwidth | **128 B/cycle** | unchanged vs H100 |
| SMEM | 228 KB/SM | (slightly larger than H100) |
| **TMEM** | **256 KB/SM** | new; 128 lanes × 512 cols × 32-bit |
| Max cluster | **16** (opt-in, B200) | portable max 8 |
| Registers | 65536 × 32-bit/SM | unchanged |

**The asymmetric-scaling consequence:** if you copy an FA3 kernel to B200 unchanged, softmax + SMEM traffic dominate and you stall the tensor cores. FA4's design exists because of this gap (Colfax: "softmax is no longer 'just the thing between the two matmuls', it is a bottleneck").

---

## Start from a working example, never a blank file

| Goal | Read first |
|---|---|
| Dense GEMM (1-CTA, BF16/FP16) | `cutlass/examples/cute/tutorial/blackwell/02_mma_async_sm100.cu` and `examples/python/CuTeDSL/blackwell/dense_gemm.py` |
| 1-CTA + TMA multicast | `cutlass/examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu` |
| **Pair-UMMA (2-CTA)** | `cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu` |
| **Persistent block-scaled GEMM (NVFP4 / MXFP)** | `cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py` (load-bearing reference for ~80% of B200 kernel patterns) |
| **Grouped block-scaled GEMM (MoE-style)** | `cutlass/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py` |
| Sub-byte (FP4/FP6) GEMM | `cutlass/examples/cute/tutorial/blackwell/05_subbyte_gemm_sm100.cu` |
| **FlashAttention-4** | `https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute` |
| Pipeline class internals | `cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py` and `include/cutlass/pipeline/sm100_pipeline.hpp` |
| Atom traits / arch primitives | `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/atom/copy_traits_sm100.hpp`, `include/cute/arch/copy_sm100.hpp`, `include/cute/arch/tmem_allocator_sm100.hpp` |

If you write a B200 kernel without reading at least one of `04_mma_tma_2sm_sm100.cu` or `dense_blockscaled_gemm_persistent.py` first, you will reinvent boilerplate badly and almost certainly miss a sync.

---

## TMEM (Tensor Memory) — the SM100-defining feature

**Physical layout (per SM):**
- 256 KB total
- Logically: **128 lanes × 512 columns × 32-bit cells**
- Address packing: `bits[31:16] = lane`, `bits[15:0] = column`
- **Each warp accesses only 32 of the 128 lanes** (warp 0 → lanes 0–31, warp 1 → 32–63, …). This is why the epilogue needs an entire warpgroup, and why `make_tmem_copy` is hardcoded to 4 warps.

**Allocation rules:**
- Allocate via `tcgen05.alloc` (PTX) / `cute::TMEM::Allocator1Sm{}.allocate(...)` (CuTe) / `cutlass.utils.TmemAllocator(...)` (CuTeDSL).
- Units: **columns**. Count must be a power of 2 and **≥ 32**.
- A **single warp** must own both `alloc` and `dealloc` for a given allocation.
- Call `tcgen05.relinquish_alloc_permit` (`tmem_allocator.release_allocation_lock()`) once you guarantee no further allocations from this CTA.
- 2-CTA kernels need an extra `two_cta_tmem_dealloc_mbar_ptr` mbarrier in SMEM for the dealloc handshake.

**Canonical 1-CTA allocator usage (CUTLASS C++):**
```cpp
cute::TMEM::Allocator1Sm tmem_allocator{};
if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
}
__syncthreads();
tCtAcc.data() = shared_storage.tmem_base_ptr;
// ... mainloop, epilogue ...
if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
}
```

**TMEM access PTX (you don't usually write these directly — but you must recognize them):**
| PTX | Direction | Notes |
|---|---|---|
| `tcgen05.alloc` / `dealloc` | host-side of TMEM | column-granular |
| `tcgen05.relinquish_alloc_permit` | finalize alloc | declares no more allocations |
| `tcgen05.cp` | SMEM → TMEM | async; ordered with `tcgen05.mma` on same pipeline |
| `tcgen05.ld.sync.aligned.SHAPE.NUM.b32` | TMEM → registers | warp-wide; SHAPE ∈ `{.16x64b, .16x128b, .16x256b, .32x32b}`; NUM ∈ `{.x1 … .x128}` |
| `tcgen05.st` | registers → TMEM | warp-wide |
| `tcgen05.commit` | mbarrier arrival | independent pipelines for `cta_group::1` vs `::2` |
| `tcgen05.mma` | UMMA | see next section |

**TMEM aliasing (FA4 trick):** TMEM is small. FA4 backward reuses columns: S/P share one column range, dP/dS/dQ share another. Plan TMEM column offsets up front and document them (FA4 paper / `flash_attn/cute/`).

**Debug:** compile with `nvcc --g-tensor-memory-access-check` — it traps on uninitialized or out-of-bound TMEM access.

---

## UMMA (`tcgen05.mma`) — the new tensor-core instruction

**PTX skeleton:**
```
tcgen05.mma.cta_group.kind                  [d-tmem], a-desc, b-desc, idesc, enable-input-d;
tcgen05.mma.cta_group.kind                  [d-tmem], [a-tmem], b-desc, idesc, enable-input-d;
tcgen05.mma.cta_group.kind.block_scale.scale_vectorsize
                                            [d-tmem], a-desc, b-desc, idesc,
                                            [scale-A-tmem], [scale-B-tmem], enable-input-d;

.cta_group ∈ { .cta_group::1, .cta_group::2 }
.kind      ∈ { .kind::f16, .kind::tf32, .kind::f8f6f4,
              .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.scale_vectorsize ∈ { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }
```

**Hard rules:**
- **Operand A**: TMEM **or** SMEM
- **Operand B**: SMEM **only**
- **Accumulator D**: TMEM **only** (unlike WGMMA, which used registers)
- **Only ONE thread** issues the instruction. Even in a 2-CTA pair, only one thread in one CTA launches it.
- Dense-GEMM K extent in operand bytes is fixed at **32 B in the K direction** (so e.g. K=64 for NVFP4, K=32 for FP8).

**Shape rules:**
- 1-CTA: `M ∈ {64, 128}`, N ≤ 256, K = 16 (16-bit) / 32 (FP8) / 64 (FP4).
  - When `M=64`: N must be a multiple of 8.
  - When `M=128`: N must be a multiple of 16.
- 2-CTA (pair-UMMA): `M ∈ {128, 256}`, accumulator split in M across the pair.

**Atom names** (`cutlass/include/cute/atom/mma_traits_sm100.hpp`):
| Variant | Atom |
|---|---|
| 1-CTA, A and B in SMEM | `SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, M, N, UMMA::Major::K, UMMA::Major::K>` |
| 1-CTA, A in TMEM | `SM100_MMA_F16BF16_TS<...>` |
| 1-CTA, FP8/FP6/FP4 | `SM100_MMA_F8F6F4_SS<...>` |
| 2-CTA pair | `SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K, UMMA::Major::K>` |

**CuTeDSL counterpart** (`cutlass.cute.nvgpu.tcgen05`):
```python
from cutlass.cute.nvgpu.tcgen05 import (
    OperandSource, OperandMajorMode, CtaGroup,
    MmaF16BF16Op, MmaF8F6F4Op, MmaMXF8Op, MmaMXF4Op, MmaMXF4NVF4Op,
)
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    ab_dtype, OperandMajorMode.K, OperandMajorMode.K,
    acc_dtype, CtaGroup.ONE,           # or CtaGroup.TWO for pair-UMMA
    mma_tiler_mnk[:2], OperandSource.SMEM,
)
```

**Mainloop pattern (1-CTA, CuTe C++):**
```cpp
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // ... TMA load A, B into SMEM, wait on tma mbarrier ...
    if (elect_one_warp) {
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;   // <- accumulate after first block
        }
        cutlass::arch::umma_arrive(&shared_storage.mma_barrier);  // tcgen05.commit
    }
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
}
```
**Critical:** `UMMA::ScaleOut::Zero` overwrites the accumulator (use on first iteration); `UMMA::ScaleOut::One` accumulates (use afterwards). In CuTeDSL: `tiled_mma.set(tcgen05.Field.ACCUMULATE, True)`.

**Epilogue (TMEM → registers → SMEM/GMEM):**
```cpp
TiledCopy tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
ThrCopy thr_t2r = tiled_t2r.get_slice(threadIdx.x);
Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
Tensor tDgD   = thr_t2r.partition_D(tCgD);
Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));
copy(tiled_t2r, tDtAcc, tDrAcc);          // tcgen05.ld
// ... cast / activation / scale, then RMEM → SMEM (stmatrix-style) → GMEM via TMA store
```
`make_tmem_copy` requires the entire warpgroup (4 warps) because each warp sees only 32 lanes.

---

## CTA pairs / 2-CTA MMA (pair-UMMA)

**Why bother:** a 256×256 pair-UMMA does the same FLOPs as two independent 128×256 1-CTA UMMAs but transfers **half the operand-B data** (B is multicast across the pair). Empirically 2-CTA gives ~5–6% on dense GEMM at 8192³ and is the foundation of FA4 backward.

**Pair geometry:**
- Cluster shape `(2, 1)` (or `(2, X)` for outer parallelism).
- CTAs that differ in the **0-th bit** of their cluster index are pairs (0&1, 2&3, …).
- **Even CTA = leader.** Pair-UMMA must be issued from a single thread of the leader.
- `M` of the pair-UMMA tile is split across the two CTAs; each CTA holds half the accumulator in its own TMEM and half each of A, B.

**Cluster layout (4-D `vmnk`):**
```cpp
auto cluster_layout_vmnk =
    tiled_divide(make_layout(cluster_shape),
                 make_tile(typename TiledMMA::AtomThrID{}));
// Mode 0 = "v" (within-pair index), modes 1-3 = m, n, k cluster coords
```

**CuTeDSL counterpart:**
```python
cluster_layout_vmnk = cute.tiled_divide(
    cute.make_layout((*cluster_shape_mn, 1)),
    (tiled_mma.thr_id.shape,))
is_leader_cta = mma_coord_vmnk[0] == 0
```

**Atom + TMA op for pair-UMMA:**
```cpp
TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float, 256, 256,
                               UMMA::Major::K, UMMA::Major::K>{});
auto tma_atom_a = make_tma_atom_A_sm100<SM100_TMA_2SM_LOAD_MULTICAST>(...);
```

**Multicast bitmasks** (16-bit, one bit per cluster CTA):
```cpp
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, ...);
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, ...);
uint16_t mma_mcast_mask_a = create_tma_multicast_mask<0,2>(cluster_layout_vmnk, ...);
uint16_t mma_mcast_mask_b = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, ...);
uint16_t mma_mcast_mask_c = mma_mcast_mask_a | mma_mcast_mask_b;
```
**Critical rule:** the MMA bitmask is `(TMA bitmask) | (peer's MMA bitmask)`. NOT just the OR of TMA masks — that mistake is silent and produces wrong results.

**Peer-CTA SMEM addressing:**
```cpp
constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
```
Bit 24 of the SMEM address corresponds to bit 0 of the CTA-in-pair index. Clearing bit 24 redirects to the leader's SMEM. Used to find the leader's mbarrier from the non-leader.

**TMA `cta_group::2`:** when set, a TMA load can arrive at the mbarrier of either CTA in the pair. `cp.async.bulk.tensor.dim.dst.src.completion_mechanism.cta_group::2` is the underlying PTX. CUTLASS atom: `SM100_TMA_2SM_LOAD_MULTICAST`.

**Sync model:**
- `cute::cluster_sync()` — full cluster-wide barrier
- `cluster_arrive() + cluster_wait()` — split-phase
- `cutlass::arch::umma_arrive_multicast_2x1SM(&mma_barrier, mma_mcast_mask_c)` — pair-UMMA arrival

**`tcgen05.commit` pipelines for `cta_group::1` and `cta_group::2` are independent.** Issuing a pair-UMMA but waiting on `umma_arrive` (without `_multicast_2x1SM`) deadlocks.

---

## The three-level async pipeline (SM100-specific)

Hopper had 2 levels (TMA → MMA). SM100 adds an epilogue + TMA-store level:

```
TMA load ─[ab pipeline]─▶ UMMA ─[acc pipeline]─▶ Epilogue ─[c pipeline]─▶ TMA store
```

Each level uses a paired `full_bar` / `empty_bar` mbarrier ring and a phase bit that flips on wraparound.

**CuTeDSL pipeline classes** (`cutlass.cute.nvgpu.cpasync.pipeline` / `pipeline.sm100`):
```python
from cutlass.cute.nvgpu import pipeline

ab_pipeline = pipeline.PipelineTmaUmma.create(
    barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
    num_stages=num_ab_stage,
    producer_group=tma_warp_group,
    consumer_group=mma_warp_group,
    tx_count=num_tma_load_bytes,
    cta_layout_vmnk=cluster_layout_vmnk,
    defer_sync=True,
)

acc_pipeline = pipeline.PipelineUmmaAsync.create(
    barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
    num_stages=num_acc_stage,
    producer_group=mma_warp_group,
    consumer_group=epilogue_warp_group,
    cta_layout_vmnk=cluster_layout_vmnk,
    defer_sync=True,
)

c_pipeline = pipeline.PipelineTmaStore.create(
    num_stages=num_c_stage,
    producer_group=epilogue_warp_group,
)
```

**CUTLASS C++ counterpart:**
```cpp
using AbPipeline  = cutlass::PipelineTmaUmmaAsync</*Stages=*/3, ClusterShape>;
using AccPipeline = cutlass::PipelineUmmaAsync</*Stages=*/2, ClusterShape>;
using CPipeline   = cutlass::PipelineTmaStore</*Stages=*/2>;
```

**Producer pattern (TMA warp):**
```python
ab_pipeline.producer_acquire(ab_state, peek)
cute.copy(tma_atom_a, gA[(None, ab_state.count)],
          sA[(None, ab_state.index)],
          tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_state),
          mcast_mask=mcast_mask_a)
# ... copy B, SFA, SFB
ab_state.advance()
```

**Consumer pattern (MMA warp, leader-only for 2-CTA):**
```python
ab_pipeline.consumer_wait(ab_state, peek)
for k_block in cutlass.range(num_kblocks, unroll_full=True):
    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[...].iterator)  # block-scaled only
    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[...].iterator)
    cute.gemm(tiled_mma, tCtAcc, tCrA[...], tCrB[...], tCtAcc)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
ab_pipeline.consumer_release(ab_state)
```

**TMA arrive PTX-equivalent** (one thread per producer):
```python
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_count)
# tx_count = sum of all bytes the TMA copies arriving on this barrier
```

**UMMA commit PTX-equivalent:**
```python
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(barrier, ...)   # tcgen05.commit
```

**Phase-bit toggle** is non-negotiable — every wait flips `phase ^= 1` for that mbarrier slot. Forgetting one wraparound flip causes silent stall or premature consume.

**Reference fences:**
- `cute::tma_store_fence()` → `fence.proxy.async.shared::cta`
- `cute.arch.fence_view_async_tmem_load()` (CuTeDSL) — order TMEM loads vs subsequent ops
- `cute.arch.fence_proxy()` — generic async-proxy fence

---

## Warp-specialization layouts on SM100

The patterns below are **role assignments** within the CTA's 4–8 warps. Pick one based on what your kernel does.

### Pattern 1: Simple 1-CTA dense GEMM (TMA + MMA split — Veitner 2-warp variant)
```
warp 0 (TMA):    issues TMA loads, optional tile scheduler
warp 1 (MMA):    consumes ab pipeline, runs cute.gemm + tcgen05.commit, produces acc
warps 2+ (EPI):  consume acc pipeline, TMEM→RMEM→SMEM→GMEM
```
Gain over warp-uniform: ~4–5% on 8192³ BF16.

### Pattern 2: Persistent block-scaled GEMM (the canonical SM100 layout)
Used in `dense_blockscaled_gemm_persistent.py` and the CUTLASS Sm100 kernels. Roles by `warp_idx`:
| `warp_idx` | Role |
|---|---|
| `tma_warp_id` | TMA producer + tile scheduler + (grouped) tensormap updates |
| `mma_warp_id` | UMMA consumer of ab; UMMA producer of acc; SF SMEM→TMEM via `tcgen05.cp` |
| `< mma_warp_id` (epilogue WG) | Acc consumer; TMEM→RMEM→SMEM; produce c pipeline |
| `epilog_warp_id[0]` | Issues TMA stores (`cp.async.bulk.tensor.S2G`) |

For **2-CTA** kernels: the TMA warp runs in **all** CTAs of the pair, but the MMA warp runs only on the **leader**. Non-leader still participates in the cluster mbarrier dance.

### Pattern 3: FA3-style ping-pong (Hopper, but still relevant for B200 SF / fwd partial overlap)
1 producer warpgroup (TMA, 24 regs) + 2 consumer warpgroups (240 regs each), with `setmaxnreg`:
```cpp
cutlass::arch::warpgroup_reg_dealloc<24>();   // producer
cutlass::arch::warpgroup_reg_alloc<240>();    // consumer
```
Cross-WG handoff via a `NamedBarrierGemm` enum and `cute.arch.barrier_arrive(barrier_id, num_threads)`.

### Pattern 4: FA4 (B200 attention)
- **Forward:** ping-pong **2× Q tiles and 2× O tiles per CTA** to maximize MMA / softmax overlap. Because accumulators live in TMEM (not registers), multiple MMAs can be in flight concurrently.
- **Backward:** 2-CTA tile (M=256, N=K=128 for dV/dK/dP/S; M=128, 2N=256 for dQ). While computing softmax for tile *j*, issue dK and dQ MMAs for tile *j-1*. Half of dS is exchanged between the two CTAs of the pair via **DSMEM** (distributed shared memory).
- **TMEM aliasing:** S/P share a column range; dP/dS/dQ share another. Plan offsets up front.
- **Software exp2** in CUDA cores (Cody-Waite range reduction + Horner polynomial: `p₀=1.0, p₁≈0.6951, p₂≈0.2276, p₃≈0.0771`) so MUFU.EX2 stops bottlenecking. **Conditional rescale** in online softmax: skip rescale when `m_j - m_{j-1} ≤ τ` to cut non-MMA work.
- Result: 1605 TFLOP/s BF16 on B200 (71% utilization), 1.3× cuDNN 9.13, 2.7× Triton.

---

## SMEM swizzle modes (carries forward from Hopper, but with sub-byte caveats)

Eight canonical atoms; pick by contiguous-direction byte width:

| Atom | Contig bytes | Picker rule (`major_mode_size_bits`) |
|---|---|---|
| `Layout_K_INTER_Atom<T>` (no swizzle) | 16 | 128 bits |
| `Layout_K_SW32_Atom<T>` | 32 | 256 bits |
| `Layout_K_SW64_Atom<T>` | 64 | 512 bits |
| `Layout_K_SW128_Atom<T>` | 128 | 1024 bits |
| (same four with `MN_…`) | for MN-major | |

Pick `Swizzle<B,4,3>` so `2^B * 128b == major_mode_bits`: `Identity → 128b`, `SW32 → 256b`, `SW64 → 512b`, `SW128 → 1024b`.

**Tile-to-shape:**
```cpp
auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
```

**Sub-byte rule (FP4 / FP6):** **only 128 B swizzle or no swizzle** is supported by `cuTensorMapEncodeTiled` for FP4/FP6 datatypes. Picking SW32/SW64 silently falls back or fails.

**SBO and LBO** — `tcgen05.mma`'s SMEM descriptors take two byte-offset operands:
- **LBO (Leading Byte Offset)**: stride for one atom along the **major** mode (K).
- **SBO (Swizzle Byte Offset)**: stride for one atom along the **minor** mode (M/N).
- For **M-major + SW128**, the directions reverse (LBO walks M, SBO walks K).
- LBO/SBO are configured by `cuTensorMapEncodeTiled()` and packed into the 64-bit SMEM descriptor consumed by `tcgen05.mma`.

You almost never write the descriptor by hand — `cute::tile_to_shape` + `make_tma_atom*` derive it — but you must understand it when debugging "MMA produces garbage".

---

## Block-scaled MMA (NVFP4 / MXFP4 / MXFP8 / MXFP6)

| Format | Operand dtype | Scale-vec length | Scale dtype |
|---|---|---|---|
| `mxf8` | E5M2 / E4M3 | 32 | UE8M0 (= 2^x, -127 ≤ x ≤ 127) |
| `mxf6` | E3M2 / E2M3 | 32 | UE8M0 |
| `mxf4` | E2M1 | 32 | UE8M0 |
| **`nvf4`** (NVFP4) | E2M1 | **16** | **UE4M3** (sign always 0) |

Math: `D = (A · scaleA) @ (B · scaleB) + C`.

**MMA op selection (CuTeDSL):**
```python
if ab_dtype in {Float8E4M3FN, Float8E5M2}:
    mma_op = MmaMXF8Op(ab_dtype, (*mma_tiler_mn, 32), cta_group, a_source,
                       a_leading_mode, b_leading_mode)
elif ab_dtype == Float4E2M1FN and sf_vec_size == 32:
    mma_op = MmaMXF4Op((*mma_tiler_mn, 64), cta_group, a_source)
elif ab_dtype == Float4E2M1FN and sf_vec_size == 16:
    mma_op = MmaMXF4NVF4Op(sf_dtype, (*mma_tiler_mn, 64), cta_group, a_source)
tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))
```

**MMA-K instruction shape is fixed:** 64 for FP4 (`mxf4` / `nvf4`), 32 for FP8 (`mxf8`), per the PTX `tcgen05-matrix-shape` table.

**Scale factors live in TMEM.** The flow is:
1. TMA loads SF tiles into SMEM along with the operand tiles.
2. The **MMA warp** copies SMEM → TMEM via `tcgen05.cp` (atom `Cp4x32x128bOp`, multicast `.warpx4` so all 32-lane partitions see them).
3. `cute.gemm` is called with `tiled_mma.set(tcgen05.Field.SFA, ptr)` / `SFB` set per K-block.

**Critical ordering rule:** `tcgen05.cp` (SF copy) and `tcgen05.mma` are async on the **same internal pipeline**, so the **same warp** must issue both. Splitting the SF copy and MMA across different warps races.

**SF SMEM atom (`BlockScaledBasicChunk`, K-major):**
```python
atom_shape  = ((32, 4), (sf_vec_size, 4))   # 32 lanes × 4 cols × (vec × 4)
atom_stride = ((16, 4), (0, 1))             # stride 0 inside K mode = scale broadcast
sf_layout = cute.tile_to_shape(BlockScaledBasicChunk(sf_vec_size).layout,
                               Shape, (2, 1, 3))   # order: K, M, L
```

**TMEM column accounting** (per the persistent block-scaled example):
```python
num_sfa_tmem_cols = (cta_tile_shape_mnk[0] // 32) * 4   # sf_atom_mn=32, mma_inst_tile_k=4
num_sfb_tmem_cols = (cta_tile_shape_mnk_sfb[1] // 32) * 4
num_sf_tmem_cols  = num_sfa_tmem_cols + num_sfb_tmem_cols
num_acc_tmem_cols = (cta_tile_shape_mnk[1] * num_acc_stage if not overlapping_accum
                    else cta_tile_shape_mnk[1] * 2 - num_sf_tmem_cols)
```

**Pair-UMMA SF semantics:** SFA is **split** in M across the pair (each CTA holds half, multicast 4× across its 4 lane-groups). SFB is **duplicated** to both CTAs (also multicast 4×).

**`bN=192` SFB stride trick:** every odd N-tile steps by 1 SFB tile, every even N-tile steps by 2.

**FP4/FP8 dequant PTX** (when you need to bring values back to FP16 in CUDA cores, e.g. for FA4 epilogue):
```
cvt.rn.f16x2.e2m1x2 $0, b;     // FP4 nibble-pair → 2× FP16
cvt.rn.f16x2.e4m3x2 $0, h0;    // FP8 byte-pair  → 2× FP16
```
Strategy: extract packed sub-vectors (8 → 4 → 2 → 1), zero-extend to Int8/Int16 for sub-byte FP4, call inline asm. CuTeDSL has helpers in `cutlass.cute.arch.numeric_conversion` (`cvt_f4e2m1x2_to_f16x2`, `cvt_f8e4m3x{2,4,8}_to_f16x{2,4,8}`). Custom converters can be ~10% faster than the default codegen on FP4/FP8 GEMV.

---

## Sub-byte (FP4 / FP6) GEMM packing rules

- TMA datatypes: `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` (FP4), `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` (FP6).
- SMEM stored "as if byte-typed": **16 packed elements + padding to a 16-byte boundary**. Allocate SMEM space for sub-byte operands as if they were byte operands.
- TMA: base address **32 B aligned**, contiguous-dim size a **multiple of 128 elements**, swizzle ∈ {none, 128 B}.
- CUTLASS types: `cutlass::float_e2m1_unpacksmem_t`, `cutlass::float_e3m2_unpacksmem_t`, `cutlass::float_e2m3_unpacksmem_t`, plus `type_erased_dynamic_float4_t`, `type_erased_dynamic_float6_t`.
- The MMA atom's `idesc_.a_format_` / `.b_format_` is set at **runtime** (not template) for sub-byte mixed-input MMAs:
  ```cpp
  tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
  ```

---

## Persistent + Stream-K scheduling on SM100

Same scheduler as Hopper (carries over). Use it for any kernel whose problem size doesn't tile evenly.

```python
tile_sched_params = utils.PersistentTileSchedulerParams(
    num_ctas_mnl, cluster_shape_mnl)
grid = utils.StaticPersistentTileScheduler.get_grid_shape(
    tile_sched_params, max_active_clusters)
```

**Threadblock swizzle (1/2/4/8) for L2 reuse:**
| Swizzle | Use when |
|---|---|
| 1 | Tiny problems |
| 2 | Tile count > 57 |
| 4 | Tile count > 31 |
| 8 | Large balanced problems |

**Stream-K = persistent CTAs + fractional tiles.** Reduction:
- **Turnstile** (deterministic): CTA 0 writes workspace, CTA n waits for barrier, CTA n reduces in.
- **Atomic** (nondeterministic): CTAs atomically reduce into workspace.

Use `cutlass::gemm::StreamKScheduler` in `GemmUniversal`.

**Grouped GEMM** (per-group MNK / strides / pointers in one launch — MoE-style): each SM keeps a **5-tensormap workspace**: `(num_SMs, 5, 16 i64)` = A/B/C/SFA/SFB × 128-byte tensormap × per-SM. Update only on group transitions via `tensormap_manager.update_tensormap(...)` and `cute.nvgpu.cpasync.copy_tensormap`.

---

## Recipe: porting an FA3 (Hopper) kernel to FA4 (Blackwell)

1. **Replace WGMMA → UMMA:**
   - `wgmma.mma_async` → `tcgen05.mma`
   - Atom: `SM90_64x*x16_F16F16F16_*` → `SM100_MMA_F16BF16_SS`
   - Move accumulator from RMEM → TMEM. Allocate via `Allocator1Sm`.
   - Drop `warpgroup_arrive` / `warpgroup_commit_batch` / `warpgroup_wait`; use `umma_arrive` + mbarrier wait instead.
2. **Restructure pipeline 2-level → 3-level:**
   - Add an `acc` pipeline (`PipelineUmmaAsync`) between MMA and epilogue.
   - Add a `c` pipeline (`PipelineTmaStore`) for TMA stores.
3. **Reassign warps:**
   - 1 thread launches UMMA (not a whole warpgroup like WGMMA).
   - Whole warpgroup needed for epilogue (TMEM-lane visibility constraint).
4. **Move softmax to CUDA cores** with software `exp2` (Cody-Waite + Horner) — MUFU.EX2 is the bottleneck on B200.
5. **Use ping-pong of 2× Q/O tiles per CTA** to keep the tensor cores fed while CUDA cores do softmax.
6. **For backward: go 2-CTA.** Cluster `(2,1)`, atom `_2x1SM_SS`, exchange dS halves via DSMEM.
7. **Plan TMEM aliasing** before coding: S/P share columns; dP/dS/dQ share another set.
8. **Consider conditional rescale** in online softmax to cut non-MMA work.

---

## Recipe: writing a B200 GEMM from scratch (1-CTA, BF16, no block-scaling)

1. **Tile sizes:** `(M, N, K) = (128, 256, 64)` is a good starting point. K=64 = 4 K-blocks of 16 per UMMA inst.
2. **SMEM layouts:** K-major, `Layout_K_SW128_Atom<bf16>` for both A and B. 3-stage pipeline.
3. **TMA atoms:** `cute.nvgpu.cpasync.make_tma_tile_atom(CopyBulkTensorTileG2SOp(), gA, sA_layout, smem_tile)`. No multicast for 1-CTA-no-cluster.
4. **MMA atom:** `SM100_MMA_F16BF16_SS<bf16, bf16, float, 128, 256, K, K>`; `make_tiled_mma`.
5. **TMEM:** allocate `Sm100TmemCapacityColumns` columns. Place the 128×256 accumulator at column offset 0.
6. **Pipelines:** `PipelineTmaUmmaAsync<3>` for ab; `PipelineUmmaAsync<2>` for acc.
7. **Warp roles:** warp 0 = TMA producer; warp 1 = MMA (single thread issues `cute.gemm` + `umma_arrive`); warps 4–7 = epilogue WG.
8. **Mainloop:** `for k_tile: { ab.acquire; tma copy; mma loop with ScaleOut::Zero→One; umma_arrive }`.
9. **Epilogue:** `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc)`; TMEM→RMEM→SMEM (stmatrix-style atom)→TMA store.
10. **Sanity:** compile with `--g-tensor-memory-access-check`, run with `compute-sanitizer racecheck` and `compute-sanitizer initcheck`.

---

## Common mistakes & gotchas

| Mistake | Symptom | Fix |
|---|---|---|
| Issuing UMMA from many threads | Hang or corrupted accumulator | `if (elect_one_warp && elect_one_thr)` (1-CTA), or leader-CTA + elect_one for 2-CTA |
| Forgetting to flip `mma_barrier_phase_bit` | Deadlock or premature consume | `phase ^= 1` after every `wait_barrier` on that mbarrier slot |
| Using `umma_arrive` on a pair-UMMA | Deadlock | Use `umma_arrive_multicast_2x1SM` with the c-bitmask |
| MMA bitmask = OR of TMA bitmasks (only) | Wrong results | `mma_mcast = mma_mcast_a \| mma_mcast_b \| peer_mma_mcast`. The peer's MMA bitmask matters. |
| Allocating TMEM from multiple warps | Hang at dealloc | One warp owns alloc + dealloc; others wait via the alloc mbarrier |
| Epilogue running on a single warp | Some lanes of accumulator are unread | Whole warpgroup — `make_tmem_copy` is hardcoded to 4 warps |
| Wrong swizzle for sub-byte | Silent garbage | FP4 / FP6 only support 128-B swizzle or none |
| SF copy on a different warp than MMA | Race | Same warp issues `tcgen05.cp` and `tcgen05.mma` |
| Forgetting `ScaleOut::Zero` on first K-block | Accumulator carries garbage from prior alloc | Set `Zero` at `k_block==0`, flip to `One` after |
| 2-CTA without `cluster_sync()` before pipeline init | Random hang | Cluster sync before any pipeline / mbarrier observation |
| Not setting `defer_sync=True` on Pipeline create | Doubled cluster sync, perf cliff | Use `defer_sync=True` and call your own `cluster_arrive_relaxed` / `cluster_wait` once |
| Accumulator pointer not reset between K-tiles | Wrong result | The accumulator is at a fixed TMEM column; reset only the SF columns / leftover columns |
| `tcgen05.commit` on wrong cta_group | Pipeline leak | `commit` must match the `mma`'s `cta_group::1` vs `::2` |

---

## Performance touchstones (sanity floor when benchmarking)

| Workload | Hardware | Result |
|---|---|---|
| Dense BF16 GEMM 8192³, 2-CTA + multicast | B200 | ≈1400 TFLOP/s (warp-spec'd) |
| FA4 BF16 attention | B200 | 1605 TFLOP/s (71% util); 1.3× cuDNN 9.13; 2.7× Triton |
| FA3 BF16 attention | H100 | ~740 TFLOP/s (75% util); 1.5–2.0× FA-2 |
| FA2 + FP8 attention | H100 | >1 PFLOP/s for head=256 |
| Warp-spec gain over warp-uniform GEMM | B200 BF16 8192³ | +4–5% |
| 2-CTA gain over 1-CTA GEMM | B200 BF16 8192³ | +5–6% |
| Custom FP4/FP8 dequant cvt | B200 GEMV | ~10% over default |

If your B200 BF16 GEMM kernel is below ~1.0 PFLOP/s at 8192³, you have a real problem (likely missing cluster sync, wrong swizzle, or single-warp epilogue).

---

## Reference libraries

| Library / repo | Why |
|---|---|
| **NVIDIA CUTLASS** — `https://github.com/NVIDIA/cutlass` | Canonical SM100 atom/pipeline source; copy from `examples/python/CuTeDSL/blackwell/` and `examples/cute/tutorial/blackwell/` |
| **flash-attention** — `https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute` | FA4 reference (CuTeDSL, SM90+SM100) |
| **Colfax cfx-article-src** — `https://github.com/ColfaxResearch/cfx-article-src` | Colfax tutorial source (TMA, persistent kernels, EVT, transpose) |
| **CuTeDSL examples (SM100)** — `https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell` | Persistent block-scaled GEMM, grouped block-scaled, dense GEMM |
| **PTX docs** — `https://docs.nvidia.com/cuda/parallel-thread-execution/` | Sections: `tcgen05-matrix-shape`, `tcgen05-mma-scale-factor-a-layout-1x`, `cp.async.bulk.tensor`, `mbarrier` |
| **Thien Tran's tcgen05 deep-dive** — `https://gau-nernst.github.io/tcgen05/` | Companion to the PTX docs; explains `tcgen05.mma` operands and SMEM descriptor SBO/LBO with diagrams |
| **MatmulTutorial SM100** — `https://github.com/KnowingNothing/MatmulTutorial/tree/main/examples/matmul/this-sm100` | Minimal end-to-end SM100 GEMM in raw CUTLASS |

---

## External blog reference index

### Colfax Research — start here for fundamentals

| URL | Topic |
|---|---|
| `https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/` | **The foundational SM100 post — TMEM + UMMA basics** |
| `https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/` | **Pair-UMMA and 2-CTA explained end-to-end** |
| `https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/` | FP4 / FP6 packing, TMA datatypes, swizzle constraints |
| `https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/` | **NVFP4 / MXFP block-scaling on hardware (SF in TMEM, tcgen05.cp)** |
| `https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/` | **FA4 design — bottlenecks, software exp2, ping-pong, 2-CTA backward, DSMEM dQ** |
| `https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/` | FlexAttention API on FA-CuTeDSL (score-mod / mask-mod) |
| `https://research.colfax-intl.com/flexattention-flashattention-4-fast-and-flexible-external/` | PyTorch FlexAttention with FA4 backend |
| `https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/` | Persistent + Stream-K (carries to SM100) |
| `https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/` | Pipelining (Hopper baseline, idioms reused) |
| `https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/` | Hopper WGMMA — read for contrast |
| `https://research.colfax-intl.com/tutorial-hopper-tma/` | TMA mastery (universal, applies to SM100) |
| `https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/` | FA3 algorithm — port-from baseline |
| `https://research.colfax-intl.com/epilogue_visitor_tree/` | EVT (Hopper-only at time of writing) |
| `https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/` | Bank-conflict-free swizzle worked example |

### Simon Veitner — concrete CuTeDSL idioms with code

| URL | Topic |
|---|---|
| `https://veitner.bearblog.dev/2-cta-gemm-on-b200/` | **2-CTA GEMM in CuTeDSL — minimal diff from 1-CTA** |
| `https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/` | **Three-level pipeline build (TMA→UMMA→Epilogue)** |
| `https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/` | NVFP4/MXFP MMA atom + TMEM column accounting |
| `https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/` | **TMA-warp / MMA-warp split** |
| `https://veitner.bearblog.dev/grouped-block-scaled-gemm-intro/` | Grouped GEMM intro (MoE-style) |
| `https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/` | Grouped GEMM — host-side construction |
| `https://veitner.bearblog.dev/grouped-blockscaled-gemm-kernel/` | Grouped GEMM — kernel + per-SM tensormap workspace |
| `https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/` | `BlockScaledBasicChunk` SF tensor atom |
| `https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/` | **FP4/FP8 ↔ FP16 dequant via inline PTX `cvt.*`** |
| `https://veitner.bearblog.dev/nvfp4-gemv/` | NVFP4 GEMV (CUDA-core path, no UMMA) |
| `https://veitner.bearblog.dev/nvfp4-gemv-improved/` | NVFP4 GEMV — extra K blocks, atomic and SMEM-2D reductions |
| `https://veitner.bearblog.dev/sbo-and-lbo-explained-visually/` | **SBO/LBO in `tcgen05.mma` SMEM descriptor** |
| `https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/` | Swizzle modes + visual bank-conflict explanation |
| `https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/` | Pipeline class API (same on SM100) |
| `https://veitner.bearblog.dev/consumer-producer-pattern-on-h100-in-cutedsl/` | Low-level mbarrier / phase-bit primitives |
| `https://veitner.bearblog.dev/pingpong-in-the-cutedsl-with-quack/` | Ping-pong scheduling with `NamedBarrier` |

---

## Quick decision flowchart

When picking your starting tile / cluster:

```
Is operand size in K (bytes) ≥ 32?            ──▶ if no, you cannot use UMMA — go BF16 GEMV / CUDA cores
Is N small (≤ 64) and M small?                ──▶ 1-CTA with M=64 atom; consider GEMV variant
Is K-major operand layout fixed by upstream?  ──▶ K-major SW128 atom, 16-bit datatype
Is operand A or B reused across CTAs?         ──▶ cluster (2,X) + multicast TMA
Is the workload bandwidth-bound on B?         ──▶ 2-CTA pair-UMMA (halves B traffic)
Is softmax / SFU on the critical path?        ──▶ FA4 pattern: software exp2, conditional rescale,
                                                  ping-pong 2× tiles, MMA-while-softmax-runs
Block-scaled (NVFP4 / MXFP)?                  ──▶ make_blockscaled_trivial_tiled_mma + tcgen05.cp
                                                  for SF copy, set tcgen05.Field.SFA/SFB per K-block
Many small problems (MoE)?                    ──▶ grouped block-scaled GEMM with per-SM
                                                  5-tensormap workspace
```

---

## Red flags — STOP and reread the relevant section

- "I'll just port this WGMMA loop" → you must read TMEM + UMMA sections; accumulators move from RMEM to TMEM.
- "I can issue UMMA from a whole warp" → no, **one thread**.
- "The mma multicast mask is the OR of A and B TMA masks" → no, also OR with the **peer's MMA mask**.
- "I'll allocate TMEM from any warp" → no, **one warp** owns alloc + dealloc.
- "Only one warp does the epilogue" → no, **a whole warpgroup** (4 warps) — TMEM lanes split across warps.
- "I'll skip `cluster_sync` if I'm not using DSMEM" → for 2-CTA + cluster mbarriers, you still need cluster sync at init.
- "MUFU.EX2 is fast" → on B200 it is the bottleneck for attention. Do exp2 in CUDA cores with a polynomial.
- "I'll use SW32 swizzle for FP4" → not supported. 128-B or none only.

If any of those thoughts cross your mind: stop, locate the matching section above, and read it.

---

## How to use this skill

1. **Before writing** any B200 kernel: read the "Architecture cheatsheet", "Start from a working example", and the section matching your kernel type (GEMM / FA / block-scaled / sub-byte).
2. **While writing**: keep the "Common mistakes" table open. Most B200 kernel bugs are listed there.
3. **When debugging**: compile with `--g-tensor-memory-access-check`, then walk the "Red flags" list.
4. **Before claiming a kernel is fast**: compare to the "Performance touchstones" table. If you are below the floor, you are leaving FLOPs on the floor.
5. **For deeper detail** on any topic, jump to the linked Colfax or Veitner post — they have full source repos and worked examples.

This skill is a map, not the territory. The territory is the CUTLASS source, the PTX docs, and the FA4 repo. When this skill and the source disagree, **trust the source**.
