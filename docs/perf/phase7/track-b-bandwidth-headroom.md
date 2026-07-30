# How much bandwidth FP4 decode can reach with softmax removed

`softmax-ceiling-probe.md` established that deleting per-element softmax
arithmetic moves the decode kernel out of the softmax wall and into a
bandwidth regime, but only to 5.36 TB/s at `seqlen 65536, batch 32` against
FA4's 6.49 TB/s on the same shape. This document answers what the real ceiling
is, which structural parameters move the number, and by how much.

The short version: the machine can stream at 7.58 TB/s, FA4 reaches 7.04 TB/s
on a shape that tiles the SM count evenly, and the softmax-free FP4 kernel
reaches 6.90 TB/s on that same shape once the Q-stage count is dropped from two
to one. That one parameter is essentially the whole result; pulling KV depth
back from 14 to 10 adds another 1%, and everything else on the candidate list
was either already past its useful point or actively harmful.

## Method

All numbers are kernel-only time, summed from the Torch profiler through
`bench_decode.measure_kernel_breakdown`, so host dispatch is excluded from
every column. Measured at commit `2257ab8` on GPU 1 with SM clocks locked
(1965 MHz requested, 1845 MHz sustained under load), GQA 32:8 unless stated
otherwise, 20 iterations after 10 warmup.

Every kernel variation is a runtime-patched subclass in
`tests/kernel_profile/probe_bandwidth_headroom.py`. Nothing under `src/` was
touched. The probe inherits the softmax-free `softmax_step` from
`probe_softmax_ceiling.py` and adds knobs for Q stages, KV pipeline depth,
epilogue stages, split count, N-block size, and per-stream L2 pinning. The
`full_control` configuration keeps production softmax and sets no knobs; it
reproduced production to within 0.2% at every shape measured, the worst case
being 560.8 against 559.9 us for MHA 8:8 at `16384, batch 32`, so the deltas
below are attributable to the knobs and not to the copied `softmax_step`.

`bench_decode.build_inputs` re-randomizes its tensors in every process, so
cosine similarities are only comparable within a single run. Every
before-and-after cosine quoted here is a matched pair measured in the same
invocation.

Repeatability is good: `free_q1_kv10` at `seqlen 65536, batch 32` measured
392.6, 393.2 and 393.4 us in three independent process invocations, a spread
of 0.2%. The headline production pair at that shape repeats as well, 1638.6 and
1638.2 us for `q_stage = 1` against 2104.9 and 2104.1 for production in two
separate runs.

## The HBM ceiling for this access pattern

`tests/kernel_profile/probe_hbm_ceiling.py` measures streaming reads with no
attention kernel involved. The first result is a warning about method: a
`torch.sum` reduction is not a bandwidth ceiling.

| Read pattern | GiB | us | TB/s |
|---|---:|---:|---:|
| Vectorized streaming read, 8192-element tiles | 16.000 | 2265.0 | **7.58** |
| Vectorized streaming read, 4096-element tiles | 16.000 | 2265.4 | 7.58 |
| Vectorized streaming read, 2048-element tiles | 16.000 | 2270.0 | 7.57 |
| `torch.sum`, one contiguous buffer | 16.000 | 2635.2 | 6.52 |
| `torch.sum`, two contiguous buffers | 16.000 | 2638.6 | 6.51 |
| FP4 K page slab `[pages, 128, 8, 64]`, all heads | 4.000 | 659.5 | 6.51 |
| FP4 V page slab `[pages, 8, 128, 64]`, all heads | 4.000 | 659.6 | 6.51 |
| All four decode tensors at the real byte ratio | 9.000 | 1506.8 | 6.41 |

`torch.sum` saturates at 6.52 TB/s no matter how large the buffer, which is a
property of that reduction kernel rather than of the memory system. A
hand-written load-and-reduce with 128-bit accesses reaches **7.58 TB/s** and is
insensitive to tile size, so that is the figure every ratio below is taken
against. Using the `torch.sum` number instead would have made FA4 look like it
was already at the hardware limit with nothing left to win; FA4 is at 93% of
the real limit, not 100%.

The paged layouts themselves cost nothing in aggregate. Reading the K slab and
the V slab whole both give 6.51 TB/s, the same as a flat contiguous buffer of
the same size under the same reduction, and reading all four decode tensors at
the production byte ratio gives 6.41 TB/s, 98% of the flat case. K's layout
gives one CTA 64 contiguous bytes out of every 512 and V's gives it 8192
contiguous bytes out of every 65536, and in aggregate neither costs anything.

Single-head strided reads in isolation do drop to about 1.75 TB/s against a
5.70 TB/s equal-byte contiguous control, but K's 64-byte runs and V's
8192-byte runs degrade by exactly the same factor, which rules out access
granularity as the cause and points at channel imbalance from the power-of-two
stride. Whether that case arises inside the kernel is answered directly by the
pinning experiment below, which finds the entire DRAM cost of the decode
kernel to be 7 us out of 393; no layout effect can be hiding inside that. The
isolated strided numbers are reported for completeness and no conclusion rests
on them.

## The dominant lever: one Q stage instead of two

`FP4DecodeKernel` sets `q_stage = 2` for the non-split path, which makes
`cta_tiler[0] = 256`. With `pack_gqa` and decode's `seqlen_q = 1`, the packed
M extent is `heads_q / heads_kv` rows, at most 128 by the page-size contract
and 4 for GQA 32:8. The tile count is therefore
`ceil(4 / 256) = 1` with two Q stages and `ceil(4 / 128) = 1` with one, so the
second stage never carries a real Q tile. It loads an out-of-range Q through
TMA, which zero-fills, and then runs a complete softmax, TMEM round trip, P
store and PV MMA over it. `softmax_loop` already skips stage 1 when
`q_stage == 1`, and the split path already sets `q_stage = 1`, so the
machinery exists; the non-split decode path simply does not use it.

Turning it on through the existing `_force_q_stage_1` hook, at
`seqlen 65536, batch 32`:

| Configuration | us | TB/s | vs FA4 | % of 7.58 |
|---|---:|---:|---:|---:|
| `free`, `q_stage=2`, `kv_stage=13` (current) | 451.2 | 5.35 | 2.93x | 70.6% |
| `free`, `q_stage=1`, `kv_stage=14` | 396.4 | 6.09 | 3.33x | 80.3% |
| `free`, `q_stage=1`, `kv_stage=10` | 393.2 | 6.14 | 3.37x | 81.0% |

That is 1.15x on the softmax-free kernel from a single constructor flag. It is
also 1.30x on the production kernel with softmax intact, which is the more
useful fact, and is covered in its own section below.

## KV pipeline depth saturates well before SMEM runs out

Sweeping `kv_stage` at `seqlen 65536, batch 32` with `q_stage = 1`. Each stage
holds either K or V, so two stages are one N-block of work.

| `kv_stage` | SMEM | us | TB/s | % of 7.58 |
|---:|---:|---:|---:|---:|
| 2 | 98 KB | 942.4 | 2.56 | 33.8% |
| 4 | 118 KB | 533.8 | 4.53 | 59.8% |
| 6 | 138 KB | 431.2 | 5.60 | 73.9% |
| 8 | 158 KB | 403.5 | 5.99 | 79.0% |
| **10** | 178 KB | **392.6** | **6.15** | **81.1%** |
| 12 | 198 KB | 397.0 | 6.08 | 80.2% |
| 14 (the default) | 218 KB | 397.3 | 6.08 | 80.2% |
| 16 | 238 KB | assertion: exceeds the 227 KB limit | | |

Depth is worth a great deal up to 8, nothing from 8 to 10, and is very
slightly negative beyond that. The default heuristic, which spends the entire
SMEM budget, lands at 14 and is 1.2% worse than 10 while using 40 KB more
shared memory. SMEM is the binding constraint on how deep the pipeline can go,
but depth stopped being the binding constraint on performance at 8, so the
SMEM budget is not what is holding the kernel back. Deeper pipelining, second
on the candidate list, is a negative result.

The same sweep with `q_stage = 2` shows the two knobs are partly independent:
`kv_stage=6` gives 487.8 us (4.95 TB/s) and `kv_stage=10` gives 436.9 us
(5.53 TB/s), so even the current path would gain 3% by moving from 13 to 10.

## The kernel is not bandwidth-bound any more; it is pipeline-bound

The decisive experiment pins a stream's page index to zero. Every TMA
transaction, every declared byte count and every consumer stays identical, but
the pinned stream's bytes come from L2 instead of DRAM. At
`seqlen 65536, batch 32`, `q_stage=1`, `kv_stage=10`:

| Configuration | us | Time given back |
|---|---:|---:|
| Nothing pinned | 392.6 | — |
| K and its scales pinned | 385.8 | 6.8 us (1.7%) |
| V and its scales pinned | 386.3 | 6.3 us (1.6%) |
| Only the scale factors pinned | 387.1 | 5.5 us (1.4%) |
| K, V and scales all pinned | 385.6 | 7.0 us (1.8%) |

Removing *all* DRAM traffic from the kernel makes it 1.8% faster. There is no
K-versus-V asymmetry to find: pinning either one gives back the whole 7 us,
because either one alone drops the DRAM time far below the other constraint.
The other constraint is the per-N-block cost of the pipeline itself, and it is
385.6 us at this shape.

The same measurement with `q_stage = 2` gives a floor of 428.0 us against an
actual 451.2, so two Q stages both raise the floor by 11% and roughly triple
the exposed DRAM stall.

Normalizing the floor by the number of N-blocks on the critical path gives a
constant:

| Shape | Blocks on the critical path | Pinned floor | ns per block |
|---|---:|---:|---:|
| `65536, batch 32`, `q_stage=1` | 1024 | 385.6 us | 377 |
| `16384, batch 128`, `q_stage=1` | 896 | 339.6 us | 379 |
| `65536, batch 16`, `q_stage=1` | 512 | 196.0 us | 383 |
| `65536, batch 32`, `q_stage=2` | 1024 | 428.0 us | 418 |
| `16384, batch 128`, `q_stage=2` | 896 | 384.4 us | 429 |

A CTA needs about 378 ns per N-block with one Q stage and about 423 ns with
two, independently of shape. An N-block is 18432 bytes of K, V and scales, so
absorbing one every 378 ns is 48.8 GB/s per SM and 7.22 TB/s across 148 SMs.
That is the wall the kernel is actually running into, and it sits 5% below the
7.58 TB/s HBM ceiling.

## Everything together: the whole run is one max()

Combining the ceiling and the floor predicts every measurement. With
`makespan = ceil(rows * heads_kv * splits / 148) * (pages_per_row / splits)`,

```
time ≈ max(bytes / 7.58 TB/s, makespan * 378 ns)
```

| Shape | DRAM term | Pipeline term | Predicted | Measured | Error |
|---|---:|---:|---:|---:|---:|
| `65536, batch 32` | 319 us | 387 us | 387 | 393.2 | +1.6% |
| `65536, batch 37` | 369 us | 387 us | 387 | 404.9 | +4.6% |
| `16384, batch 128` | 319 us | 339 us | 339 | 356.8 | +5.3% |
| `65536, batch 16` | 159 us | 194 us | 194 | 197.8 | +2.2% |
| `16384, batch 32` | 80 us | 97 us | 97 | 106.9 | +10.4% |
| `16384, batch 16` | 40 us | 48 us | 48 | 55.8 | +15.3% |

The residual is the imperfect overlap of the two terms plus a fixed prologue
that only matters when there are few blocks per tile.

The practical consequence is that "achieved TB/s" for this kernel is mostly a
statement about how well the shape tiles 148 SMs, not about the memory system.
At `batch 32` there are 256 work tiles over two rounds of 148 slots, so 13.5%
of the machine's slot-time goes unused. At `batch 37` there are 296 tiles,
exactly two full rounds, and none does.

| Shape | Work tiles | SM occupancy | `free_q1_kv10` TB/s | FA4 TB/s | ratio |
|---|---:|---:|---:|---:|---:|
| `65536, batch 16` | 128 | 86.5% | 6.11 | 6.79 | 0.90 |
| `65536, batch 32` | 256 | 86.5% | 6.14 | 6.49 | 0.95 |
| `65536, batch 37` | 296 | 100.0% | **6.90** | 7.04 | **0.98** |
| `16384, batch 32` | 256 | 86.5% | 5.65 | 6.39 | 0.88 |
| `16384, batch 128` | 1024 | 98.8% | **6.77** | 6.75 | **1.00** |

FA4 is subject to the same effect, which is why FA4 itself reads 6.49 TB/s at
`batch 32` and 7.04 TB/s at `batch 37`. On the two shapes that tile the machine
evenly, the softmax-free FP4 kernel is at 98% and 100% of FA4's achieved
bandwidth. Of the 17% shortfall this investigation started from, about two
thirds was the wasted Q stage and the remainder splits between a comparison
shape that leaves 13.5% of the machine idle and the last few percent of
imperfect DRAM overlap.

## Split count: what splitting does to the rest of the kernel matters more
than the count

At `seqlen 65536, batch 32` the heuristic already selects one split, and
forcing more is worse:

| Splits | `kv_stage` | us | TB/s |
|---:|---:|---:|---:|
| 1 | 10 | 392.6 | 6.15 |
| 2 | 4 (the split-path cap) | 545.2 | 4.43 |
| 4 | 6 | 426.4 | 5.67 |
| 8 | 6 | 471.6 | 5.12 |

The interesting case is `batch 16`, where the heuristic does split. Splitting
sets `is_split_kv`, which pins `q_stage = 1`, caps `kv_stage` at 4 through
`KV_STAGE_FP4_SPLITK_CAP`, makes the partial output FP32 so the epilogue buffer
quadruples, overlaps sO with sQ, and turns off the persistent scheduler.

| Configuration at `16384, batch 16` | us | TB/s |
|---|---:|---:|
| splits=2, `kv_stage=4` (what production picks) | 82.8 | 3.65 |
| splits=2, `kv_stage=6` | 70.2 | 4.30 |
| splits=2, `kv_stage=8` | 66.1 | 4.57 |
| splits=2, `kv_stage=9` (225 KB, the most that fits) | 65.7 | 4.60 |
| splits=1, `kv_stage=10` | **55.9** | **5.40** |
| splits=4, `kv_stage=4` | 95.6 | 3.16 |
| splits=8, `kv_stage=4` | 109.1 | 2.77 |

At `65536, batch 16` the same ordering holds: splits=2 gives 276.7 us and
splits=1 with `kv_stage=10` gives 197.8 us, 1.40x. So on GQA at 128 work tiles
splitting is a net loss, and the depth cap accounts for about two thirds of it:
of the 26.9 us that splits=1 recovers at `16384, batch 16`, 16.7 us comes back
just from raising `kv_stage` from 4 to 8 while still splitting.

This does not generalize to every head configuration. At MQA 32:1,
`batch 32, seqlen 16384` there are only 32 unsplit work tiles for 148 SMs, and
forcing one split is catastrophic: production's 62.0 us becomes 208.2 us,
3.4x slower. Lifting the depth cap there changes nothing (22.6 us at
`kv_stage=4` against 22.2 us at 8), because that shape is limited by having too
few CTAs, not by pipeline depth. The split heuristic's job at MQA is real; its
side effect on `kv_stage` is what should be revisited.

## What the recommended configuration does to the production kernel

`q_stage = 1` is not a probe-only trick. It changes no arithmetic, only how
many Q tiles a CTA carries, and the second tile carries nothing. Cosine
similarity against FA4 is unchanged to four decimal places at every shape and
head configuration measured:

| Shape | Production us | Recommended us | Speedup | cosine before / after |
|---|---:|---:|---:|---|
| GQA 32:8, `65536, batch 32` | 2104.9 | 1624.4 | **1.296x** | 0.9874 / 0.9874 |
| GQA 32:8, `16384, batch 128` | 1854.2 | 1428.6 | **1.298x** | 0.9874 / 0.9874 |
| GQA 32:8, `16384, batch 32` | 557.1 | 413.3 | **1.348x** | 0.9876 / 0.9876 |
| MHA 8:8, `16384, batch 32` | 559.9 | 412.9 | **1.356x** | 0.9875 / 0.9875 |
| GQA 32:8, `16384, batch 16` | 221.7 | 210.2 | 1.055x | 0.9873 / 0.9873 |
| GQA 32:8, `65536, batch 16` | 827.7 | 815.6 | 1.015x | 0.9874 / 0.9874 |

The recommended column is `q_stage = 1` with `kv_stage = 10` and one split. The
`65536, batch 32` timings come from the run that measured both; the cosine pair
for that shape comes from a separate run that measured `full_control` at 0.9874
and `q_stage = 1` at 0.9874, where `q_stage = 1` alone gave 1638.2 us against
2104.1, or 1.284x.

MQA 32:1 at `16384, batch 32` is left out of the table because production
already runs one Q stage there, having split, so there is nothing for this
change to recover: 62.0 us before and 62.1 us after lifting the depth cap to 8.
The MQA head-configuration run does contain a matched cosine pair, 0.9879 for
production and 0.9879 for the one-Q-stage variant, so the numerics are
untouched at MQA as well.

The gain is large exactly where the non-split path runs with two Q stages, and
near zero where production already splits and therefore already uses one Q
stage. MHA 8:8 and GQA 32:8 gain the same amount at the same shape, 1.36x and
1.35x, which is the expected signature: the waste is in the padded M tile, not
in the query rows.

## Configurations that failed

**`n_block_size = 256` and `n_block_size = 64` do not compile.** Both die
building the V scale-factor TMA descriptor at
`src/nvfp4_decode_kernel/fp4_decode_kernel.py:1150`:

```
MLIRError: Operation creation failed:
error: make_tiled_tma_atom_B: expected top-level shape equivalence between the
SMEM layout and the CTA V-map, but got
'!cute.layout<"((((32,4),1),(16,(4,2))),1,2):((((16,4),0),(0,(1,512))),0,1024)">'
and
'!cute.layout<"(((32,4),(16,4)),1,2):(((1@0@0@0,1@1@0@0),(1@0@0@1,1@1@0@1)),0,1@1@1)">'
```

for `n_block_size = 256`, and the analogous `((16,2))` against `(16,4)`
mismatch for 64. The cause is that for the PV MMA the K axis is the sequence
axis, so the SMEM scale-factor layout built from `mma_tiler_pv` scales with
`n_block_size`, while `mma_tiler_sfb_pv` derives its K tiling from
`mma_inst_tile_k`, which is computed from `head_dim_padded` and does not. The
two therefore disagree for any N tile that is not 128. This is a limitation of
how the descriptor is derived, not a hardware rule, but fixing it means editing
`src/` and was out of scope here. A second obstacle sits behind it: the paged
TMA indexes `mPageTable[batch_idx, n_block]` and `load_KV` states that it
assumes `page_size == n_block_size`, so a 256-wide block would additionally
need two TMA copies from two page IDs into the two halves of one SMEM tile.

Reaching these errors first required relocating the TMEM layout in the probe,
because the base constructor's `assert self.tmem_total <= 512` rejects a
256-wide S region under the two-stage layout. Folding the dead stage-1 S region
into one wider stage puts S at columns [0, 256), O at [256, 384) and the Q
scale factors at [384, 512), which fits. TMEM is not what blocks a wider N
tile.

**`kv_stage = 16` on the non-split path** fails with
`AssertionError: SharedStorage 238KB exceeds 227KB limit`, and `kv_stage = 10`
on the split path fails at 235 KB because the split path's FP32 partial-output
buffer adds about 57 KB of fixed shared memory. The split path tops out at
`kv_stage = 9` (225 KB).

**`epi_stage = 1` with `q_stage = 2` deadlocks the GPU.** It hangs at
`65536, batch 32` (256 work tiles, two per CTA) and at `4096, batch 64` (512
tiles, four per CTA); in both cases the kernel never returns and the process
has to be killed. It completes normally at `4096, batch 8`, where every CTA
gets exactly one tile. Tiles per CTA is the only variable that separates the
hanging cases from the working one, so that is the likely trigger, on three
data points. With `q_stage = 1` the same setting is safe even at four tiles per
CTA (61.4 us against 61.2 us for `epi_stage = 2`) and simply buys nothing,
freeing 32 KB that the pipeline has no use for. Confirming that:
`q_stage=1, epi_stage=1, kv_stage=16` at `65536, batch 32` runs at 404.4 us,
worse than the 393.2 us of `epi_stage=2, kv_stage=10`.

## Recommendation

Set `q_stage = 1` for decode on the non-split path, and set `kv_stage = 10`
rather than letting the SMEM heuristic run to 14.

The Q-stage change is the whole of the gain and is close to free to make: the
code path exists, the split kernel already uses it, and the tile arithmetic
guarantees it is correct for decode. With `seqlen_q = 1` and `pack_gqa`, the
packed M extent is `heads_q / heads_kv`, which the contract bounds at 128
because it requires `128 % qhead_per_kvhead == 0`. One M block of 128 rows
therefore always covers the whole query, and the second Q stage is always
empty. Expected gain on the production kernel is 1.30x at
`65536, batch 32`, 1.35x at `16384, batch 32`, and 1.30x at `16384, batch 128`,
with no change in output. Expected gain on the softmax-free kernel is 1.15x,
taking it from 70.6% to 81.0% of the HBM ceiling at `batch 32` and to 91.0% at
a shape that tiles the SMs evenly.

The `kv_stage` change is worth about 1% and its real value is that it returns
40 KB of shared memory, which is currently spent on pipeline depth that does
nothing.

Two follow-ups worth their own work items, in order:

1. **Decouple `KV_STAGE_FP4_SPLITK_CAP` from the split decision.** On GQA at
   128 work tiles the cap costs 1.25x on its own, and the rest of the split
   path's overhead costs another 1.18x. The right fix is probably to raise the
   cap to 8 and to reconsider whether splitting should be chosen at all when
   `rows * heads_kv` is already within a factor of two of the SM count; at MQA,
   where there are 32 tiles, splitting must stay.
2. **Fix `mma_inst_tile_k` for the PV scale factors so an N tile wider than
   the page can be built.** This is the only remaining lever with real headroom.
   The per-block pipeline cost of 378 ns caps the kernel at about 7.22 TB/s
   across 148 SMs while HBM allows 7.58 TB/s, and per-block cost is what a
   wider N tile amortizes. It is worth at most 5%, so it should be sequenced
   after anything on Track A that reduces the per-block TMEM traffic directly,
   and it also needs the two-pages-per-block TMA work described above.

## What could not be tested

- **Any `n_block_size` other than 128**, for the descriptor reason above. The
  claim that a wider N tile would help is therefore an inference from the
  per-block cost model, not a measurement.
- **`m_block_size` below 128.** The block-scaled FP4 MMA is already known to
  reject it (`expects the M-mode to be 128, but got 64`), so it was not
  retried; that error is carried over from the task framing rather than
  reproduced here.
- **Two-CTA MMA**, which the kernel asserts off (`use_2cta_instrs == False`).
  It would halve the number of CTAs and change the SM-balance arithmetic that
  dominates the achieved-bandwidth numbers here, so it may be worth revisiting,
  but it is a much larger change than a probe can carry.
- **A non-sequential page table.** `bench_decode.build_inputs` fills the page
  table with `arange`, so every measurement here reads pages in physical order.
  A fragmented page table would test the TMA and DRAM behaviour that a real
  serving cache produces, and nothing in this document speaks to it.
- **Whether `q_stage = 1` passes `tests/kernel`.** Enabling it requires editing
  `src/`, which was out of scope. The evidence offered instead is that cosine
  against FA4 is unchanged to four decimal places at seven shape and
  head-configuration combinations, and that the split path already ships with
  `q_stage = 1`.
- **The BF16-residual path.** Every measurement used the pre-quantized FP4-Q,
  pure-FP4-KV entry. `fused_residual_first_block` changes the SMEM budget and
  turns off the persistent scheduler, so the `kv_stage` recommendation should
  be re-checked there before it is applied.

Raw data: `hbm-ceiling.json`, `track-b-qstage-65536x32.json`,
`track-b-kvdepth-65536x32.json`, `track-b-pin-split-65536x32.json`,
`track-b-balance-65536.json`, `track-b-shapes-16384.json`,
`track-b-splits-batch16.json`, `track-b-epi-nblock-65536x32.json`,
`track-b-prod-65536x32.json`, `track-b-heads-mqa.json`,
`track-b-heads-mha.json`, `track-b-splitcap-gqa16.json`,
`track-b-splitcap-mqa.json`.
