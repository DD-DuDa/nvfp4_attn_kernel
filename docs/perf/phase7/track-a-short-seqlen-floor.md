# Track A: the short-seqlen floor is the output store, not the pipeline

Branch `perf/fp4-decode-2x`, commit `2257ab8`, GPU 1 (SM100, 148 SMs, clocks
locked at 1965 MHz and sustaining 1845 MHz under load). All times are
kernel-only, summed from the Torch profiler over 50 iterations after 10 warmup
iterations, with every measurement serialized under `/tmp/nvfp4_gpu1.lock` and
compiled with `CUTE_DSL_CACHE_ENABLED=0`. Head configuration is GQA 32:8 unless
stated otherwise. No file under `src/` was modified.

Scripts: `tests/kernel_profile/probe_short_seqlen_floor.py` and
`tests/kernel_profile/iket_cta_timeline.py`. Data: `track-a-breakdown.json`,
`track-a-variants.json`, `track-a-epilogue.json`, `track-a-epilogue-long.json`,
`track-a-seqsweep.json`, `track-a-equivalence.json`, `track-a-highbatch.json`,
`track-a-split-compose.json`, `track-a-epifast-batch.json`, `track-a-mha.json`,
`track-a-mqa.json`, and the IKET analyses `track-a-iket-prod-b1-s1024.txt`,
`track-a-iket-np-b1-s1024.txt`, `track-a-iket-qs1-b1-s1024.txt`.

## Attribution

At `batch 1, seqlen 1024` the decode launch costs 40.5 us and splits as
follows.

| component | us | share |
|---|---:|---:|
| epilogue: one warp storing O after every other warp has retired | 19.7 | 49% |
| main loop: 8 n_blocks at about 2.0 us each, softmax-bound | 16.0 | 39% |
| launch, prologue, and drain | 4.8 | 12% |
| `split_k_combine` | 0.0 | 0% |

The 28 us that survives with softmax arithmetic removed is the same 19.7 us
epilogue plus the same 4.8 us of launch cost plus a main loop that has shrunk
to roughly 3 us. Nothing about it is a mystery once the epilogue is separated
out: 24.5 of those 28 us are paid before and after the main loop, not inside
it.

The cause is a single code path. `epilogue_s2g`
(`fp4_decode_kernel.py:4362`) takes its non-TMA branch because `use_tma_O` is
false whenever `seqlen_q_static_one` is set (`:580`), which both decode paths
set. That branch runs on the single warp in `epilogue_warp_ids = (13,)`
(`:208`), which has just executed `warpgroup_reg_dealloc(num_regs_other)` with
`num_regs_other = 24` (`:271`, `:2060`). It then materializes the entire
128x128 output tile as one per-thread fragment —
`cute.make_fragment_like(tOsO[None, None, None, 0])` at `:4391` is 512 BF16
values, 256 registers per thread — and hands it to `PackGQA.store_O`
(`_fa4/pack_gqa.py:126`), which walks 64 row-steps and predicates all but the
first two away. A decode with `seqlen_q == 1` under PackGQA can only produce
`qhead_per_kvhead` rows, four of them here, and all four are in Q stage 0, so
stage 1's entire store is dead. Nsight Compute at this shape reports 34208
local-memory load sectors and 33376 local-memory store sectors, which is
2.16 MB of register-spill traffic across eight CTAs that write 8 KB of output.

## What actually runs at seqlen 1024

`split_k_heuristic` returns 1 at every batch here, so the whole 40.5 us is one
kernel and no combine kernel is launched. The guard is arithmetic:
`max_pages_per_row` is 8, `min_pages_per_split` is 8 (`_decode.py:59`), and
`splits * 8 <= 8` admits only `splits == 1`. The hypothesis that an aggressive
split count leaves each CTA a single n_block is exactly backwards at this
shape.

Grids read from the profiler trace rather than inferred:

| batch | fp4 grid | CTAs | n_blocks per CTA | fp4 us | FA4 best us |
|---:|---|---:|---:|---:|---:|
| 1 | `[8, 1, 1]` | 8 | 8 | 40.21 | 10.36 (8 splits) |
| 2 | `[16, 1, 1]` | 16 | 8 | 40.89 | 10.89 (8 splits) |
| 4 | `[32, 1, 1]` | 32 | 8 | 40.94 | 12.38 (4 splits) |
| 8 | `[64, 1, 1]` | 64 | 8 | 41.48 | 14.85 (2 splits) |
| 16 | `[128, 1, 1]` | 128 | 8 | 42.38 | 16.01 (1 split) |

The non-split path is persistent, so the grid is
`min(148, num_block * heads_kv * batch)` with `num_block == 1`, giving
`8 * batch` CTAs. Every batch from 1 to 16 fits in one wave on 148 SMs, which
is why the time is flat. Each CTA runs 8 n_blocks and, because the non-split
path uses `q_stage = 2` (`:144-149`), two 128-row Q stages per n_block, the
second of which is entirely outside the four rows a decode produces.

FA4 on the same 8-CTA grid with `num_splits=1` takes 14.74 us against our
40.21 us, so at matched launch geometry and matched per-CTA n_block count we
are 2.7x slower. The gap is not the grid.

## Where the time goes inside the kernel

IKET, steady-state launch, `batch 1, seqlen 1024`, CTA 0 on SM 142. Wall time
under instrumentation is 42.3 us against 40.5 us clean, so the instrumentation
costs about 4%.

| warp | role | lifetime us |
|---:|---|---:|
| 14 | load | 5.5 |
| 15 | empty | 1.1 |
| 1-3 | softmax stage 0 | 20.1-20.5 |
| 4-7 | softmax stage 1 | 20.6-20.9 |
| 8-11 | correction | 21.1-21.4 |
| 12 | MMA | 21.6 |
| 13 | epilogue | 39.3 |
| 0 | softmax stage 0, also holds TMEM to the end | 39.3 |

Warp 13's own timeline on that CTA is the whole answer. Its first
`epi_wait_corr` runs from 1.70 us to 20.99 us, which is simply the main loop.
It then stores Q stage 0 from 20.99 us to 32.26 us, waits 0.19 us for stage 1's
correction, and stores stage 1 from 32.45 us to 39.07 us. Every other warp has
retired by 21.8 us. So 18.1 us of a 39.5 us CTA lifetime is one warp writing
output with the rest of the machine idle, and the two stores cost 11.3 us and
6.6 us for tiles that contain four useful rows between them.

Inside the main loop the critical path is softmax, which is the already-known
Phase 7 result and not this Track's subject. Per n_block a softmax warp spends
0.90 us in `sm_exp`, 0.65 us in `sm_pquant`, 0.36 us in `sm_wait_s` and 0.11 us
in `sm_rowmax`; the MMA warp spends 67.5% of its life in `mma_wait_p` and the
correction warps spend 81% in `corr_wait_sm`. Both consumers of softmax are
starved by it.

## Confirming the cause by removing it

`EpilogueProbeKernel` in `probe_short_seqlen_floor.py` reimplements
`epilogue_s2g` with two independent knobs and is patched into `_decode` at
runtime. `per_m_fragment` stages the shared-to-register copy one row-step at a
time instead of materializing the whole tile. `row_limit` stops the row loop
past the last reachable row and skips Q stages that begin beyond it. Both only
remove stores that the existing predicate already discards.

At `batch 1, seqlen 1024`, against FA4's 10.28 us:

| variant | us | local ld+st sectors | cosine vs FA4 |
|---|---:|---:|---:|
| production | 40.52 | 34208 + 33376 | 0.9877 |
| `epi_control`, both knobs off | 40.48 | not measured | 0.9877 |
| `epi_perm`, per-row-step fragment only | 23.18 | 640 + 640 | 0.9877 |
| `epi_rowlimit`, bounded rows only | 20.81 | not measured | 0.9877 |
| `epi_fast`, both | 20.84 | 512 + 512 | 0.9877 |
| `epi_fast` plus `q_stage = 1` | 17.29 | not measured | 0.9877 |

The control reproduces production to 0.1% here, and to within 1.4% on the split
path at seqlen 4096 and 16384, so the reimplementation is faithful and the
deltas are attributable. Spill traffic falls by 53x. Output is
**bitwise identical** to production, with `max_abs_diff` exactly zero, at
`b1 s1024`, `b4 s1024`, `b1 s4096` and `b2 s16384` for all three fixed
variants, which is the strongest available evidence that the removed work was
dead.

The fix is not specific to GQA 32:8 or to short sequences:

| shape | production us | `epi_fast` us |
|---|---:|---:|
| gqa4 b1 s1024 | 40.52 | 20.84 |
| gqa4 b16 s1024 | 42.42 | 21.68 |
| gqa4 b32 s1024 | 65.89 | 39.43 |
| gqa4 b128 s1024 | 199.82 | 128.17 |
| mha 8:8 b1 s1024 | 39.28 | 20.88 |
| mqa 32:1 b1 s1024 | 39.82 | 21.04 |
| gqa4 b1 s4096 (4 splits) | 22.93 | 19.79 |
| gqa4 b1 s16384 (16 splits) | 24.68 | 21.49 |
| gqa4 b8 s16384 (2 splits) | 112.98 | 109.58 |

On the split path the same fix is worth a flat 3.1 to 3.3 us per decode kernel
(20.32 to 17.18 us at `b1 s4096`, 20.97 to 17.77 us at `b1 s16384`, 110.33 to
107.01 us at `b8 s16384`). That path is cheaper to begin with because it uses
`q_stage = 1`, so it stores once instead of twice, and because
`SingleTileScheduler` compiles the store as straight-line code rather than a
loop body.

## Per-launch cost or per-CTA work

Both, in a specific proportion. Sweeping seqlen at batch 1 separates the
intercept from the slope; seqlen 128 through 1024 all take the non-split path
with one CTA per KV head, while 2048 and 4096 cross into split-K.

| seqlen | n_blocks per CTA | splits | FA4 us | production us | `epi_fast` us |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 1 | 6.33 | 26.53 | 6.82 |
| 256 | 2 | 1 | 7.55 | 28.95 | 8.77 |
| 512 | 4 | 1 | 9.93 | 32.60 | 12.77 |
| 1024 | 8 | 1 | 10.25 | 40.94 | 20.85 |
| 2048 | 8 | 2 | 11.54 | 22.59 | 19.46 |
| 4096 | 8 | 4 | 13.91 | 22.76 | 19.62 |

Production's non-split line is 24.5 us of intercept plus 2.06 us per n_block.
The fixed part does not move with seqlen: at seqlen 128, where each CTA has a
single n_block and the main loop is one eighth as long, production still costs
26.5 us. With the epilogue fixed the intercept falls to 4.8 us and the slope is
unchanged at 2.0 us per n_block, so about 20 of the 24.5 us of fixed cost is
the epilogue and the genuine launch, prologue, TMEM-allocation and drain cost
is under 5 us. FA4's own floor is no better: at seqlen 128, where a CTA has one
n_block of work, FA4 costs 6.33 us against `epi_fast`'s 6.82 us.

The intercept is paid per work tile, not once per launch. At `batch 32`,
where 256 tiles land on 148 CTAs so most CTAs run a second tile, production
goes from 40.5 to 65.9 us, a marginal 25.4 us, and `epi_fast` goes from 20.9 to
39.4 us, a marginal 18.6 us. Nothing about the persistent scheduler lets the
epilogue of one tile overlap the main loop of the next: with `q_stage = 2` and
`epi_stage = 2` both output buffers belong to the current tile, so the next
tile's correction warps block on `mbar_corr_epi_empty` until the store drains.

## Hypotheses that died

**`split_k_combine` is part of the 41 us.** No. At seqlen 1024 the heuristic
returns 1 for every batch and no combine kernel is launched; the profiler shows
exactly one CUDA kernel in the FP4 call. The combine kernel costs 2.3 to 3.9 us
when it does run.

**The split heuristic picks too many splits at short seqlen.** The opposite. It
picks one, because `min_pages_per_split = 8` cannot be satisfied with 8 pages
per row.

**TMEM allocation, mbarrier init, or scheduler startup dominates.** No. With
the epilogue fixed, the whole intercept at one n_block is 6.82 us including the
main loop's 2.0 us, so everything from launch through TMEM allocation, mbarrier
init, Q load, pipeline fill and drain is under 5 us, against FA4's 6.33 us at
the same shape.

**It is per-CTA main-loop work that simply does not shrink.** Only 16 of the
40.5 us. The main loop is real, it is softmax-bound, and at 2.0 us per n_block
it is about 1.7x FA4's per-n_block cost, but it is not the floor.

**The persistent tile scheduler carries an independent cost.** It looked like
it: forcing `SingleTileScheduler` on the non-split path takes 40.52 us to
23.89 us at `b1 s1024` with the identical grid of 8 CTAs. IKET shows why, and
it is not a scheduler cost. The non-persistent variant's epilogue tail is
3.9 us against the persistent variant's 18.1 us; the store is the same source
code, but compiled as a straight-line single iteration instead of a dynamic
`while` body the compiler keeps far more of the fragment in registers. The
scheduler is a lever on the epilogue bug, not a separate defect.

**`q_stage = 2` is a 2x waste.** Partly. It looked like exactly 2x —
`_force_q_stage_1` on the non-split path takes 40.52 us to 20.25 us — but most
of that is the second epilogue store, which is entirely dead work. With the
epilogue already fixed, `q_stage = 1` is worth 20.84 us to 17.29 us at
`b1 s1024` and 39.43 us to 31.86 us at `b32 s1024`. Real, but a third the size
it first appeared.

## Relationship to the parallel split-guard finding

`split-guard-miscalibration.md`, produced from Track C, reads the same 28 us as
occupancy: 8 CTAs on 148 SMs, and forcing `splits=8` at `b1 s1024` brings the
kernel to 10.4 us of graph-replay time. That measurement is right and my own
forced-split runs agree with it (12.20 us kernel-only for `prod_split8` at
`b1 s1024`, of which 9.26 us is decode and 2.94 us is combine). The
interpretation that the floor is mostly serialized per-CTA work is not
supported by the seqlen sweep: at seqlen 128 the occupancy is the same 8 CTAs
and the per-CTA main loop is one eighth as long, yet production still costs
26.5 us while the same shape with a fixed epilogue costs 6.8 us. Splitting
helps at `b1 s1024` partly because it shortens the main loop and partly because
it moves the work onto a code path whose epilogue does not spill.

The two fixes compose, and after the epilogue fix the case for relaxing the
split guard is weaker but not gone:

| variant at s1024 | batch 1 us | batch 16 us |
|---|---:|---:|
| production | 40.52 | 42.42 |
| `epi_fast` | 20.84 | 21.68 |
| `prod_split8` | 12.20 | 61.69 |
| `epi_fast` + 8 splits | 9.05 | 39.57 |
| `epi_fast` + 4 splits | 10.16 | 29.95 |
| `epi_fast` + 2 splits | 13.15 | 23.53 |
| FA4 best | 10.28 | 16.12 |

At batch 1 the two together reach 0.88x FA4. At batch 16 every split count is
worse than not splitting once the epilogue is fixed, so a relaxed guard has to
be batch-aware rather than a single constant.

## Candidate fixes, ordered by gain over cost

**1. Bound the epilogue store to the rows a decode can produce.** Touches the
non-TMA branch of `epilogue_s2g` (`fp4_decode_kernel.py:4362-4428`) and
`PackGQA.store_O` (`_fa4/pack_gqa.py:126-166`), which needs an optional
constexpr row bound. Under `seqlen_q_static_one` with PackGQA the reachable row
count is the compile-time constant `qhead_per_kvhead`, so the row loop bound
and the dead-stage skip are both constexpr and nothing becomes dynamic. Gain:
1.94x at `b1 s1024`, 1.96x at `b16 s1024`, 1.67x at `b32`, 1.56x at `b128`,
and a flat 3.1 to 3.3 us on every split-path launch including seqlen 16384 and
65536. Output is bitwise identical, so the numerical gates cannot move. Cost:
roughly forty lines, all under a flag that already exists.

**2. Stage the shared-to-register copy one row-step at a time.** Same call
site, `fp4_decode_kernel.py:4391-4392`. This is the shape-agnostic half of fix
1: it does not assume anything about how many rows are valid, and on its own it
takes `b1 s1024` from 40.52 to 23.18 us and cuts spill traffic by 53x. Worth
listing separately because it is safe for any `seqlen_q` and could land first
if the row bound needs more validation. Combining both is only 2.3 us better
than the row bound alone at this shape, so if only one lands, prefer the row
bound for decode and this one for generality.

**3. Give the epilogue warp registers, or give the store more warps.**
`num_regs_other = 24` (`fp4_decode_kernel.py:271`) and
`epilogue_warp_ids = (13,)` (`:208`) are what make a 32 KB tile spill. Raising
the epilogue's register budget, or routing the store through the four
correction warps as `use_correction_warps_for_epi` already does for varlen
(`:229-231`), attacks the same defect from the resource side. Not measured;
listed because it is the fix that would also help a future non-decode caller,
and because it interacts with softmax's 216 registers so it needs its own
occupancy check.

**4. Use `q_stage = 1` on the non-split decode path.** One condition at
`fp4_decode_kernel.py:144-149`. After fix 1 it is worth 20.84 to 17.29 us at
`b1 s1024` and 39.43 to 31.86 us at `b32 s1024`, which crosses FA4 at batch 32.
It also leaves softmax warps 4 to 7 idle, which `phase5b/final-report.md`
already flagged as wasted capacity, so it should be decided together with any
warp-repartitioning work rather than on its own. It is not numerically neutral:
at `b128 s1024` the cosine against FA4 moved from 0.9875 to 0.9868, small but
real, so it needs `tests/kernel` evidence in its own right.

**5. Make the split guard batch-aware.** `_decode.py:49-63`. Track C's note
covers this; the numbers above say the guard is worth relaxing at batch 1 to 4
and worth keeping at batch 16 and up, so the change is a rule, not a constant.
Do it after fix 1, since fix 1 removes most of what makes `splits=1` look bad.

## What I could not determine

I did not read SASS, so I cannot say precisely why the persistent scheduler's
dynamic loop body spills far more of the output fragment than the equivalent
straight-line code. The measurement is unambiguous but the compiler mechanism
is inferred.

The epilogue probe asserts pure FP4 with no residual, no `out_indices`, and
PackGQA. The BF16-residual path and the direct-scatter path use the same
`epilogue_s2g` and should benefit identically, but I did not measure them, and
`use_out_indices` changes how `mO_cur` is formed so it needs its own check.

I did not measure fix 3, and I did not check whether raising the epilogue
warp's register budget would cost occupancy elsewhere.

The small cosine change under `q_stage = 1` at batch 128 was seen once and not
investigated; it could be a genuine difference in the SFQ TMEM path taken at
`q_stage == 1` (`fp4_decode_kernel.py:3198`) or measurement variation across
recompiles.

## Reproducing

```bash
export ENV=/apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/BitKV_nvfp4/_local/envs/vllm-nvfp4
export CUDA_VISIBLE_DEVICES=1
cd /apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/nvfp4_attn_kernel

flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
  PYTHONPATH=src:tests/kernel_profile $ENV/bin/python \
  tests/kernel_profile/probe_short_seqlen_floor.py --mode breakdown \
  --grid 1x1024,2x1024,4x1024,8x1024,16x1024,1x4096,1x16384"

flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
  PYTHONPATH=src:tests/kernel_profile $ENV/bin/python \
  tests/kernel_profile/probe_short_seqlen_floor.py --mode variants \
  --grid 1x1024 --variants epi_control,epi_perm,epi_rowlimit,epi_fast"

flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
  PYTHONPATH=src:tests/kernel_profile $ENV/bin/python \
  tests/kernel_profile/probe_short_seqlen_floor.py --mode equivalence \
  --grid 1x1024,4x1024,1x4096,2x16384 --variants epi_fast"

flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
  PYTHONPATH=src:tests/kernel_profile $ENV/bin/run-iket \
  --output-dir /tmp/iket_prod_1x1024 --clobber profile --postprocess all -- \
  $ENV/bin/python tests/kernel_profile/probe_short_seqlen_floor.py \
  --mode iket --grid 1x1024 --variants prod --iters 2"
$ENV/bin/python tests/kernel_profile/iket_cta_timeline.py \
  /tmp/iket_prod_1x1024/*.trace.json --launch 1 --cta 0 --warps 8,13
```

The `iket` mode clears both compile caches before launching, which the IKET
workflow requires; a cached kernel produces a silently empty trace.
