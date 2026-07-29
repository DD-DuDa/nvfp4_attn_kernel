# Where to Put Markers

## Procedure

1. Find the `@cute.kernel` function. Host `@cute.jit` wrappers are context only.
2. Split the kernel into its natural phases. For GEMM-shaped code: setup, copy/TMA issue,
   mainloop, MMA, waits, epilogue. Use names from the kernel's own algorithm otherwise.
3. Note every warp-specialized region (`if warp_idx == ...`, `if is_leader_cta:`). Both
   ends of a role-specific range must live inside the same guard.
4. Identify asynchronous work. TMA copies, `cp.async`, MMA issue, and mbarrier operations
   have distinct issue and completion points; those two points are what you want to see.

## Start coarse

One lifetime range plus three or four phase ranges per role is enough to orient. Add
detail only where the coarse trace already shows a problem. Every extra unique name costs
buffer and can perturb the compiler.

```python
@cute.kernel
def kernel(...):
    life = iket.range_start("warp_life")

    iket.range_push("prologue")
    # partitioning, pipeline setup, scheduler setup
    iket.range_pop()

    iket.range_push("mainloop")
    for k_tile in cutlass.range(k_tile_count):
        ...
    iket.range_pop()

    iket.range_push("epilogue")
    iket.range_pop()

    iket.range_end(life)
```

## Warp-specialized layering

Place each range inside the guard for the warp that does the work:

```python
if warp_idx == load_warp_id:
    iket.range_push("tma_main")
    ...
    iket.range_pop()

if warp_idx == mma_warp_id:
    iket.range_push("mma_main")
    ...
    iket.range_pop()
```

## The wait pattern is the point

For diagnosing a pipeline, the single most valuable measurement is **time spent waiting**.
Wrap the wait, not just the work:

```python
iket.range_push("ab_wait")
ab_full = ab_consumer.wait_and_advance()
iket.range_pop()
```

A consumer role that spends most of its life in a wait is producer-starved. A producer
role that spends most of its life waiting for empty buffers means consumers are the
bottleneck. This distinction is the main thing IKET buys over aggregate counters, and it
is invisible to ncu.

Apply it to pipeline acquires, mbarrier waits, allocator waits — any synchronization
point whose return marks completion of the awaited work.

## This repository's kernel

`src/nvfp4_decode_kernel/fp4_decode_kernel.py` declares a 16-warp specialization around
line 201:

| Warps | Role attribute | Suggested range prefix |
|---|---|---|
| 0–3 | `softmax0_warp_ids` | `sm0_` |
| 4–7 | `softmax1_warp_ids` | `sm1_` |
| 8–11 | `correction_warp_ids` | `corr_` |
| 12 | `mma_warp_id` | `mma_` |
| 13 | `epilogue_warp_ids` | `epi_` |
| 14 | `load_warp_ids` | `load_` |
| 15 | `empty_warp_ids` | none |

These assignments are not static across configurations. When `use_correction_warps_for_epi`
is set, epilogue moves onto the correction warps and warp 15 joins the load role. Read
the constructor rather than assuming, and record the effective mapping alongside the
trace so `analyze_trace.py --roles` can label it correctly.

Warp 15 is idle in the default configuration. A trace that shows it consuming lifetime is
itself a finding.

## Name budget

At most 32 characters per name. Keep the total number of unique names under about 30 —
more than that increases overhead, risks buffer overflow, and can change compiler
behavior. Reusing one name across loop iterations is the intended way to record a
recurring phase and costs nothing extra in unique-name budget.

Timer granularity is 32 ns. A range shorter than that will not be distinguishable, so do
not instrument individual instructions.
