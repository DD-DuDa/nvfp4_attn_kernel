# Decode nondeterminism: a missing async-proxy fence on the P scale factors

Branch `debug/decode-nondeterminism`, from commit `1ea4715`. All work on GPU 2
(SM100, 148 SMs), clocks not locked, every compile under
`CUTE_DSL_CACHE_ENABLED=0`. Probes:
`tests/kernel_profile/check_epilogue_equivalence.py --determinism` and
`tests/kernel_profile/nondet_structure.py`.

## Root cause

The softmax warps publish the FP4 P-tile scale factors to shared memory with
ordinary register-to-shared stores, and the MMA warp reads them back with
`tcgen05.cp`, which is an asynchronous-proxy access. Nothing in between makes
the stores visible to the async proxy, so the MMA warp can copy a stale
`sSFP` stage into TMEM and multiply a whole 16-element group of P by the
previous n_block's scale factor.

The producer is `softmax_step` in `src/nvfp4_decode_kernel/fp4_decode_kernel.py`:

- `:3642` `cute.autovec_copy(tSrPSF_2d, sSFP_thread)` writes the E4M3 scale
  bytes for this thread's row into `sSFP[..., stage]`. These are generic-proxy
  STS instructions.
- `:3665` `mbarrier_arrive(mbar_P_full_O_rescaled + stage)` publishes the tile.

The consumer is `mma` in the same file:

- `:2938`–`:2946` and `:3051`–`:3059`, after waiting on
  `mbar_P_full_O_rescaled + stage`, run
  `cute.copy(tiled_copy_s2t_sfp, tCsSFP_compact_s2t_cur, tCtSFP_compact_s2t)`.
  `tiled_copy_s2t_sfp` is built from `tcgen05.Cp4x32x128bOp`
  (`mainloop_s2t_copy_and_partition`, `:4762`), i.e. a `tcgen05.cp` SMEM→TMEM
  copy issued through the async proxy.

An mbarrier arrive/wait pair orders generic-proxy traffic. It says nothing
about what the async proxy will observe. PTX requires
`fence.proxy.async.shared::cta` between a generic-proxy write to shared memory
and an async-proxy read of it. `sSFP` is the only mainloop buffer that is
written generically and read asynchronously, and it is the only one that was
missing the fence. The two comparable paths in this kernel both have it:
`quantize_Q_bf16_to_fp4` fences `sQ`/`sSFQ` before signalling the MMA warp
(`:4668`, with the comment "Memory fence so MMA warp's sQ/sSFQ reads see our
writes"), and `correction_epilogue` fences `sO` before the TMA store (`:4215`).
K, V, SFK and SFV are all TMA-written, so their mbarrier completion already
carries async-proxy visibility.

Because `sSFP` has only `q_stage = 2` slots and is rewritten on every n_block,
the value a stale read returns is the previous n_block's scale factor for the
same (row, K-group). That is a plausible-looking number, which is why the
symptom is a small perturbation rather than a NaN.

## The fix

One line, at `:3648`:

```python
cute.autovec_copy(tSrPSF_2d, sSFP_thread)
cute.arch.fence_view_async_shared()   # fence.proxy.async.shared::cta
```

`cute.arch.fence_view_async_shared()` expands to `fence_proxy(kind="async.shared",
space="cta")`. Every softmax thread fences its own stores before its own
arrive on `mbar_P_full_O_rescaled`, and that barrier requires all 128 arrivals,
so the whole tile is visible before the MMA warp's S2T copy runs.

## Evidence

### The symptom is a scale-factor substitution, not a lost or garbage buffer

At `32x8x18x4`, twelve repeats of the `residual` call shape move 101 to 106 of
the 576 (row, head) pairs. Within a moving pair, roughly 115 of 128 output
channels change, each by about 1e-3 on outputs of order 0.1 to 1. The change
is not a common factor across the channels, so it is not a perturbed softmax
denominator; and it is far too small and too broad to be a dropped or
garbage K/V tile. It is what one wrong 16-element group of P looks like after
it has been multiplied into V and spread across all 128 output channels.

### Ruling out the KV ring buffer

`kv_stage` is 13 for pure FP4 and 4 for the residual configuration; the
residual's extra `sK_bf16`/`sV_bf16`/`sQ_bf16` tiles cost it nine stages. The
coincidence between `kv_stage = 4` and the four-page trigger threshold is
exactly that. Two measurements break it:

| configuration | result |
|---|---|
| pure FP4, `kv_stage` 13, `32x8x64x4` (8 KV loads per tile, no wrap) | nondeterministic |
| pure FP4, `kv_stage` clamped to 4, `32x8x18x8` (16 KV loads, four wraps) | stable, 16 repeats |

Clamping `kv_stage` moves the wrap point by more than a factor of three and
does not move the trigger at all, and the trigger fires in a configuration
that never wraps. The KV pipeline is not involved.

### The trigger is desynchronisation, not a specific count

Every "stable" point below survived 48 repeats, so these are structural, not
lucky.

| shape | CTAs | tiles/CTA | pages | result |
|---|---:|---:|---:|---|
| `32x8x18x1..3` | 144 | 1 | 1–3 | stable |
| `32x8x2x4`, `32x8x4x4`, `32x8x8x4` | 16/32/64 | 1 | 4 | stable |
| `32x8x18x4..10` | 144 | 1 | 4–10 | residual shapes nondeterministic |
| `32x8x64x2` | 512 | 1 or 3–4 | 2 | stable |
| `32x8x64x4..8` | 512 | 1 or 3–4 | 4–8 | nondeterministic, pure FP4 included |

The pure-FP4 and residual exposures look like two different mechanisms because
`is_persistent = not has_residual` (`_decode.py:233`): pure FP4 runs the
`StaticPersistentTileScheduler` with several work tiles per CTA, while the
residual path runs one tile per CTA. Both are the same mechanism seen through
different sources of skew.

The cleanest demonstration is the residual length. Holding the shape at
`32x8x18x4` and forcing every row to the same `seqused_residual` makes the
kernel deterministic; only a batch whose rows differ reproduces the bug.

| `seqused_residual` | result over 12 repeats |
|---|---|
| the probe's default, 18 distinct values in [0, 122] | 101/576 pairs move |
| all 0 | bitwise identical |
| all 1 | bitwise identical |
| all 128 | bitwise identical |

Uniform lengths do not change how much work a CTA does — the mask is applied in
registers — but they do keep every CTA on the same schedule. Once the CTAs
drift apart, memory latency varies enough per SM that the MMA warp sometimes
arrives at the S2T copy inside the window where the softmax warp's stores are
not yet visible to the async proxy. The four-page floor is the same story from
the other side: the main loop needs three or four iterations to reach the
steady state where the MMA warp is running immediately behind the softmax
warps.

### The fix is a fence, not a timing perturbation

Adding any instruction to the softmax critical path perturbs timing, so the
placement was tested both ways. The identical instruction placed *before* the
`sSFP` store instead of after it — same instruction count, same cost, no
ordering effect on the stores — leaves the kernel broken.

| variant | `32x8x18x4` | `32x8x18x6` | `32x8x64x4` | `32x8x64x8` |
|---|---|---|---|---|
| no fence (`1ea4715`) | nondeterministic | nondeterministic | nondeterministic | nondeterministic |
| fence before the store | nondeterministic | nondeterministic | nondeterministic | nondeterministic |
| fence after the store | stable | stable | stable | stable |

### Determinism after the fix

Every shape below is bitwise identical across all four public call shapes
(`pure_fp4`, `residual`, `indexed_rows`, `direct_scatter`).

| shapes | repeats | result |
|---|---:|---|
| `32x8x18x4`, `18x5`, `18x7`, `18x10`, `64x5`, `64x8`, `128x8`, `8x8x64x8`, `32x1x64x8` | 24 | stable |
| `32x8x1x64`, `5x64`, `16x128`, `64x128` (the last two split K) | 16 | stable |
| `32x8x18x4`, `18x6`, `64x4`, `64x8` | 16 | stable |

`32x8x16x128` and `32x8x64x128` are the split-K shapes previously reported as
nondeterministic; both are now stable.

### What failed to reproduce

`compute-sanitizer --tool racecheck --racecheck-report all` reports **0 hazards**
on the failing shape `32x8x18x4` with the bug present, and the output is
bitwise stable across 6 repeats under the sanitizer. Racecheck is not a useful
tool for this class of bug. It models generic shared-memory accesses against
mbarrier synchronisation, and in that model the `sSFP` handoff is correctly
synchronised; the hazard exists only in the proxy the tool does not model.
The sanitizer's serialisation also removes the warp skew the bug needs, so it
masks the symptom as well as missing the cause.

## Cost

Interleaved A/B in one process: both variants compiled once, their
`_decode_compile_cache` entries stashed, and the timing loop alternating
between them so clock drift lands on both. Median of 5 event-timed blocks of
200 iterations for end to end, median of 3 Torch-profiler passes of 100
iterations for the decode kernel alone. **Clocks were not locked, so treat
these as indicative.**

| shape | call | kernel, fence | kernel, no fence | delta |
|---|---|---:|---:|---:|
| `32x8x18x4` | residual | 49.82 us | 49.83 us | -0.03% |
| `32x8x64x8` | residual | 300.09 us | 302.97 us | -0.95% |
| `32x8x64x8` | pure FP4 | 75.32 us | 74.97 us | +0.46% |
| `32x8x128x32` | residual | 1214.24 us | 1228.62 us | -1.17% |
| `32x8x128x32` | pure FP4 | 467.68 us | 466.94 us | +0.16% |

The deltas are within ±1.2% and change sign, which is the run-to-run spread of
this measurement rather than a real effect. The fix is free at the resolution
available here. That is what one expects: `fence.proxy.async.shared::cta` is
issued once per softmax step per thread and does not stall the warp.

## Tests

`PYTHONPATH=src python -m pytest -q tests/kernel` passes: 58 tests, up from
the 57 at `1ea4715`.

The added test is
`tests/kernel/test_fp4_decode_correctness.py::test_repeated_identical_calls_are_bitwise_identical`.
It runs `fp4_decode` eight times on identical inputs and requires
`torch.equal` against the first result. The shape is chosen to sit just past
every threshold established above, and each factor is necessary:

- 4 pages per row, so the main loop reaches steady state; 3 pages is stable
  over 48 repeats.
- 18 rows x 8 KV heads = 144 CTAs against 148 SMs, so nearly the whole machine
  is resident; 64 CTAs is stable over 48 repeats.
- Per-row residual lengths that differ, so the CTAs do not stay in lockstep; a
  uniform batch is stable.

Verified to fail 3 out of 3 runs with the fence commented out and to pass with
it in place. It costs about 16 seconds, most of which is one compile.

The existing suite could not have caught this. Its largest decode case is
`AttentionCase(1, 64, (8192,), 32, 8)` — 8 CTAs — and its largest batch is
7 rows, so no case comes close to filling the machine.

## Not determined

- Why the CTA-count floor sits between 64 and 144 CTAs rather than somewhere
  else. The bug needs enough resident CTAs to make per-SM memory latency vary,
  but no measurement here pins the mechanism more precisely than that, and the
  threshold is likely specific to this GPU's SM count and memory system.
- Whether any residual-length pattern other than a uniform batch is safe. Only
  uniform-0, uniform-1, uniform-128 and the probe's 18-value spread were
  measured.
- Two unrelated issues were noticed while reading the residual path and are
  **not** part of this bug — neither is a source of nondeterminism, and neither
  was changed:
  - The three BF16 residual barriers are waited on with a literal phase
    `Int32(0)`: the MMA warp on `mbar_bf16_P_full` and `mbar_bf16_P_full_2`
    (`:2813`, `:2817`) and the softmax warps on `mbar_bf16_S_full` (`:3720`).
    None of them uses a flipping phase variable, unlike the neighbouring
    `residual_kv_full_phase` a few lines away. On a second
    work tile in the same CTA those waits would return immediately. It is
    currently unreachable because the residual path sets `is_persistent=False`,
    so every CTA handles exactly one tile.
  - With the residual, softmax stage 0's first FP4 `softmax_step` runs with
    `is_first=False` and writes an `acc_scale` into `sScale`, but the
    correction warps' pre-loop consumes that `mbar_softmax_corr_full` arrival
    without applying the rescale (`:3862`–`:3867`, "Ignore first signal from
    softmax as no correction is required"). The O accumulator therefore misses
    the rescale for the BF16-to-first-FP4-block transition. This is
    deterministic, so it is an accuracy question, not this one.
