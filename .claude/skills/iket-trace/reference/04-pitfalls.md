# Failure Modes

## Empty trace

**Cause 1 — the kernel did not JIT-compile inside the profiled run.** `run-iket` requests
IKET lowering for kernels compiled during the profiled process. Anything already compiled
and reused is untouched. Any compilation cache in the project must be cleared or bypassed
before profiling. This is the most common cause and it fails silently.

**Cause 2 — instrumentation is in host code.** `iket.*` calls in `@cute.jit` wrappers or
plain Python emit nothing. They must be inside `@cute.kernel`.

**Cause 3 — the workload was not launched under `run-iket`.** Without the profiler, and
without `CUTE_DSL_COMPILER_OPT=iket` or `options="iket"`, the calls are stripped and cost
nothing.

Raise `--log-level debug` to see which kernels were instrumented.

## Corrupted or nonsensical timeline

Divergent instrumentation. Unbalanced or divergent `range_push` / `range_pop` may fail
profiling, produce a wrong visualization, or yield undefined results. Placement must be
warp-uniform and both endpoints must be inside the same warp guard.

## Buffer overflow

`run-iket` sizes device buffers from a dry pass and assumes the per-warp record count is
reasonably stable between passes. High-frequency instrumentation, data-dependent event
counts, or many launches can overflow. Reduce event frequency, or raise
`--context-buffer-size`.

## Cannot run alongside other profilers

`run-iket` conflicts with Nsight Compute, Nsight Systems, and other CUPTI-based tools over
driver profiling resources. Collect separately.

## Missing GPC/TPC attribution

In some containers the profiler cannot read SM topology and logs:

```
VsmTopologyMapper: RmCtrlGetVsmMappings (count query) failed for device 0
```

`gpcId` and `tpcId` then come back as `-1`. This is not fatal: `smId`, `ctaId`, and
`warpId` remain valid, which is what warp-role analysis needs. Only GPC/TPC-level
grouping is lost.

An accompanying `CUDA_CALL error = 0001 "invalid argument"` from the injection library is
also non-fatal and does not prevent trace generation.

## Distorted measurements

IKET adds fixed per-kernel entry and exit overhead, plus host-side overhead. It may also
change compiler behavior — instrumentation can inhibit interleaving optimizations after
unrolling. Consequences:

- Never use an instrumented run for performance gates. Timing numbers come from clean
  runs only.
- The trace-reported kernel duration includes some IKET overhead that is not shown as a
  separate event.
- Application wall clock is not a valid measure of IKET overhead. To assess overhead,
  compare kernel durations inside the trace.
- Payloads increase stored data; add them only where the value is needed.

## Names silently rejected

Names must be at most 32 characters. Longer names are not supported.

## Ranges that vanish

Timer granularity is 32 ns. A range whose start and end fall within one tick may show
zero or near-zero duration. Instrument phases, not instructions.
