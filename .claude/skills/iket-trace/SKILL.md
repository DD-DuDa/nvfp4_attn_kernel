---
name: iket-trace
description: Collect and analyze IKET (In-Kernel Event Tracing) traces for CuTe DSL kernels on SM90+. Use when asking where time goes inside a warp-specialized or persistent kernel, which warp role is the critical path, whether the software pipeline overlaps, or why per-CTA throughput is low — including Chinese variants ("kernel 内部时间去哪了", "哪个 warp 在等", "流水线有没有重叠", "iket 分析").
---

# Skill: IKET In-Kernel Event Tracing (CuTe DSL)

**When to use:** the question is *where time goes inside one kernel launch* — which warp
role is on the critical path, how long each pipeline phase takes, whether producer and
consumer warps overlap, how work is distributed across CTAs and SMs.

**When NOT to use:** the question is *how much* of something — bytes moved, cache hit
rate, occupancy limiter, register pressure. IKET has no counters. Use `ncu-report-skill`
for those. See "Tool selection" below.

**Requires:** SM90 or newer, and a `nvidia-cutlass-dsl` install that ships `run-iket`
(4.6.0+). IKET is a **beta** feature; the API and output format may change.

---

## Golden rule

**Instrument coarsely, collect once, read the JSON — not the picture.**

The Perfetto timeline is for orientation. The actionable answer ("warp role R spends
X% of its life waiting on Y") comes from `helpers/analyze_trace.py` against the JSON
export. Start with 5-8 coarse ranges per warp role; add detail only where the coarse
trace already shows a problem.

---

## Tool selection: IKET or ncu

They answer different questions and **cannot run at the same time** (both need CUPTI
driver resources). Each collection is a separate run, so picking wrong costs a full
round.

| Question | Tool |
|---|---|
| Which warp role is the critical path? | IKET |
| Do producer and consumer warps overlap, or serialize? | IKET |
| What fraction of a warp's life is spent in a wait? | IKET |
| Is work imbalanced across CTAs / SMs? | IKET |
| How many bytes actually moved to/from DRAM? | ncu |
| What limits occupancy (registers, smem, warp slots)? | ncu |
| L2 hit rate, sector efficiency, instruction mix | ncu |
| Is the kernel memory-bound or compute-bound overall? | ncu |

Rule of thumb: IKET answers **timing and structure**, ncu answers **quantity and
resources**. If a hypothesis is about *ordering or waiting*, reach for IKET first.

---

## Workflow

### 1. Instrument

Put `iket.*` calls inside the `@cute.kernel` function. Calls in host-side `@cute.jit`
wrappers or Python launch code emit nothing.

```python
from cutlass.cute.experimental import iket

@cute.kernel
def kernel(...):
    life = iket.range_start("warp_life")

    if warp_idx == load_warp_id:
        iket.range_push("tma_issue")
        # ...
        iket.range_pop()

    if warp_idx == mma_warp_id:
        iket.range_push("mma_wait_ab")
        # ...
        iket.range_pop()

    iket.range_end(life)
```

For warp-specialized kernels, put **both ends of a range inside the same warp guard**.
A range that opens in one role and closes in another produces an undefined trace.

Instrumentation is free when not profiling: **IKET calls are stripped by default**. If
neither `run-iket` nor an explicit compile option enables IKET lowering, `iket.*` adds
no code to the final kernel. Markers can therefore live permanently in kernel source.

### 2. Collect

```bash
run-iket --output-dir ./iket_out --clobber \
  profile --postprocess all -- \
  python your_workload.py
```

`run-iket` makes two passes: a dry run to size device trace buffers, then the real
profiling run. Your workload therefore executes twice.

Outputs in `./iket_out/`: `*.pftrace` (Perfetto), `*.trace.json` (machine-readable),
`*.pftrace.gz`, and a self-contained `*.html` viewer.

### 3. Analyze

```bash
python .claude/skills/iket-trace/helpers/analyze_trace.py ./iket_out/*.trace.json
```

Reports per-warp-role lifetimes, the critical role, range time attribution as a
percentage of warp life, and CTA/SM distribution. Use `--roles` to name warp indices
so output reads in domain terms instead of raw warp ids.

For the timeline: open the generated `*.html`, or import the `*.pftrace` at
<https://ui.perfetto.dev/>.

---

## Non-negotiables

**The kernel must JIT-compile inside the profiled run.** Cached or already-compiled
kernels get no instrumentation and produce a silently empty trace. If the project has a
compilation cache, clear or bypass it before profiling. This is the single most common
cause of "IKET produced nothing".

**Never take performance numbers from an instrumented run.** IKET adds per-kernel entry
and exit overhead, can alter compiler behavior, and adds host-side cost. Timing gates
must come from a clean, uninstrumented run. Compare overhead only via kernel durations
inside the trace, never via application wall clock.

**Keep both ends of a range in the same warp role**, and keep placement warp-uniform.
Divergent `range_push` / `range_pop` may fail profiling or produce a wrong timeline.

**Budget the names.** At most 32 characters per name; keep unique names under ~30 total.
Timer granularity is 32 ns, so ranges shorter than that are indistinguishable.

**Do not run with ncu or nsys.** CUPTI resource conflict. Separate runs only.

---

## Reference

| File | Contents |
|---|---|
| `reference/01-workflow.md` | `run-iket` CLI options, two-pass model, artifacts |
| `reference/02-api.md` | The seven APIs, pairing rules, payload typing |
| `reference/03-instrumentation.md` | Where to place markers in warp-specialized kernels |
| `reference/04-pitfalls.md` | Failure modes and how each one presents |
| `reference/05-trace-format.md` | Verified JSON schema |
| `helpers/analyze_trace.py` | JSON → warp-role critical path and wait attribution |

Official documentation:
<https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/iket_profiling.html>
