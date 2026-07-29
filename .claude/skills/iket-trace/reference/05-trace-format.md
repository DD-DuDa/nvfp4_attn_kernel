# Trace JSON Schema

Verified against `nvidia-cutlass-dsl` 4.6.0 output. IKET is beta; treat the shape as
observed rather than contractual, and re-check after an upgrade.

## Top level

```json
{
  "graphLaunches": {},
  "launches": [ ... ],
  "locationTable": [ ... ],
  "stringTable": [ "kernel_total", "loop_body", "loop_mark" ]
}
```

`stringTable` interns every marker and range name. Events reference it by index.

## locationTable

One entry per traced warp instance. Events reference it by index (`locIdx`).

```json
{ "ctaId": [0, 0, 0], "gpcId": -1, "smId": 142, "tpcId": -1, "warpId": 0 }
```

`warpId` is the warp index within the CTA, so it maps directly onto warp-role
assignments. `gpcId` and `tpcId` are `-1` when the profiler cannot read SM topology
(see `04-pitfalls.md`); `smId`, `ctaId`, and `warpId` remain valid.

## launches

One entry per kernel launch.

```json
{
  "kernelName": "kernel_cutlass__trace_kernel__0",
  "gridDimX": 2, "gridDimY": 1, "gridDimZ": 1,
  "blockDimX": 64, "blockDimY": 1, "blockDimZ": 1,
  "contextId": 1, "gridId": 1,
  "markers": [ ... ],
  "ranges": [ ... ],
  "warpLifetimes": [ ... ]
}
```

### warpLifetimes

Start and end of each traced warp. The most useful array for warp-specialized kernels:
grouping by `warpId` gives per-role duration directly, and the longest role is the
critical path.

```json
{ "locIdx": 0, "startTs": 1785326265766439520, "endTs": 1785326265766439968 }
```

### markers

```json
{ "locIdx": 0, "markerNameIdx": 2, "timestamp": 1785326265766439712, "color": 4294967295 }
```

### ranges

```json
{
  "rangeNameIdx": 1,
  "startTs": 1785326265766439712,
  "endTs": 1785326265766439744,
  "warpLocIdxs": [0, 0],
  "internalEvents": [
    { "eventId": 2,  "timestamp": 1785326265766439712 },
    { "eventId": 31, "timestamp": 1785326265766439744 }
  ],
  "rangeId": 844017592,
  "rangeType": 2,
  "rangeScope": 0,
  "rangeColor": 4294967295
}
```

`warpLocIdxs` holds the `locationTable` indices for the start and end events. They match
for a well-formed range; a mismatch means the range crossed warps and the trace should be
distrusted. `rangeId` is stable per instrumentation site, so all iterations of one loop
phase share it — useful for aggregating a recurring phase.

Timestamps are nanoseconds with 32 ns granularity, which is why consecutive values step
by 32.

## Sanity checks

Before drawing conclusions:

- `len(markers)` should equal `emissions per warp x warps x CTAs`. A shortfall suggests
  buffer overflow.
- Every `range` should have `startTs <= endTs` and matching `warpLocIdxs`.
- Distinct `smId` values indicate how many SMs the launch actually reached; a small
  number on a large grid points at a scheduling or occupancy problem rather than a
  per-CTA throughput problem.
