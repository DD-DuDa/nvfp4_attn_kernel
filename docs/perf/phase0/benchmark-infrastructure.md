# Phase 0a Benchmark Infrastructure Evidence

## Contract and baseline

- Clean GPU timing uses CUDA events; Torch Profiler is only a separate kernel-breakdown pass.
- FA4 always runs through `flash_attn_varlen_func`.
- D0 is encoded as the lower CUDA-event latency of `fa4_bf16` (`num_splits=1`) and `fa4_split` (FA4 heuristic); the chosen variant and split count are stored per ratio.
- D4 is explicit: `fp4_pure_bf16q` is timed now, while `fp4_pure_fp4q` is represented as `unavailable_until_phase1` rather than fabricated.
- Default coverage is batches `{1,2,4,8,16,32,64,128}`, seqlens `{1024,4096,16384,65536}`, and MHA/GQA/MQA head configurations.

## Provenance

Schema v2 records the environment path, Python and package versions, GPU identity and capacity, Git commit/branch/dirty state, exact arguments, timestamp, baseline definition, coverage and skipped cases.

## Per-kernel attribution

`kernel-breakdown.json` is a clean, separate Torch-Profiler run for
`batch=1, seqlen=1024, heads_q=32, heads_kv=8`. It contains 12 non-empty GPU
entries. The BF16-Q query quantization kernel is 0.00611 ms and 11.15% of the
summed profiler kernel time for that run. These figures diagnose composition;
the CUDA-event `gpu_ms` remains the benchmark comparison value.

## Stability

Three independent clean runs of `batch=16, seqlen=16384, GQA-4`, each with 10
warmups and 30 measured iterations:

| Variant | Run 1 ms | Run 2 ms | Run 3 ms | Geomean ms | Range / geomean |
|---|---:|---:|---:|---:|---:|
| FA4 split=1 | 0.164074 | 0.163996 | 0.164517 | 0.164195 | 0.318% |
| FA4 heuristic | 0.164378 | 0.163984 | 0.164397 | 0.164253 | 0.251% |
| FP4 BF16-Q | 0.637499 | 0.635958 | 0.638197 | 0.637217 | 0.351% |

All are below the 3% stability threshold.

## Coverage and long-case capacity smoke

- `phase0a_coverage_smoke.json` (preserved outside the repository during development) exercised batches 1/2 at seqlen 1024 for MHA, GQA, and MQA with no skipped cases.
- Large setup quantization is chunked by pages outside measured decode calls. This avoids a CuTe launch-size failure while preserving the quantizer's native packed/scale layouts.
- `batch=128, seqlen=65536, MHA-8` ran all four result entries (including the explicit unavailable FP4-Q row) with **48.000 GiB peak allocated memory**, below the 178.35 GiB device capacity. FA4 split=1, FA4 heuristic and FP4 BF16-Q all completed; no max-token skip occurred.

The full grid is intentionally deferred to Phase 5 for the final gate and to
Phase-level checkpoints after Phase 1 makes the FP4-Q path available. Phase 0a
establishes that the declared grid is representable, long cases are no longer
silently skipped, and all required aggregation/provenance fields exist.
