# Phase 1 — Pre-quantized FP4-Q contract

## Contract

- One public `fp4_decode` entry accepts either BF16 `query` or pre-quantized
  `query_fp4 + query_scales`.
- Both paths call the same `decode_fp4` core and compiled attention kernel.
- FP4-Q currently targets pure FP4 K/V with page-aligned lengths and rejects
  BF16 residual; this ordering was explicitly approved by the user. Existing
  BF16-Q residual semantics remain green.
- Query scale shape, dtype, storage offset and quantizer-native strides are
  validated before launch.

## Numerical evidence

The FP4-Q test quantizes the same BF16 query with the repository quantizer and
asserts exact `torch.equal` output against the BF16-Q entry for MHA-8, GQA-4,
and MQA-32. Contract-negative tests cover missing, partial, ambiguous,
indexed-FP4-Q and materialized-wrong-scale-layout inputs.

Latest gate: **55 passed** across `tests/kernel` plus benchmark infrastructure.
No numerical threshold changed.

## Clean short-case performance

Case: `batch=1, seqlen=1024, heads_q=32, heads_kv=8`, 10 warmups, 50 measured
iterations, CUDA events, no profiler or IKET.

| Path | GPU ms | Wall ms |
|---|---:|---:|
| FA4 split=1 | 0.04404 | 0.04208 |
| FA4 heuristic | 0.06029 | 0.05956 |
| FP4 BF16-Q | 0.35903 | 0.33606 |
| **FP4 FP4-Q** | **0.13178** | **0.13212** |

FP4-Q is **2.72x faster** than the BF16-Q composition (`0.367x` latency).
This exceeds the pre-registered 5–15% expectation because it removes not only
the query quantization kernel but also BF16-path scratch/allocation and
associated validation/composition work. It remains 2.99x slower than the D0
FA4 baseline in this cell, so later structural phases are still necessary.

`short-case.json` records both paths separately and uses FP4-Q versus D0 as the
D4 gate path.
