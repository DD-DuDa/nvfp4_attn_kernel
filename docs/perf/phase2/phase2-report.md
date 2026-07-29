# Phase 2 — True decode query length

## Change

The quantizer and decode core now carry pre-quantized Q as
`[rows, 1, heads_q, 64]` rather than `[rows, 128, heads_q, 64]`.
`seqlen_q_static_one=True` makes output and LSE predicates use the true decode
length. Query scale factors remain quantizer-native and PackGQA-packed; no host
materialization was reintroduced.

## Numerical gate

`tests/kernel` plus benchmark tests: **56 passed**. Exact MHA/GQA/MQA FP4-Q
versus BF16-Q equality remains green, as do BF16-Q residual/index/scatter tests.

## High-batch clean performance

Clean grid: batch `{16,64,128}`, seqlen `{1024,4096,16384}`, MHA-8/GQA-4/MQA,
30 measured iterations after 10 warmups.

Geometric improvement relative to the Phase 1 padded-Q FP4-Q path:

| Head config | Geomean speedup | Range |
|---|---:|---:|
| MHA 8:8 | 0.89x | 0.69–1.01x |
| GQA 32:8 | **1.35x** | 0.98–1.74x |
| MQA 32:1 | **2.89x** | 0.97–10.50x |

The 1.5–4x expectation is met strongly for MQA and partially for GQA, but not
for MHA. This is consistent with the mechanism: MHA has no grouped-query head
packing to recover, while the benefit grows with Q-heads per KV-head. The
occasional MHA regression is recorded and not hidden; later phases must keep it
under observation.

Relative to D0 after this Phase, high-batch FP4-Q remains slower:

| Head config | 1024 | 4096 | 16384 |
|---|---:|---:|---:|
| MHA 8:8 | 2.52x | 2.31x | 1.97x |
| GQA 32:8 | 2.50x | 2.38x | 2.01x |
| MQA 32:1 | 2.51x | 3.03x | 3.46x |

## IKET high-batch evidence

Case: batch 64, seqlen 16384, GQA-4, pure FP4-Q. The persistent grid is 148
CTAs, all traced across all 148 SMs, one CTA/SM. Softmax0 owns the launch-tail
envelope. Dominant structural waits are:

- correction waiting on softmax: 89.8%
- load waiting for KV buffer release: 88.4%
- MMA waiting for P/O readiness: 79.4%
- epilogue waiting for correction: 99.6%

The grid is fully spread, so the next high-batch lever is intra-CTA
softmax/correction/P readiness rather than additional CTA count. This confirms
Phase 2b's prerequisite. DRAM byte quantification remains an ncu question and
is not inferred from this trace.

## Failed/negative branches

- Treating MHA as a padding-win target was disproved; there is no GQA packing
  redundancy to recover and some cases regress.
- The exact static “four CTAs per KV head” estimate is not used: persistent
  launch shape caps at SM count, and work tiles are consumed dynamically.
