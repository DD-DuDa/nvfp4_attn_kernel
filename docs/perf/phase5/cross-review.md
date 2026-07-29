# Phase 5 alternate cross-review — PASS

## Scope

Internal alternate cross-check of commit `f44fa70`
(`f44fa708b0b95b61a7f6a990e50abdc41a0bc98f`), covering the recorded Phase 5
grid math, the D3 decision, the roofline statement, and the mandated
stop/no-residual disposition. No production code or benchmark was changed.

## Decision

**PASS.** The stop report is mathematically consistent with
`docs/perf/phase5/final-grid.json`, its clean-grid/provenance claims are
supported, and it does not continue into the deferred residual phase after
the failed D3 gate.

## Cross-checks

### 1. Grid completeness and provenance — PASS

The artifact records the required batches
`{1, 2, 4, 8, 16, 32, 64, 128}`, seqlens
`{1024, 4096, 16384, 65536}`, and head configurations MHA `8:8`, GQA
`32:8`, and MQA `32:1`: 96 cases and 384 rows across the four variants.
`requested_complete` is true, `requested_missing` and `skipped` are empty,
and all 384 rows have status `ok`.

The JSON provenance records the measured artifact as `dirty: false` at
`db5326f703d4d838c68c713409fd4fe28a66fac1`, matching the report's statement
that the grid was clean. The checked repository is also clean at the review
commit `f44fa70`; no post-report edits are present.

### 2. D3 ratio and failure math — PASS

The report's D3 ratios match `summary.paths.fp4_pure_fp4q` and recomputation
from each FP4-Q GPU time divided by the better (minimum GPU-time) `fa4_bf16`
or `fa4_split` result for the same case:

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 2.558 | 2.697 | 3.038 | 3.555 |
| GQA 32:8 | 2.526 | 2.631 | 3.033 | 3.587 |
| MQA 32:1 | 2.497 | 3.081 | 4.722 | 9.234 |

Each D3 cell must be `<= 0.5`; every listed geomean is above that bound, so
the reported **FAIL** is required rather than discretionary. Independent
recomputation gives the all-grid FP4-Q geomean `3.3222219579708008` and the
maximum ratio `19.128954239965235` at MQA, batch 2, seqlen 65536, matching
the report's rounded `3.322x` and `19.129x`.

The “additional speedup” table is also correct: reaching a ratio of `0.5`
from a current ratio `r` requires `2r`; the displayed values are the exact
`2r` values rounded to two decimals.

### 3. Roofline statement — PASS

The recorded KV sizes are BF16 `0.00390625 GiB` and FP4
`0.0010986328125 GiB`, whose ratio is exactly `3.555555...`; therefore FP4
uses `1/3.556` of the BF16 KV bytes, as stated. Under equal effective
bandwidth, the ideal FP4 time ratio is `1/3.555555... = 0.28125`.
The D3 target ratio `0.5` is therefore `0.5 / 0.28125 = 1.777777...` above
that ideal same-bandwidth floor, matching the report's **1.778x** statement.

This supports the limited conclusion actually made: the 2x target is not
excluded by the KV-byte floor. It does **not** claim that the measured
kernel is bandwidth-bound or that the target will be achieved. The report
appropriately distinguishes the unchanged byte ratio from the current
kernel's much slower measured ratios.

### 4. D3 failure, mandated stop, and residual prohibition — PASS

The Phase 5 plan requires stopping if any D3 gate fails, with no residual
phase and no threshold change. The report:

- records the complete failed D3 table and the deficit;
- preserves the D0/better-FA4 baseline and numerical/interface gates;
- explicitly states that execution stops for human decision;
- explicitly states that the post-speedup FP4-Q + BF16 residual phase was not
  started; and
- lists future directions only as requiring a new human-approved plan.

The earlier failed branches are recorded consistently with the Phase 2b and
Phase 3 ledgers: the warp-15 load attempt failed at launch, the epilogue
attempt stalled, cooperative softmax was not partially attempted, and split-K
failed the unchanged `0.99` cosine gate and was reverted. No residual
implementation or residual benchmark evidence is introduced by `f44fa70`.

## Final disposition

**PASS — accept `docs/perf/phase5/final-report.md` as the Phase 5 stop
report. No fixes, reruns, threshold changes, residual Phase, or further
execution are authorized by this cross-review.**
