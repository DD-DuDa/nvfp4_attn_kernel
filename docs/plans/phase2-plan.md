# Phase 2 Plan — Remove padded query tiles and redundant KV work

## Goal Description

For the pure page-aligned FP4-Q path, remove the artificial 128-query-row padding so PackGQA presents only real decode query/head rows to the shared core. Improve batch >=16 per-CTA throughput without changing D0–D8, K/V layouts, numerical gates, or existing BF16-Q residual semantics.

Expected effect: static analysis predicted redundant M tiles/CTA for grouped query, but Phase 0's low-batch trace falsified the exact four-CTA estimate for that case. Re-establish the grid and launch-tail evidence on a high-batch target before/after the change. Expected clean improvement is 1.5–4x for GQA-heavy high-batch cases; lower results require a documented bottleneck correction, not gate relaxation.

## Acceptance Criteria

- **AC-1:** FP4-Q path no longer allocates or carries 128 padded query rows; its logical Q length is one and PackGQA M extent is real `qhead_per_kvhead`.
- **AC-2:** MHA/GQA/MQA exact numerical gates and all existing residual/index/scatter semantics stay green; no tolerance changes.
- **AC-3:** High-batch clean performance improves relative to Phase 1, reported separately by head configuration; low batch and BF16-Q do not materially regress.
- **AC-4:** IKET high-batch evidence records grid/CTA/SM spread, launch-tail role, and relevant waits before and after. If the traffic question remains load-related, a separate ncu run quantifies DRAM bytes and records why IKET cannot answer it.
- **AC-5:** Query/K/V scale layouts remain native; no per-call transpose/materialized scale copy or experimental switch enters production.
- **AC-6:** Phase ends with `tests/kernel` green, failure branches logged, a committed state, and internal Terra/high plus alternate-model review.

## Path Boundaries

- May change `_kernel.py`, `_decode.py`, `fp4_decode_kernel.py` query layout/static-length handling, quantizer output shape/layout if shared-core invariants require it, tests, benchmark and Phase 2 evidence.
- Must not implement split-K, host-sync cleanup, residual FP4-Q, prefill/backward, or change `_fa4/` unless the decode core demonstrably requires it.

## Dependencies and Sequence

1. Capture clean and IKET high-batch FP4-Q baselines for MHA/GQA/MQA; record CTA counts and tail/waits.
2. Trace the padded query dimensions from public contract through pointer shape, PackGQA and scheduler grid.
3. Change one layout boundary at a time so Q represents one sequence row while preserving head grouping and scale TMA layout.
4. Add/adjust exact tests for the new logical shape and run full numerical gate each round.
5. Run clean high/low batch comparison and IKET; use ncu only for byte counts.
6. Record expected vs measured, failed layouts, green commit and dual-model review.

## Implementation Notes

- Every shell starts with the mandated environment lines.
- Pure FP4/page-aligned performance is the current mainline; BF16-Q residual remains a regression gate only.
- No external Codex transport. Primary reviewer is GPT-5.6-Terra/high and cross-review uses another internal model.
