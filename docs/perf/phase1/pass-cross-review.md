# Phase 1 Alternate-Model Cross-Review — Final Post-Fix Check

**Verdict: NEEDS_FIX**

**Reviewed HEAD:** `a1fe0ed50742ddcd5067d70260430bc7b514030d`

**Scope:** internal static cross-check only. No tests or benchmarks were run and
no production code was edited. Reviewed the prior final-review blockers, the
post-fix implementation, expanded negative tests, clean benchmark provenance,
host-repack removal, and the recorded 56-test gate.

## Summary

The prior implementation blockers are resolved in the current tree:

- The FP4-Q path bypasses BF16 query quantization and padded BF16 query scratch.
- Query scales are produced in the quantizer-native PackGQA layout and passed
  directly to the shared `decode_fp4` core; the former host-side repack/cache,
  `torch.take`, and `index_copy_` path is gone.
- BF16-Q and FP4-Q remain separate benchmark variants with correct contract
  labels.
- `short-case-final.json` is clean (`dirty: false`) and records the measured
  implementation revision `19edfbd`; HEAD `a1fe0ed` contains only evidence,
  review, and test additions after that implementation revision.
- The committed final transcript records `56 passed`.
- The approved Phase 1 boundary remains intact: pure page-aligned FP4-Q is
  supported, while FP4-Q plus BF16 residual is rejected and BF16-Q residual
  behavior remains supported.

One acceptance gap remains. AC-3 explicitly requires negative coverage for wrong
**device**, in addition to dtype, shape, strides, row/head count, and partial
arguments. The runtime validator checks devices, but the committed negative
suite does not directly exercise a wrong-device query or scale tensor. Under the
Phase 1 stop-on-failure rule, this keeps the verdict at NEEDS_FIX.

## Cross-check findings

### F-1 — HIGH: direct wrong-device negative coverage is still missing

`src/nvfp4_decode_kernel/_decode.py` validates that `query_fp4`,
`query_scales`, K/V tensors, page metadata, and residual tensors share the query
CUDA device. The current tests cover:

- missing, partial, and ambiguous query contracts;
- FP4-Q plus `query_row_indices` rejection;
- wrong packed query dtype and shape;
- wrong scale shape and dtype;
- wrong scale row/head metadata;
- wrong packed-query contiguity/head shape; and
- exact MHA, GQA, and MQA output equality.

However, `tests/kernel/test_fp4_decode_correctness.py` has no focused assertion
that a query or query-scale tensor on a different device is rejected before
launch. The test environment may expose only one CUDA device, but the required
contract still needs a direct device-mismatch regression (or an explicit
portable test using an available alternate device, with a skip only when no
alternate device exists).

**Required fix:** add a focused wrong-device negative case for the FP4-Q contract,
covering at least `query_fp4` or `query_scales`, while preserving the existing
metadata and exact-equality tests. Re-record the final test transcript afterward.

## Resolved prior blockers

| Area | Result | Cross-check |
|---|---|---|
| Single public API | PASS | `fp4_decode` remains the single public entry; BF16-Q and paired FP4-Q contracts are mutually exclusive. |
| Shared decode core | PASS | Both contracts converge on `decode_fp4`; no duplicate attention core or K/V re-quantization was introduced. |
| No FP4-Q Q quantization/scratch | PASS | The FP4-Q branch does not call `quantize_query` and passes `query_padded_bf16=None`. |
| No host scale repack | PASS | `_pack_sfq_for_gqa`, `_sfq_pack_cache`, `torch.take`, and `index_copy_` are absent from the implementation. `quantize_query(..., heads_kv=...)` writes the native layout directly. |
| Stream-safety concern from prior review | PASS | Removal of the mutable repack cache removes the previously identified shared-buffer hazard. |
| Residual-order decision | PASS | FP4-Q with BF16 residual is deliberately rejected per the approved Phase 1 ordering; BF16-Q residual behavior is retained. |
| Numerical gate | PASS | Recorded exact `torch.equal` coverage includes MHA `(8,8)`, GQA `(32,8)`, and MQA `(32,1)` pure page-aligned cases. |
| Negative validation implementation | PASS in code / NEEDS_FIX in coverage | Runtime checks cover device, dtype, shape, layout, storage offset, and head/row relationships; direct wrong-device test coverage is the remaining gap. |
| Benchmark separation and labels | PASS | `fp4_pure_bf16q` is `bf16_q_fp4_kv`; `fp4_pure_fp4q` is `fp4_q_fp4_kv`; both remain separately reported. |
| Clean benchmark evidence | PASS with provenance note | `short-case-final.json` records `19edfbd`, `dirty: false`, 10 warmups, 50 CUDA-event iterations, and the requested representative case. |
| Final test evidence | PASS as recorded | `test-results-final.txt` records `56 passed`; this review did not rerun tests. |

## Evidence cross-check

- Repository state at review: clean working tree, HEAD `a1fe0ed`.
- `19edfbd` is an ancestor of HEAD and is the only implementation revision
  between the earlier review and the clean benchmark artifact; the intervening
  HEAD changes are docs, review evidence, and negative-test additions.
- `short-case-final.json` records the representative
  `batch=1, seqlen=1024, heads_q=32, heads_kv=8` case with 10 warmups and 50
  iterations. It reports `0.32179264068603514 ms` for BF16-Q and
  `0.1090272045135498 ms` for FP4-Q, consistent with the documented ~2.95x
  improvement.
- The benchmark artifact uses the corrected FP4-Q contract label and records
  `dirty: false`.
- The shared `case_peak_memory_bytes` field is not treated here as isolated
  per-variant allocation proof; the verdict does not rely on such a claim.

## Final disposition

**NEEDS_FIX** — add the missing direct wrong-device negative regression required
by AC-3, then update the final test evidence. No host repack, benchmark-label,
clean-provenance, 56-test, or residual-order blocker remains. The deferred
FP4-Q + BF16 residual path is not a finding because it was explicitly approved
for a later phase.
