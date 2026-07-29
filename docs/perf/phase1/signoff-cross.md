# Phase 1 Alternate Signoff — Final Post-Fix Review

**Verdict: PASS**

**Reviewed HEAD:** `1ca82285ed34065f1d562a2aa4e581c4f6c99c7c` (`1ca8228`)

**Scope:** Internal static alternate-model signoff only. No production code,
benchmark, or test files were edited by this review, and tests/benchmarks were
not rerun.

## Disposition of the prior blocker

**PASS — wrong-device validation coverage is fixed.**

`tests/kernel/test_fp4_decode_correctness.py::test_prequantized_query_contract_rejects_bad_tensor_metadata`
now constructs `cpu_scales` with the quantizer-native shape and strides and
passes it with otherwise valid CUDA FP4-Q inputs. The call is asserted to raise
`ValueError` matching `query_scales must be a CUDA tensor`.

This directly exercises the missing device-mismatch contract before launch.
The implementation gate in `src/nvfp4_decode_kernel/_decode.py` checks every
FP4-Q/KV/metadata tensor through `_require_cuda_tensor`, and the query-scale
case is now covered by a committed negative regression. The prior
wrong-device-gap finding is therefore resolved.

## Phase 1 close-out checks

- Single `fp4_decode` entry with mutually exclusive BF16-Q and FP4-Q contracts: **PASS**.
- BF16-Q and FP4-Q converge on the shared `decode_fp4` core: **PASS**.
- FP4-Q bypasses BF16 query quantization and padded BF16 query scratch:
  **PASS**.
- Quantizer-native query-scale layout is consumed directly; host-side scale
  repack/cache and `torch.take`/`index_copy_` path are absent: **PASS**.
- Required dtype, shape, layout/stride, row/head, partial, ambiguous, and
  device negative validation coverage: **PASS**.
- Exact pure page-aligned MHA/GQA/MQA FP4-Q numerical gate: **PASS** by the
  committed Phase 1 transcript.
- BF16-Q and FP4-Q benchmark variants remain separately reported with clean
  provenance: **PASS**.
- Committed kernel gate records `56 passed`: **PASS**.
- Approved Phase 1 boundary is preserved: pure page-aligned FP4-Q is covered;
  FP4-Q plus BF16 residual remains intentionally deferred, while BF16-Q
  residual behavior remains protected: **PASS**.
- Primary review and this post-fix alternate review are both passing at the
  reviewed HEAD: **PASS**.

## Final disposition

**PASS — no remaining Phase 1 blocker identified.** The prior alternate-review
blocker was limited to direct wrong-device negative coverage; HEAD `1ca8228`
adds that regression and preserves the validated implementation and evidence.
Phase 1 may be signed off.
