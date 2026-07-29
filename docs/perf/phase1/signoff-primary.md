# Phase 1 Primary Signoff

**Reviewer lane:** internal GPT-5.6-Terra/high primary  
**Signoff HEAD:** `1ca82285ed34065f1d562a2aa4e581c4f6c99c7c`  
**Implementation revision:** `19edfbd424cb7d633e597b9beb66401fdcd90364`  
**Review mode:** internal static signoff; no implementation edits, test rerun, or
benchmark rerun were performed by this review.

## Verdict

# PASS — COMPLETE

Phase 1 fully passes its stated close criteria. The final HEAD resolves the
only remaining primary and alternate-review blocker: direct FP4-Q wrong-device
negative coverage. `1ca8228` adds a CPU `query_scales` tensor with the native
shape and strides and asserts the pre-launch CUDA-device validation error.
The committed final gate records **56 passed**.

The implementation revision remains `19edfbd`: later commits through HEAD add
only review/evidence material and the focused negative regression. Therefore
the clean timing artifact remains valid implementation provenance rather than
an incorrectly claimed exact-HEAD benchmark.

## Final Acceptance-Criteria Disposition

| Criterion | Result | Signoff basis |
|---|---|---|
| **AC-1 — one public entry, mutually exclusive contracts** | **PASS** | `interface.fp4_decode` remains the sole public entry. BF16 `query` and paired `query_fp4`/`query_scales` are mutually exclusive, and neither/partial/ambiguous cases are rejected. |
| **AC-2 — shared core and preserved BF16 semantics** | **PASS** | `_kernel.fp4_decode_impl` routes both contracts into one `decode_fp4` core. The BF16 path retains indexed-query, residual, `out`, and `out_indices` behavior; no K/V re-quantization or duplicate decode core was introduced. |
| **AC-3 — native layout validation and negative matrix** | **PASS** | Validation covers packed-query dtype, shape, contiguity, device; native query-scale dtype, shape, strides, storage offset, device; and row/head relationships. Tests cover partial/ambiguous inputs, dtype, shape, layout, rows, heads, indexed FP4-Q rejection, and the final direct wrong-device `query_scales` regression. |
| **AC-4 — exact pure-FP4 numerical gate** | **PASS by committed transcript** | The FP4-Q equality test uses repository-produced quantized Q and `torch.equal` for MHA `(8,8)`, GQA `(32,8)`, and MQA `(32,1)` page-aligned pure-FP4 cases. |
| **AC-5 — separate BF16-Q and FP4-Q performance paths** | **PASS** | The harness separately reports `fp4_pure_bf16q` (`bf16_q_fp4_kv`) and `fp4_pure_fp4q` (`fp4_q_fp4_kv`) and compares each with the lower-latency recorded FA4 variant. |
| **AC-6 — green gate, clean evidence, and dual-model review close** | **PASS** | `test-results-pass.txt` records `56 passed` in 65.74 s. `short-case-final.json` records `dirty: false`, 10 warmups, 50 CUDA-event iterations, and the implementation commit. The different-model post-fix signoff in `signoff-cross.md` reviews this exact HEAD and records PASS after verifying the new wrong-device regression. |

## Prior-Finding Closure

| Prior finding | Final disposition |
|---|---|
| Dirty or stale benchmark provenance | **Resolved.** `short-case-final.json` is clean and identifies `19edfbd`, the measured implementation revision. |
| FP4-Q benchmark mislabeled as BF16-Q | **Resolved.** The final artifact labels FP4-Q `fp4_q_fp4_kv` and retains BF16-Q as a separate variant. |
| Host-side GQA scale repack / mutable cache / `torch.take` / `index_copy_` | **Resolved.** The quantizer creates the PackGQA-native scale layout directly, and the FP4-Q decode route consumes it unchanged. No host repack/cache path remains. |
| FP4-Q query quantization or padded BF16 scratch | **Resolved.** FP4-Q bypasses `quantize_query` and sets `query_padded_bf16=None`. |
| Native negative-test matrix incomplete | **Resolved.** Earlier dtype/layout/row/head additions are present; `1ca8228` supplies the final required direct wrong-device check. |
| Lack of passing post-fix alternate review | **Resolved.** `signoff-cross.md` is a PASS alternate-model signoff of exact HEAD `1ca8228`; it verifies the direct wrong-device regression and finds no remaining Phase 1 blocker. |
| FP4-Q + BF16 residual unsupported | **Approved deferral, not a finding.** Phase 1 intentionally supports pure page-aligned FP4-Q/K/V only. BF16-Q residual semantics remain protected; FP4-Q plus BF16 residual is explicitly deferred until after the speed target. |

## Performance Evidence and Scope

The clean representative short case is `batch=1`, `seqlen=1024`,
`heads_q=32`, `heads_kv=8`. The artifact records:

| Path | GPU time |
|---|---:|
| FP4 BF16-Q | 0.32179 ms |
| FP4 FP4-Q | 0.10903 ms |

This demonstrates the Phase 1 interface-path benefit while retaining separate
BF16-Q accounting. It is not a claim that the later Phase 5 full-grid 2x
performance target has already been reached.

## Final Scope Decision

No implementation change is requested. No host repack remains. The direct
wrong-device regression is present, the 56-pass transcript is recorded, clean
benchmark provenance is acceptable, and the residual deferral is approved.

**Terminal status: COMPLETE — Phase 1 is fully signed off.**
