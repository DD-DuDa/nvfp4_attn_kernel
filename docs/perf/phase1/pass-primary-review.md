# Phase 1 Final PASS-Readiness Primary Audit

**Reviewer lane:** internal GPT-5.6-Terra/high primary  
**Audited HEAD:** `a1fe0ed50742ddcd5067d70260430bc7b514030d`
(`Complete FP4 query contract validation`)  
**Implementation revision:** `19edfbd424cb7d633e597b9beb66401fdcd90364`
(`Eliminate per-call FP4 query scale repacking`)  
**Scope:** Phase 1 acceptance criteria in
`docs/plans/phase1-plan.md`; D2 and D4 in
`docs/tasks/2.fp4_decode_speedup.md`; the approved FP4-Q + BF16-residual
deferral; the prior primary/cross-review findings; and the review-fix commit
at HEAD.  
**Audit method:** static source/evidence audit only. No tests, benchmarks, or
production-code edits were performed by this audit.

## Verdict

# NEEDS_FIX

The implementation is materially ready: the original dirty/stale benchmark
evidence, FP4-Q contract-label error, host-side scale-repacking/cache path, and
most negative-validation gaps have been addressed. The clean benchmark is
correctly attributable to the implementation revision `19edfbd`, and the final
committed transcript records 56 passing tests.

Phase 1 nevertheless cannot close as **COMPLETE** yet. Two close conditions
remain:

1. The expanded negative tests still do not directly exercise the required
   **wrong-device** FP4-Q rejection case.
2. The available different-model cross-review of the final review-fix HEAD
   `a1fe0ed` (`pass-cross-review.md`) ends **NEEDS_FIX**, not PASS, because it
   independently identifies the same missing wrong-device regression. The
   earlier committed alternate review of `1e97864` is also non-passing.

These are Phase 1 close-out items under AC-3 and AC-6, respectively. The
approved FP4-Q residual deferral is not a finding.

## Evidence Reviewed

| Evidence | Assessment |
|---|---|
| `19edfbd` implementation | Native PackGQA query-scale layout is produced by `quantize_query(..., heads_kv=...)`; the FP4-Q path consumes it directly. |
| `docs/perf/phase1/short-case-final.json` | Clean (`dirty: false`) 10-warmup / 50-iteration CUDA-event benchmark at `19edfbd`; it separates BF16-Q and FP4-Q and labels FP4-Q `fp4_q_fp4_kv`. |
| `docs/perf/phase1/test-results-final.txt` | Committed final transcript records `56 passed` in 65.46 s. This audit did not rerun it. |
| `a1fe0ed` test changes | Adds direct scale-dtype, scale-row-count, and scale-head-count rejection cases in addition to prior partial, ambiguous, packed-dtype/shape, scale-layout, and indexed-FP4-Q cases. |
| Prior and final reviews | `primary-review.md` / `cross-review.md` identify the original implementation and evidence defects; `final-primary-review.md` / `final-cross-review.md` identify the post-`19edfbd` gaps; `pass-cross-review.md` confirms the remaining wrong-device gap at `a1fe0ed`. |

The benchmark provenance is intentionally tied to `19edfbd`, rather than to
the later evidence-only commit `1e97864` or the subsequent test-only commit
`a1fe0ed`. `19edfbd..1e97864` changes only Phase 1 evidence/report files;
`1e97864..a1fe0ed` changes only review artifacts, the final transcript, and
negative tests. Therefore the clean timing remains attributable to the
reviewed implementation revision and does not claim a false exact-HEAD
measurement.

## Acceptance-Criteria Assessment

| Criterion | Result | Audit basis |
|---|---|---|
| **AC-1 — one public entry and mutually exclusive contracts** | **PASS** | `interface.fp4_decode` remains the single public entry. `_kernel.fp4_decode_impl` accepts BF16 `query` or paired `query_fp4`/`query_scales`, and rejects neither, partial, and ambiguous contracts. |
| **AC-2 — shared core and preserved BF16 semantics** | **PASS** | Both contract branches converge on one `decode_fp4(...)` invocation. There is no second decode core or K/V quantization in `fp4_decode`; BF16-Q retains `query_row_indices`, residual, `out`, and `out_indices` handling. The compile-cache key is independent of the query contract. |
| **AC-3 — quantizer-native layout validation and negative coverage** | **NEEDS_FIX** | Source validation rejects packed-query dtype/shape/contiguity, native scale shape/strides/storage offset/dtype, device mismatches, and incompatible row/head metadata before launch. Committed tests now directly cover dtype, shape, stride/layout, row, head, partial, ambiguous, and indexed inputs. They do **not** directly cover a wrong-device FP4-Q tensor/scale case required by the Phase 1 negative matrix. |
| **AC-4 — exact pure-FP4 numerical gate** | **PASS by committed transcript** | The parameterized test quantizes once with the repository quantizer and uses `torch.equal` for MHA `(8,8)`, GQA `(32,8)`, and MQA `(32,1)` with page-aligned pure-FP4 K/V. No tolerance was relaxed. |
| **AC-5 — separately reported paths and D4 gate path** | **PASS** | `fp4_pure_bf16q` and `fp4_pure_fp4q` are distinct variants. Final JSON labels them `bf16_q_fp4_kv` and `fp4_q_fp4_kv`; the summary compares each to the lower-latency FA4 split=1/heuristic result. |
| **AC-6 — green close evidence and dual-model review** | **NEEDS_FIX** | The committed transcript supplies the 56-pass gate and the clean benchmark is provenance-valid. The available different-model review of `a1fe0ed` is `NEEDS_FIX`, not PASS, because the direct wrong-device negative test is still absent. |

## D2 and D4 Verification

| Decision | Result | Verification |
|---|---|---|
| **D2 — single-entry dual query contract** | **PASS** | The public entry is still `fp4_decode`; BF16-Q quantizes then uses the shared core, while FP4-Q passes native `query_fp4` and `query_scales` directly to that same core. |
| **D4 — FP4-Q performance path measured separately against better FA4** | **PASS** | The clean artifact records both FP4 paths. Its summary selects the minimum CUDA-event time of FA4 split=1 and FA4 heuristic as the baseline. For the recorded representative GQA case, FP4-Q is 0.10903 ms and BF16-Q is 0.32179 ms; both are separately preserved. |

The recorded representative short case is not a D3 full-grid success claim.
It is valid Phase 1 contract/performance evidence only; later structural phases
remain necessary for the overall 2x target.

## Prior-Finding Disposition

| Prior finding | Disposition |
|---|---|
| Dirty / wrong-revision short-case evidence | **Resolved.** `short-case-final.json` records `commit=19edfbd...` and `dirty=false`. |
| FP4-Q benchmark mislabeled as BF16-Q | **Resolved.** `fp4_pure_fp4q` is labeled `fp4_q_fp4_kv` in the harness and final artifact. |
| Per-call host GQA scale repack, allocations, and mutable cache | **Resolved.** `19edfbd` removes `_pack_sfq_for_gqa`, `_sfq_pack_cache`, `torch.take`, and `index_copy_`; the quantizer writes the kernel-native scale layout directly. |
| Shared peak-memory field could not prove per-path allocation behavior | **Resolved as a reporting issue.** The Phase 1 report does not rely on that shared field to prove no FP4-Q allocation. Source inspection establishes that the FP4-Q branch does not create padded BF16 Q scratch or invoke `quantize_query`. |
| Compile-cache sharing lacked a dedicated regression | **Pass by source structure, not a new blocker.** Both paths use `decode_fp4`, and the cache key contains device/head/residual/output-scatter properties rather than the input contract. |
| Missing negative tests after `19edfbd` | **Partially resolved.** `a1fe0ed` adds scale dtype, row-count, and head-count cases. A direct wrong-device regression remains absent. |
| Final alternate-model review | **Open.** `pass-cross-review.md` reviews `a1fe0ed`, but it is correctly `NEEDS_FIX` for the same direct wrong-device test gap. A PASS review is still required after that fix. |

## Approved Residual Deferral

**PASS — intentionally deferred, not a regression.**

The FP4-Q path rejects a BF16 residual because it has no BF16 padded-query
buffer. This exactly matches the user-approved Phase 1 boundary: FP4-Q is
limited to pure page-aligned FP4 K/V, while FP4-Q + BF16 residual is deferred
until after the speed target. The existing BF16-Q residual path remains in the
shared public entry and is included in the committed 56-pass evidence.

## Required Final Close-Out

1. Add a focused pre-launch negative regression for wrong-device FP4-Q input
   (preferably covering an off-device `query_scales` and/or `query_fp4` against
   otherwise valid CUDA decode tensors). It must assert a clear validation error
   before compilation/launch.
2. Record a different-internal-model cross-review of `a1fe0ed` that verifies:
   the native scale layout, repack/cache removal, expanded negative matrix,
   D2/D4 behavior, valid `19edfbd` benchmark provenance, 56-pass transcript,
   and the approved residual deferral.
3. If that cross-review passes without further findings, update this primary
   audit to a final PASS/COMPLETE disposition. No benchmark rerun is required
   solely because `a1fe0ed` changes tests and review evidence rather than the
   measured implementation.

**Terminal status: NEEDS_FIX — Phase 1 work remains; do not mark COMPLETE.**
