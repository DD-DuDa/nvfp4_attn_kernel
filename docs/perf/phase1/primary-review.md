# Phase 1 Primary RLCR Review

**Reviewer lane:** internal GPT-5.6-Terra/high primary review  
**Scope:** `adf98ad`, `902a8e3`, and `4c31c11`, reviewed against
`docs/plans/phase1-plan.md`, D2/D4, and task §6.6.  
**Review constraints:** no external reviewer transport, no product-code edits,
and no new test runs by this reviewer.

## Goal Alignment Summary

| Requirement | Classification | Review result |
|---|---|---|
| D2 single public entry; BF16-Q and keyword-only FP4-Q contracts are mutually exclusive | **PASS** | `interface.fp4_decode` remains the one public entry. `query_fp4` and `query_scales` are keyword-only; the composition layer rejects neither/both/partial query contracts. |
| Both contracts use the same decode core | **PASS** | Both branches converge on one call to `decode_fp4`; no duplicate attention kernel or K/V re-quantization was added. |
| FP4-Q avoids Q quantization and padded BF16 Q scratch | **PARTIAL / BLOCKER** | The FP4-Q branch does skip `quantize_query` and passes `query_padded_bf16=None`. However, every FP4-Q launch calls `_pack_sfq_for_gqa`, which materializes scale data into mutable scratch with `torch.take` and `index_copy_`; a new shape/device cache entry allocates `storage`, `gathered`, and index tensors. Thus the implementation does not establish the requested no-per-call scratch property. It also shares that mutable cache across streams. |
| Native FP4-Q layout validation | **PARTIAL** | Code validates packed E2M1 query dtype/shape/device/contiguity and E4M3 scale shape, native strides, device, and storage offset before launch. The added negative test exercises only one materialized-stride case plus query-contract cases; it does not evidence the requested wrong dtype, shape, device, row/head-count, and both partial-argument directions. |
| Exact MHA/GQA/MQA equality | **CODE/TEST PRESENT; EXECUTION UNVERIFIED** | The added parameterized test uses MHA `(8,8)`, GQA `(32,8)`, and MQA `(32,1)`, page-aligned `seqused_fp4=256`, repository `quantize_query`, and exact `torch.equal`. No immutable execution log for this commit was supplied. |
| Deferred FP4-Q + BF16 residual; retained BF16-Q residual behavior | **PASS IN DESIGN; EXECUTION UNVERIFIED** | The user decision is documented in the plan/task, and FP4-Q with residual is rejected because `query_padded_bf16` is absent. BF16-Q still constructs the padded residual input only when a residual K cache is present. Existing residual tests were not modified, but current-commit green evidence is missing. |
| Negative contracts | **PARTIAL** | Missing, partial (`query_fp4` only), ambiguous, FP4-Q with `query_row_indices`, and one invalid scale-layout case are covered. Coverage is incomplete for the full AC-3 negative matrix. |
| D4 benchmark separation | **PASS IN CODE; BLOCKED AS EVIDENCE** | `fp4_pure_fp4q` is implemented separately; summaries compare it with the better FA4 timing while retaining BF16-Q. The recorded short-case artifact is not attributable to the committed implementation. |
| §6.6 green Phase 1 close: clean evidence, full `tests/kernel` gate, primary and different-model cross review | **BLOCKER** | The repository contains a report assertion of “55 passed,” but no durable test transcript for the final commit and no Phase 1 different-model cross-review artifact. The only timing artifact records an older commit and a dirty tree. |

## Findings

### P0 — FP4-Q scale repacking uses mutable scratch on every decode

**Location:** `src/nvfp4_decode_kernel/_decode.py`, `_pack_sfq_for_gqa`
(lines 149–275 in `4c31c11`).

The FP4-Q route correctly bypasses `quantize_query` and does not allocate a
padded BF16 query tensor. It nevertheless always calls `_pack_sfq_for_gqa`
before dispatch. That helper:

1. allocates `storage`, `packed_bytes`, index tensors, and `gathered` for each
   previously unseen `(GQA ratio, shape, stride, device)` cache key; and
2. on every call, gathers the supplied query-scale bytes and writes them into
   the cached `packed_bytes` scratch before the decode launch.

This is not a direct no-scratch FP4-Q path. A warm cache removes repeated
allocation for one shape, but it does not remove the per-call scratch
materialization; a cold shape still allocates during a decode call. The global
cache also has no stream dimension or synchronization, so overlapping calls on
different CUDA streams can overwrite the same packed scale buffer before an
earlier decode consumes it.

**Required disposition:** stop the Phase 1 mainline. Either make the compiled
core consume the quantizer-native FP4-Q scale layout directly, or provide a
caller-owned/prepacked contract whose allocation and lifetime are outside the
decode call. Then add focused proof that no FP4-Q decode invokes quantization,
allocates scratch, or races across streams.

### P1 — The claimed benchmark evidence predates the implementation commit

**Location:** `docs/perf/phase1/short-case.json` and
`docs/perf/phase1/phase1-report.md`.

`short-case.json` identifies Git commit
`adf98adf37d3e966391f73ce20231af8350a233a` and `"dirty": true`. The actual
FP4-Q implementation and benchmark variant arrived in `4c31c11`. Therefore
the artifact cannot be evidence for the committed FP4-Q path or for the
reported FP4-Q measurement. The report's table and “2.72x” claim have no
traceable clean-run provenance for `4c31c11`.

**Required disposition:** rerun the stated clean short case from a clean,
committed Phase 1 tree and archive its JSON with the resulting commit identity.
Keep BF16-Q and FP4-Q results separate and retain FP4-Q versus better FA4 as
the D4 gate path.

### P1 — Phase 1 close evidence is incomplete

`phase1-report.md` states “55 passed across `tests/kernel` plus benchmark
infrastructure,” but no Phase 1 RLCR summary/review artifact records the
command, result, elapsed time, and commit for that run. The available
`.humanize/rlcr/2026-07-29_20-37-08` artifacts are Phase 0 records and end at
51 tests before the Phase 1 implementation. No different-model Phase 1
cross-review artifact is present.

Under §6.6, the Phase must end at a green `tests/kernel` commit. Under AC-6,
the Phase additionally needs clean representative timings and both required
internal reviews. The report assertion alone is not sufficient auditable
evidence, particularly because the adjacent timing JSON is dirty and
pre-implementation.

**Required disposition:** after resolving P0, run and record the final
`tests/kernel` gate (and the intended benchmark-infrastructure tests if they
are included in the advertised count), create the distinct internal
different-model cross review, and close on a clean green commit.

### P2 — Native-layout negative coverage is narrower than the plan

The implementation contains substantial validation, including query FP4
shape/dtype/device/contiguity, scale shape/strides/device/storage offset, and
head divisibility. But the new regression test only demonstrates one
materialized scale-layout failure along with query contract failures. It does
not add targeted tests for query FP4 dtype/shape/device/contiguity, scale
dtype/shape/device/storage offset, or mismatched row/head counts. This is not
the stop-the-task finding, but it leaves AC-3's explicit negative contract
matrix incompletely demonstrated.

## Evidence Reviewed

- `adf98ad`: introduced the Phase 1 contract plan.
- `902a8e3`: recorded the approved ordering: Phase 1 supports pure,
  page-aligned FP4-Q/K/V and defers FP4-Q plus BF16 residual until after the
  speed target.
- `4c31c11`: introduced the optional keyword-only FP4-Q inputs, common core
  dispatch, numerical tests, separate benchmark variant, report, and JSON
  evidence.
- `docs/tasks/2.fp4_decode_speedup.md` D2/D4 and §6.6.
- `docs/plans/phase1-plan.md`.
- Implementation and test sources listed in the Phase 1 commit.

No new tests or benchmarks were run for this review, per the review request.

## Mainline Progress Verdict

**STOP — Phase 1 is not verified.**

The single-entry API, direct FP4-Q contract, shared decode core, deferred
residual behavior, exact-equality test design, and D4 benchmark separation are
substantively on the intended path. However, the FP4-Q route still repacks
query scales into mutable scratch on every call, contrary to the Phase 1
no-per-call-scratch requirement and with a multi-stream correctness hazard.
In addition, the performance artifact predates the implementation and is dirty,
and the required final green/cross-review evidence is absent.

Because Phase 1 is the §6.6 stop-on-failure phase, do not start a later Phase
until the P0 implementation issue and the associated final evidence gaps are
resolved.

INCOMPLETE
