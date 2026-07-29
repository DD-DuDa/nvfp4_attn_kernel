# Phase 2 alternate cross-review — NEEDS_FIX

**Reviewed commit:** `f24cfe0` (`Use true decode query length in FP4 core`)

**Scope:** internal read-only review of the commit and the checked-in Phase 2
artifacts. No source or benchmark edits were made.

## Verdict

**NEEDS_FIX.** The core shape/layout change is internally coherent for the
pre-quantized decode path, and the recorded numerical gate is useful, but the
Phase 2 sign-off evidence is not provenance-clean and the regression/coverage
claims are incomplete. The report should not be promoted as a clean Phase 2
pass until the evidence is regenerated/annotated and the missing compatibility
checks are added.

## Findings

### 1. Shape/layout and output compatibility — conditional pass

- `quantize_query()` and `quantize_decode_q_to_padded_fp4()` now allocate and
  validate FP4 Q as `[rows, 1, heads_q, head_dim/2]`; the kernel launch passes
  `(rows, 1, heads_q, 128)`, and the decode output defaults to
  `[rows, 1, heads_q, 128]` before returning `[rows, heads_q, 128]`.
- `seqlen_q_static_one=True` is applied consistently to output and LSE
  predicates. The checked-in exact pre-quantized-vs-BF16 tests cover MHA, GQA,
  and MQA and passed.
- The residual path deliberately retains a separate BF16 padded Q buffer with
  `[rows, 128, heads_q, head_dim]`; this is compatible with the changed FP4 Q
  layout and is tested through the existing residual cases.
- The public output/scatter contract remains `[output_rows, heads_q, 128]`, and
  the existing scatter test passed.

However, there is no explicit assertion of the returned compact output shape,
no direct test that LSE (if enabled internally) is written only for the true
query length, and no independent test of the raw pre-quantized `[rows, 1, ...]`
layout against a hand-built tensor outside the quantizer. These are coverage
holes rather than a demonstrated code defect.

### 2. MHA regression — must be treated as a release blocker for a broad pass

The clean before/after JSON comparison does show the claimed MHA behavior:
Phase 2 FP4-Q geomean is **0.887x** versus the Phase 1 padded-Q path, with
individual speedups from **0.686x to 1.014x**. Thus MHA regresses materially at
several points, especially batch 128 / seqlen 1024 (0.686x), while GQA and MQA
improve.

The report records this fact, which is good, but it still describes the overall
Phase 2 result as meeting the expected mechanism without defining an acceptance
rule for the MHA regression. A Phase 2 pass should either (a) explicitly mark
MHA unsupported/non-goal with an owner and follow-up gate, or (b) add a
regression threshold and resolve/waive it. Current evidence is insufficient for
an unqualified performance pass.

### 3. Performance/provenance — evidence is not clean

The checked-in artifacts do not provide a valid same-commit before/after
comparison:

- `baseline-padded.json` was recorded at commit `5f83434`, clean.
- `after-unpadded.json` records commit `83d47ea`, **dirty**, even though the
  reviewed change is `f24cfe0`, which is the current HEAD.
- The after artifact timestamp is `2026-07-29T15:26:34Z`, while `f24cfe0` was
  committed at `2026-07-29T23:29:21+08:00` (`15:29:21Z`), so the artifact
  predates the reviewed commit by about 2 minutes 47 seconds.

The JSON therefore cannot be used as provenance for the exact reviewed commit.
The benchmark arguments and environment are otherwise recorded well (30
iterations, 10 warmups, GPU/package versions, complete requested 3x3x3 grid),
but the dirty-worktree flag and commit mismatch are decisive. Regenerate both
or at minimum regenerate the after run at `f24cfe0` with a clean worktree and
record the exact command/output.

Also, the report's D0 comparison is not independently auditable from the Phase
2 directory: the report gives ratios but does not identify the exact D0 artifact
and commit for every comparison. The baseline definition is present in JSON,
but the report should link/name the source artifact explicitly.

### 4. Test coverage and MHA regression coverage — incomplete

`test-results.txt` documents 56 passing tests, including exact MHA/GQA/MQA
pre-quantized query equality, residual/index/scatter cases, and quantizer
checks. This supports basic numerical correctness.

Gaps relevant to this commit:

- No explicit API test asserts `fp4_decode(...)` returns exactly
  `(rows, heads_q, HEAD_DIM)` after the internal 4-D output change.
- No test exercises a caller-supplied raw pre-quantized tensor with shape
  `[rows, 1, heads_q, 64]` constructed independently of `quantize_query()`.
- No dedicated output/LSE boundary test proves that the removed 127 query
  positions are not written for MHA, GQA, and MQA.
- The benchmark grid omits batch sizes 1, 2, 4, and 8 and seqlen 65536 from the
  Phase 0 required grid; the artifact itself says
  `phase0_required_grid_complete: false`. That is acceptable as a targeted
  high-batch experiment, but not as complete regression coverage.
- The test log contains 3,755 warnings. They are mostly dependency/API
  deprecations, not a Phase 2 failure, but the report should distinguish them
  from a clean warning-free gate.

### 5. IKET conclusion — useful but overclaimed

The IKET artifact is internally consistent for the stated case (batch 64,
seqlen 16384, GQA-4): 148 traced CTAs, all 148 SMs, one CTA/SM, no malformed
ranges, and softmax0 reaches the launch tail. The listed wait ranges support
identifying producer/consumer synchronization as the immediate structural
issue.

The conclusion that the next lever is intra-CTA softmax/correction/P readiness
is a reasonable hypothesis and is appropriately stronger than a DRAM claim;
the report correctly says DRAM byte quantification still needs NCU.
Nevertheless, this is one structural-only trace from a single case, with no
corresponding before/after IKET comparison and no timing attribution proving
that the query-padding change caused or fixed the waits. Phrase it as a
single-case diagnosis/hypothesis, not as confirmation of a general Phase 2
prerequisite.

## Required fixes before PASS

1. Regenerate `after-unpadded.json` from clean `f24cfe0` (and preferably rerun
   the padded baseline under a recorded clean commit/environment), then update
   the report's provenance references.
2. Add explicit output-shape and raw `[rows, 1, heads_q, 64]` pre-quantized
   layout tests, including MHA/GQA/MQA and a true-length output/LSE boundary
   check where feasible.
3. Define and document the MHA regression policy (threshold, waiver, or
   non-goal) and retain the observed 0.686x worst point rather than presenting
   Phase 2 as an unqualified speed pass.
4. Identify the exact D0 source artifact/commit and label the IKET conclusion as
   single-case structural evidence; do not infer DRAM behavior from IKET.

## Positive evidence retained

- The implementation changes the FP4 Q extent and static query-length
  predicates consistently.
- Exact FP4-Q versus BF16-Q equality is covered for MHA, GQA, and MQA.
- Existing residual, indexed-query, vLLM-page, and output-scatter tests passed.
- The high-batch grid is complete for its stated requested points, and the
  report transparently records the MHA regression and IKET/NCU limitation.
