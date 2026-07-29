# Phase 2 alternate signoff — NEEDS_FIX

**Reviewed target:** `b29f11d5898e33bb196d652431c5ed506ef273c2`
(`Close Phase 2 evidence gaps`)

**Review mode:** internal, read-only review of the committed implementation and
Phase 2 evidence. No source, benchmark, or test edits were made. This file is
the only signoff artifact added by this review.

## Verdict

**NEEDS_FIX.** The true-query-length implementation is coherent and the new
clean artifact is useful, but the Phase 2 close is not yet cleanly attributable
to the final reviewed commit, and the §6.6 failure-record policy has not been
satisfied strongly enough to support a PASS.

## Evidence that passes

- `after-clean.json` is a clean, complete artifact for its requested grid:
  batches `{1,4,16,64,128}`, seqlens `{1024,4096,16384}`, and MHA/GQA/MQA.
  It includes both BF16-Q and FP4-Q measurements, with 10 warmups and 30
  measured iterations.
- `after-clean.json` records `dirty=false` and the expected D0 definition:
  the better of FA4 split=1 and the FA4 heuristic.
- `test-results-clean.txt` records **56 passed**. The existing exact
  FP4-Q/BF16-Q equality coverage for MHA, GQA, and MQA, plus residual,
  index, and scatter coverage, remains positive evidence.
- The implementation audit and prior reviews support the shape/layout change:
  FP4-Q uses the true query extent, static output/LSE predicates use
  `seqlen_q_static_one=True`, and native query-scale packing is retained.
- The IKET result is a valid single-case structural trace. Its stated
  limitation that it does not quantify DRAM bytes is correct.

## Blocking findings

### 1. Clean evidence is not attributable to the final reviewed commit

The final reviewed commit is `b29f11d`, but:

- `after-clean.json` provenance records commit `763d452` and `dirty=false`.
- `test-results-clean.txt` has no revision or dirty-state header; the report
  asserts that it is from the same clean revision, but the transcript itself
  does not prove that.
- `baseline-padded.json` is clean at `5f83434`, while the older
  `after-unpadded.json` is dirty at `83d47ea`; that older comparison is not
  acceptable close evidence.

`b29f11d` appears to add only evidence/report files after `763d452`, so this is
not evidence of a source regression. It is nevertheless a provenance gap:
the signoff must either regenerate the final artifacts at
`b29f11d` with `dirty=false`, or explicitly record and verify that the source
tree under test is byte-identical and that the artifact is intentionally
carried forward. The test transcript should include the exact commit, branch,
dirty state, command, and result.

### 2. AC-3 is not fully closed by the available before/after evidence

The clean high-batch FP4-Q comparison reports:

- MHA: **0.89x** geometric speedup, with a worst point around **0.69x**.
- GQA: **1.35x** geometric speedup.
- MQA: **2.89x** geometric speedup.

The MHA regression is honestly reported, but AC-3 also requires low-batch and
BF16-Q non-regression evidence. `after-clean.json` contains those measurements;
the checked-in padded baseline does not contain the corresponding low-batch or
BF16-Q before values. Therefore those results cannot establish a clean
before/after non-regression gate.

The report's statement that the MHA result is retained as a failed/negative
branch is directionally useful, but it does not define a numerical bound,
owner, or explicit human waiver. Narrowing the acceptance wording to GQA/MQA
is not, by itself, permission to relax AC-3 or §6.6 item 5.

### 3. §6.6 failure-record policy is only partially met

§6.6 item 4 requires that when a gate is not beaten, the last green state be
committed and the failure reason plus excluded branches be written in the log,
without relaxing the gate. The report does record two negative branches and
the MHA regression, which is positive. However, the close record does not yet
provide all of the required audit detail:

- the exact failed measurement set and threshold/gate being considered;
- whether the MHA result is a failure of the broad AC-3 gate, an explicit
  non-goal, or a human-approved exception;
- owner/follow-up phase for the MHA regression;
- a complete list of the unavailable low-batch/BF16-Q comparison evidence;
- the exact final green commit and test provenance.

Until those points are recorded without silently weakening the gate, the
failure-record policy cannot be treated as closed.

### 4. IKET evidence is not before/after evidence

The committed IKET artifact is one high-batch trace and has no comparable
before trace, case-command metadata, or revision/dirty-state provenance in the
artifact itself. It can support a **single-case structural hypothesis**, not
AC-4's requested before/after evidence or a causal claim that Phase 2 changed
the waits. No DRAM conclusion should be inferred from it; an ncu artifact is
needed if the traffic question remains material.

## Required disposition before PASS

1. Produce or explicitly validate final-commit-attributed clean performance and
   test artifacts for `b29f11d`, including command, environment, commit, branch,
   and dirty state.
2. Add the missing low-batch and BF16-Q before/after comparison, or state an
   explicit human-approved exception with numerical bounds.
3. Expand the §6.6 failure record with the MHA gate decision, measured failed
   cells, owner/follow-up, excluded branches, and the final green commit.
4. Add comparable attributed IKET before/after evidence, or document the
   approved reason AC-4 is being deferred; use ncu for byte quantities if they
   remain material.

**Final disposition: NEEDS_FIX.**
