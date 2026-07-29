# Phase 2 Primary RLCR Review

**Reviewer lane:** internal GPT-5.6-Terra/high primary review  
**Review target:** `f24cfe0407729016b4d6ac08676d8f5758a42fef` — *Use true decode query length in FP4 core*  
**Scope:** `docs/plans/phase2-plan.md`; Phase 2/2b in
`docs/tasks/2.fp4_decode_speedup.md`; D0–D8; the approved FP4-Q + BF16
residual deferral; and the committed Phase 2 artifacts.  
**Method:** source and artifact audit only. The reviewer ran **no tests,
benchmarks, IKET, or ncu**.

## Verdict

**NOT COMPLETE — Phase 2 cannot close.**

The implementation change is directionally correct and the recorded numerical
transcript is encouraging, but Phase 2's required clean, target-commit
performance/profiling provenance is absent. In addition, high-batch MHA has
material recorded regressions and neither the required low-batch nor BF16-Q
performance-regression evidence is present. A separate alternate-model review
is also still required by AC-6 after the evidence is repaired.

## Findings

| ID | Severity | Finding | Required disposition |
|---|---|---|---|
| P1 | blocker | `after-unpadded.json` is not attributable to `f24cfe0`: its provenance names parent `83d47ea40...` and `dirty: true`. The baseline is clean at `5f83434...`, but the comparison does not prove the committed target's result. | Rerun the exact high-batch clean grid at clean `f24cfe0` and replace/add a final artifact whose provenance records `commit=f24cfe0...`, `dirty=false`, 10 warmups, 30 measured iterations, device, package versions, and D0 baseline definition. |
| P1 | blocker | AC-3 is not met/evidenced. The recorded MHA geometric change is **0.887x** (an 11.3% geomean regression), with a worst case **0.686x** (31.4% regression). The review found no low-batch comparison and no BF16-Q comparison. The report acknowledges MHA regressions but neither supplies an approved materiality threshold nor a deferral. | Diagnose/fix the MHA regression, or obtain an explicit user-approved Phase-2 exception with a numerical bound. Add clean low-batch and BF16-Q before/after regression evidence; do not relabel the high-batch-only grid as satisfying the full AC-3 regression requirement. |
| P1 | blocker | AC-4 asks for high-batch IKET evidence **before and after**. `docs/perf/phase2/` contains only one high-batch trace, and the trace JSON/TXT has no case command, input-shape declaration, git revision, or dirty-state provenance. There is no Phase-2 ncu byte-count artifact. | Capture and preserve comparable padded and unpadded IKET traces for the declared high-batch case, each with clean revision/case metadata. If, after comparison, the traffic question remains material, run ncu and record DRAM/L2 byte counts; otherwise record why the structural trace resolves the question. |
| P1 | blocker | AC-6's dual-model closure is incomplete: this is the required Terra/high primary review, but no Phase-2 alternate-model/cross-review record exists. The 56-pass transcript itself has no revision/dirty-state header. | After repairing evidence, obtain an independent alternate-model review of exact clean HEAD and add its report. Preserve a clean final `tests/kernel` transcript attributable to that same revision. |

## Implementation Audit

### True decode Q length — PASS

The changed BF16-Q quantizer allocation is `[rows, 1, heads, head_dim // 2]`
in `src/nvfp4_decode_kernel/_quantize.py`. Its CuTe kernel writes only
`query_fp4[row, 0, head, ...]`. The decode boundary now validates
`query_fp4` as `[rows, 1, heads_q, 64]`, and `_compile_decode` constructs the
shared core with `seqlen_q_static_one=True`. This is the expected chain from
public FP4-Q representation through the static core predicate; it is not a
cosmetic shape-message-only change.

The remaining 128-row BF16 allocation is `query_padded_bf16`, guarded by the
existing residual path. It is not part of pure FP4-Q transport and remains
necessary for the preserved BF16-Q residual semantics. Therefore it does not
contradict AC-1 or the approved residual ordering.

### Native query-scale layout — PASS

The scale allocation remains a native storage allocation followed by
`permute(...).permute(...).as_strided(...)`, without a per-call transpose,
materialized repack, cache, or production switch. The decode boundary requires
the exact seven-axis shape and strides returned by `quantize_query`; the
quantizer writes directly to that layout. This meets AC-5 on source audit.

### Exact numerical coverage and residual deferral — PASS AS RECORDED

`tests/kernel/test_fp4_decode_correctness.py` parametrizes exact FP4-Q versus
BF16-Q equality for all three required head forms:

* MHA: `8:8`
* GQA: `32:8`
* MQA: `32:1`

The test compares outputs with `torch.equal`, not a relaxed tolerance.
`docs/perf/phase2/test-results.txt` ends with **`56 passed`** and includes the
MHA/GQA/MQA prequantized-query cases plus the retained BF16-Q residual,
zero-length residual, index, and scatter cases. No test was rerun in this
review.

The user-approved ordering is preserved: pure page-aligned FP4-Q/K/V is the
Phase 1–5 performance contract, while FP4-Q plus BF16 residual remains
deferred; BF16-Q residual behavior remains a regression gate. This deferral is
not a finding. Its evidence still needs final target-commit provenance before
AC-6 can pass.

## Performance and D0 Interpretation

The performance artifacts use the required D0 definition: the minimum
CUDA-event latency of FA4 split=1 and FA4 heuristic, with varlen FA4. They
separately label `fp4_pure_fp4q`, use batches 16/64/128, seqlens
1024/4096/16384, and head configurations 8:8, 32:8, and 32:1. Both artifacts
state 10 warmups and 30 measured iterations.

Recomputing the FP4-Q padded-to-unpadded changes from the committed JSON values
gives:

| Configuration | Geomean padded / unpadded | Range | Interpretation |
|---|---:|---:|---|
| MHA 8:8 | 0.887x | 0.686–1.014x | Regression overall; no grouped-query packing benefit is expected, but the regression is not an approved non-regression exception. |
| GQA 32:8 | 1.351x | 0.983–1.735x | Partial improvement; below the plan's stated 1.5–4x expectation on geomean and includes a small regression. |
| MQA 32:1 | 2.888x | 0.967–10.502x | Strong improvement consistent with the grouped-query mechanism, but still includes one near-neutral/regressive cell. |

These figures support the report's qualitative mechanism explanation, but they
do **not** establish a clean performance result for `f24cfe0`, because the
after artifact explicitly records `dirty: true` at `83d47ea40...`.

The Phase 2 grid is not the D3 final full-grid gate, so this review does not
treat remaining ratios versus D0 as a Phase-2 failure by themselves. It does
reject any implication that this partial, dirty high-batch measurement closes
D3 or AC-3.

## IKET and Phase 2b

The recorded high-batch IKET artifact reports a 148-CTA persistent grid,
148 traced CTAs across 148 SMs, and at most one CTA per SM. `softmax0` reaches
the launch tail. The leading recorded structural waits are correction waiting
on softmax (89.8%), load waiting for KV release (88.4%), MMA waiting for P
(79.4%), and epilogue waiting for correction (99.6%).

On its technical content, this supports the Phase 2b prerequisite: a softmax/
correction-side dependency is on the launch-tail critical structure, rather
than an unspread grid. It also correctly avoids inferring DRAM bytes from IKET.
It does **not** prove a Phase-2 before/after reduction in waits, nor can its
unproven revision and undeclared workload be used as final closure evidence.
Phase 2b may be investigated once the trace provenance is repaired; its own
same-round numerical gate remains mandatory.

## D0–D8 Ledger

| Decision | Status | Audit result |
|---|---|---|
| D0 — better FA4 baseline | PASS IN METHOD; provenance blocker | Both JSON artifacts state the required better-of split=1/heuristic CUDA-event baseline, but final after data is dirty. |
| D1 — permanent IKET markers | PASS / unchanged | This change does not alter marker infrastructure; Phase-2 IKET output is non-empty. |
| D2 — single-entry dual query contract | PASS | FP4-Q still feeds the existing shared decode core; exact dual-path equality tests cover MHA/GQA/MQA. |
| D3 — final per-seqlen 2x gate | Not a Phase-2 close claim | No gate was relaxed. The recorded high-batch subset remains slower than D0, as expected before later phases. |
| D4 — FP4-Q gate path, BF16-Q separately recorded | PARTIAL | The FP4-Q path and correct FA4 baseline are used; a BF16-Q Phase-2 regression comparison is absent. |
| D5 — green committed phase close | BLOCKED | `f24cfe0` is committed and current tree was clean at review, but final test/performance evidence is not cleanly attributable to it. |
| D6 — bounded rounds and logged failures | PARTIAL | The report logs negative branches, but no Phase-2 RLCR round ledger is present in the committed evidence. |
| D7 — no unilateral relaxation | PASS | No numerical tolerance, D0 baseline, or D3 definition was relaxed. |
| D8 — IKET primary, ncu quantitative | PARTIAL | IKET is usable and structurally informative; before/after provenance and any needed ncu quantity evidence are missing. |

## Closure Checklist

Phase 2 may close only after all of the following are committed and reviewed:

1. A clean `f24cfe0` (or later committed fix) high-batch before/after comparison
   with complete provenance, preserving D0 and FP4-Q labels.
2. A disposition for the 0.887x MHA geomean / 0.686x worst regression, plus
   clean low-batch and BF16-Q regression measurements.
3. Comparable, fully attributed high-batch IKET before/after artifacts; ncu
   byte counts if the remaining question is traffic quantity.
4. A clean, revision-attributed 56-pass `tests/kernel` transcript.
5. Alternate-model cross-review of the exact final revision.

Until then the precise status is **BLOCKED FOR EVIDENCE AND REGRESSION
RESOLUTION**, not a rejection of the true-Q-length implementation.
