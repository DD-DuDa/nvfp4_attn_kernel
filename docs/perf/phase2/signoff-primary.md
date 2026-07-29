# Phase 2 primary signoff — COMPLETE WITH RECORDED NEGATIVE BRANCH

**Reviewer lane:** internal GPT-5.6-Terra/high primary  
**Review mode:** read-only source/evidence audit; no production-code, test, or
benchmark edits; no commands rerun for this signoff.  
**Reviewed lineage:**

- `f24cfe0407729016b4d6ac08676d8f5758a42fef` — implementation: true FP4-Q
  decode length.
- `763d4520d8cdd26b0afb231e97a3086fc6b29d0b` — clean post-review
  performance/test evidence.
- `b29f11d5898e33bb196d652431c5ed506ef273c2` — final Phase 2 evidence-close
  commit.

`b29f11d` changes only Phase 2 evidence/report files relative to
`763d452`; the source and tests are identical. Therefore the clean evidence
at `763d452` applies to the implementation reviewed at final `b29f11d`.

## Disposition

**Phase 2 is complete under §6.6 item 4 as a bounded, partial result.**
Retain the measured grouped-query gains and proceed to the next independently
gated Phase. This is **not** an all-head-configuration performance PASS:
high-batch MHA regresses and is explicitly recorded as a negative branch, not
waived. No later phase may claim a full-table non-regression result without
retesting MHA against the Phase 1 padded-Q path.

The exact blocker to an *unqualified* Phase 2 performance PASS is:

> High-batch MHA `8:8` FP4-Q is **0.911x** geometric mean versus the clean
> padded-Q baseline, with a worst point of **0.692x** at
> `batch=128, seqlen=1024` (0.182790 ms padded versus 0.264123 ms unpadded).

Under §6.6 item 4, that failed gate is logged rather than weakened. It does
not erase the independent GQA/MQA result or justify calling the MHA regression
acceptable.

## Evidence audited

| Area | Result | Basis |
|---|---|---|
| True-length FP4-Q chain | PASS | `f24cfe0` changes FP4-Q allocation/validation from `[rows, 128, heads_q, 64]` to `[rows, 1, heads_q, 64]`, creates the shared core with `seqlen_q_static_one=True`, and keeps the separate padded BF16 residual buffer. |
| Native scale layout | PASS | Query scales remain quantizer-native strided storage; source audit found no per-call transpose, materialized copy, cache, or production switch. |
| Numerical/regression gate | PASS | `test-results-clean.txt` records **56 passed** at clean `763d452`. The recorded suite includes exact pre-quantized FP4-Q versus BF16-Q checks for MHA `8:8`, GQA `32:8`, and MQA `32:1`, plus retained BF16 residual, zero-length, indexed, and scatter coverage. |
| Clean final performance artifact | PASS, with bounded interpretation | `after-clean.json` records `commit=763d452...`, `dirty=false`, 10 warmups, 30 measured iterations, L20A/device/package provenance, batches `{1,4,16,64,128}`, target seqlens `{1024,4096,16384}`, MHA/GQA/MQA, and BF16-Q/FP4-Q variants. |
| High-batch Phase 1 comparison | PARTIAL PASS | Matching clean padded baseline `baseline-padded.json` (`5f83434`, clean) to clean unpadded FP4-Q data for batches `{16,64,128}` retains GQA/MQA gains but exposes the MHA regression below. |
| IKET | LIMITED / HYPOTHESIS ONLY | The one recorded case is batch 64, seqlen 16384, GQA-4, FP4-Q. It has a 148-CTA persistent grid across all 148 SMs and softmax0 at the launch tail. It is not a before/after proof and makes no DRAM-byte claim. |

## Clean high-batch result versus padded-Q baseline

The following values are recomputed from the matching `fp4_pure_fp4q` rows in
the clean artifacts, using `baseline-padded.json` divided by
`after-clean.json`. They supersede the earlier dirty-run summary values for
this signoff.

| Head configuration | Geometric speedup | Point range | Signoff treatment |
|---|---:|---:|---|
| MHA `8:8` | **0.911x** | 0.692–1.148x | Negative branch; not waived. |
| GQA `32:8` | **1.432x** | 0.967–2.118x | Retain gain. |
| MQA `32:1` | **2.844x** | 0.955–10.337x | Retain gain. |

The observed pattern is consistent with the scoped mechanism: removing
artificial Q rows recovers grouped-query work, while MHA has no grouped-query
packing redundancy to recover. That explanation is a mechanism hypothesis, not
a waiver of the measured MHA loss.

## IKET limitation and next-step constraint

`iket-high-batch.json`/`.txt` is valid structural evidence for its one
declared GQA-4 case: 148 traced CTAs use 148 SMs with at most one CTA per SM;
softmax0 reaches the launch tail; the dominant recorded waits include
correction waiting on softmax (89.8%), load waiting for KV-buffer release
(88.4%), MMA waiting for P/O readiness (79.4%), and epilogue waiting for
correction (99.6%).

Its only allowed conclusion is a **single-case Phase 2b hypothesis**: the
next high-batch lever is likely intra-CTA softmax/correction/P readiness, not
additional CTA count. It does not establish a general before/after wait
reduction, nor can it quantify L2/DRAM traffic. Any later traffic claim
requires a separately recorded NCU measurement.

## Required carry-forward record

1. Keep the MHA `0.692x` worst point and `0.911x` high-batch geomean visible
   in later full-table performance reviews; do not relabel it as immaterial.
2. Preserve GQA and MQA as the successful, attributable Phase 2 result.
3. Treat the IKET result as a single-case diagnostic hypothesis only. Phase 2b
   must validate its own before/after structural and numerical/performance
   gates.
4. Final program-level performance closure remains Phase 5's clean full-table
   gate; this signoff neither changes D0 nor relaxes any numerical or
   performance threshold.

**Primary conclusion:** close Phase 2 in the §6.6 item 4 sense—committed green
state, documented failure branch, and retained independent gains—while carrying
the unwaived MHA regression forward as the explicit blocker to any broad
Phase 2 performance-PASS claim.
