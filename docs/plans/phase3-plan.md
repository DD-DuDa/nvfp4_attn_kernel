# Phase 3 Plan — Low-batch split-K and combine

## Goal Description

Add pure FP4 split-K decode for page-aligned FP4-Q inputs: split the KV block range across CTAs, write FP32 partial O and LSE, combine them in a private standalone CuTe kernel, and select splits only where low-batch occupancy gains exceed launch/combine overhead.

## Expected Benefit and Falsifiable Basis

For decode with one query token, the unsplit grid has approximately
`batch * kv_heads` CTAs. At batch 1 this is only 1 CTA for MQA, 8 for the
gate GQA shape, and 16 for the MHA test shape, far below the 148 available
SMs. A split count near `ceil(148 / (batch * kv_heads))`, capped by the number
of 128-token KV blocks and by measured combine overhead, should raise SM
coverage by roughly the split count. The expected Phase-level effect is a
large improvement for batch `{1, 4}` at long sequence lengths and fallback to
unsplit at high batch. This is a hypothesis, not a replacement for §8.1.

## Acceptance Criteria

- AC-1: Split scheduler produces valid split indices and disjoint, exhaustive
  KV ranges; partial O/LSE shapes, dtypes, strides, and mathematical meanings
  match the combine contract.
  - Positive tests (expected to pass):
    - For split counts `{2, 4, 8}` and page-aligned lengths, recorded block
      ranges are pairwise disjoint and their union equals every KV block.
    - Each non-empty partial agrees with an independently launched unsplit
      kernel over that split's page slice; empty partial LSE is exactly `-inf`.
    - A PyTorch FP32 oracle combining locally normalized partial O with
      `softmax(partial_lse, dim=split)` agrees with the unsplit output.
  - Negative tests (expected to fail):
    - Deliberately transposing the query-head and sequence axes, treating
      natural-log LSE as log2 LSE, or combining partial O without LSE weights
      must be detected by the diagnostic.
- AC-2: Combined output meets unchanged numerical gates against unsplit FP4
  and FA4 for MHA/GQA/MQA and multiple useful split counts.
  - Positive tests (expected to pass):
    - Split `{2, 4, 8}` cases satisfy cosine `>= 0.99` and max absolute error
      `<= 5e-2`, without changing either threshold.
    - MHA `(Hq=Hkv)`, GQA, and MQA `(Hkv=1)` page-aligned cases pass.
  - Negative tests (expected to fail):
    - Invalid split counts and split use on the deferred FP4-Q residual path
      are rejected or routed to the existing unsplit contract.
- AC-3: Low batch `{1,4}` ratios versus D0 improve significantly, with IKET
  showing more SMs used and no pathological CTA lifetime imbalance.
  - Positive tests (expected to pass):
    - Clean benchmarks show a material low-batch improvement versus the Phase
      2/4 unsplit state for at least the long-context rows where occupancy is
      the registered bottleneck.
    - IKET records increased `sms_used` and balanced split CTA lifetimes.
  - Negative tests (expected to fail):
    - A split setting whose main-kernel plus combine latency is not lower than
      unsplit is not selected by the production heuristic.
- AC-4: High batch does not regress because split selection falls back to
  unsplit where appropriate; the Phase 2 MHA negative branch remains visible.
  - Positive tests (expected to pass):
    - Full-grid clean benchmarking shows no high-batch regression attributable
      to the split heuristic.
  - Negative tests (expected to fail):
    - Forcing split-K on a high-batch shape may be benchmarked diagnostically
      but cannot be selected if it regresses the unsplit path.
- AC-5: No residual FP4-Q support is required yet; existing BF16-Q residual
  tests remain green.
  - Positive tests (expected to pass):
    - Existing `tests/kernel` residual, zero-length residual, and scatter
      contracts remain unchanged and green.
  - Negative tests (expected to fail):
    - The pure FP4 split entry rejects residual arguments rather than silently
      double-counting them.
- AC-6: Phase closes in a green committed state with a failure/config ledger,
  clean performance evidence, and two internal-agent reviews: GPT-5.6-Terra
  High plus a different model.
  - Positive tests (expected to pass):
    - `PYTHONPATH=src python -m pytest -q tests/kernel` passes at the closing
      commit.
    - Review findings and fixes are recorded in `docs/perf/phase3/`.
  - Negative tests (expected to fail):
    - No `codex exec`, `codex review`, `ask-codex`, or other external-review
      connection is used; any workflow that requires it is not launched.

## Path Boundaries

### Upper Bound

Pure page-aligned FP4-Q split-K, standalone private combine, production split
selection, numerical tests, clean benchmarks, and IKET evidence. FP4-Q plus
BF16 residual remains Phase 6 work after the §8.1 speed gate.

### Lower Bound

A numerically correct fixed-split pure FP4 path with a PyTorch oracle that
unambiguously proves both partial semantics and combine semantics. It is not
enough for Phase closure unless the production heuristic and performance
evidence also satisfy AC-3/AC-4.

### Allowed Choices

- May change `_decode.py`, `fp4_decode_kernel.py`, a standalone private combine
  module, tests, profiling scripts, and Phase 3 documentation.
- May touch `_fa4/` only for decode-required support, with the reason recorded.
- Must not change D0–D8, §8 gates, the single-entry FP4-Q interface, or the FA4
  baseline definition.
- Must not add residual support before the speed gate, rely on experimental
  environment flags, or use external Codex review connections.

## Dependencies and Sequence

1. Milestone 1 — isolate partial semantics.
   - Restore the fixed-split producer without exposing it publicly.
   - Export FP32 partial O/LSE.
   - Compare every split against page-sliced unsplit launches.
   - Use the PyTorch combine oracle to decide whether the producer or CuTe
     combine owns each numerical failure.
2. Milestone 2 — make fixed split numerically correct.
   - Repair producer boundaries/partial semantics if the oracle fails.
   - Repair combine axes/layout/math if the oracle passes.
   - Gate split `{2,4,8}` over MHA/GQA/MQA.
3. Milestone 3 — production selection and performance.
   - Benchmark candidate split counts at low and high batch.
   - Derive and freeze a deterministic heuristic.
   - Capture IKET SM-spread/lifetime evidence and clean full-grid timings.
4. Milestone 4 — internal review and green close.
   - Run GPT-5.6-Terra High and a different internal model as independent
     reviewers.
   - Resolve findings, run all kernel tests, update the failure/config ledger,
     and commit the green Phase 3 state.

## Implementation Notes

- Per-split O is expected to be locally normalized. If `LSE_s` is the natural
  log-sum-exp for split `s`, the final result is
  `sum_s(exp(LSE_s - logsumexp(LSE)) * O_s)`.
- Empty splits must have `LSE_s = -inf`; their O contents are ignored.
- Diagnostic helpers may exist during the loop but must not become an
  undocumented production dump or environment-switch path.
