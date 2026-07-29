# Phase 1 Plan — Single-entry pre-quantized FP4-Q contract

## Goal Description

Implement D2 exactly: keep `fp4_decode` as one public entry, support either a BF16 `query` or optional pre-quantized `query_fp4 + query_scales`, and route both paths into the same compiled decode core. The FP4-Q path must not quantize or allocate padded BF16 scratch per call. This Phase is the unique stop-on-failure Phase.

Per the user's execution-order decision, the new FP4-Q path in this Phase targets
pure FP4 K/V with page-aligned lengths (`len % 128 == 0`) and no BF16 residual.
FP4-Q + residual is a post-Phase-5 compatibility Phase after the speed target is met.
Existing BF16-Q residual behavior remains protected by the unchanged test suite.

Expected effect before implementation: Phase 0's diagnostic composition attributes 0.00611 ms (11.15%) to Q quantization for a short case, so clean short-case GPU latency is expected to improve about 5–15%. Correctness expectation is exact byte equality of outputs when FP4-Q inputs are produced by the repository's existing query quantizer.

## Acceptance Criteria

- **AC-1: Public API remains a single entry with mutually exclusive query contracts.**
  - Positive: BF16 `query` alone follows the existing path; `query_fp4` and `query_scales` together follow the new path.
  - Negative: neither contract, both contracts, or only one FP4-Q tensor raises a clear validation error.
- **AC-2: Both paths share the decode core and preserve all existing semantics.**
  - Positive: both call `decode_fp4`; residual, `query_row_indices`, `out`, and `out_indices` behavior remains unchanged for BF16-Q.
  - Negative: no duplicate core kernel, no K/V quantization inside `fp4_decode`, no signature replacement.
- **AC-3: FP4-Q layout validation matches the quantizer-native contract.**
  - Positive: packed E2M1 shape/dtype/device/contiguity and E4M3 scale layout from `quantize_query` are accepted for MHA/GQA/MQA.
  - Negative: wrong dtype, shape, device, strides, row/head count, or partial arguments are rejected before launch.
- **AC-4: FP4-Q numerical gate is added and exact for the pure page-aligned path.**
  - Positive: quantize a BF16 query once, feed both paths, and assert output byte equality across representative MHA/GQA/MQA pure-FP4 cases with page-aligned lengths; `tests/kernel` includes these tests.
  - Negative: no tolerance relaxation or replacement of existing cosine/oracle gates.
- **AC-5: The performance harness records both paths separately.**
  - Positive: `fp4_pure_fp4q` becomes available, D4 gate path is FP4-Q versus D0 better FA4, and BF16-Q remains separately reported.
  - Negative: interface-change benefit is not mixed invisibly into later kernel gains.
- **AC-6: Phase closes green or the whole task stops.**
  - Positive: full `tests/kernel` passes, clean representative timings are recorded, internal Terra/high primary and different-model cross-review pass, and a green Phase 1 commit exists.
  - Negative: any unresolved Phase 1 failure stops for user input; no later Phase begins.

## Path Boundaries

### Upper Bound

- `interface.py`, `_kernel.py`, `_decode.py` validation/adaptation needed for the optional input contract; quantizer facade reuse; kernel tests; benchmark variants/results; Phase 1 evidence/docs.

### Lower Bound

- A real pre-quantized Q path with no per-call Q quantization/scratch, exact output equality, tests, benchmark separation, and green close commit.

### Allowed Choices

- Can use quantizer-native `query_fp4` and `query_scales`, and reject FP4-Q +
  residual until the post-speedup compatibility Phase. The pure FP4-Q path must
  not allocate padded BF16 scratch per call.
- Cannot change D2, add a second public function, modify numerical gates, alter K/V contracts, add experimental environment switches, or proceed after Phase failure.

## Dependencies and Sequence

1. Map the existing quantizer/decode tensor layouts and validation invariants.
2. Extend the single public signature and enforce mutually exclusive contracts.
3. Refactor composition so BF16-Q quantizes then calls the same core while FP4-Q calls it directly.
4. Add exact pure-FP4 FP4-Q tests across head configurations and page-aligned lengths; keep existing BF16-Q residual regressions green.
5. Enable separate BF16-Q/FP4-Q benchmark variants and record clean speedup.
6. Run full numerical gate, internal dual-model review, and green close commit.

## Implementation Notes

- Every shell starts with the mandated §6.1 environment lines.
- Primary review is internal GPT-5.6-Terra/high; cross-review uses a different internal model. Never invoke external `codex exec` or `codex review`.
- AC terminology must not enter production comments.
- Any need to alter D0–D8 requires stopping for the user.
