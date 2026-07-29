# Phase 1 Final Alternate Cross-Review

**Reviewed HEAD:** `1e97864` (`Record clean Phase 1 verification`)

**Scope:** internal static cross-check only. No tests or benchmarks were run and
no production code was edited. Reviewed the single-entry FP4-Q contract, native
query-scale layout, per-call repack removal, correctness tests, benchmark
separation, provenance, and the explicitly deferred residual path.

## Verdict

# NEEDS_FIX

The implementation is substantially on the intended Phase 1 path: one public
`fp4_decode` entry supports mutually exclusive BF16-Q and pre-quantized FP4-Q
contracts; both converge on the same `decode_fp4` core; FP4-Q bypasses query
quantization and padded BF16 scratch; and the former host-side scale repack was
removed. The approved FP4-Q + BF16-residual deferral is also respected.

However, the final review should not close as `PASS` because the committed test
evidence does not fully cover the Phase 1 AC-3 negative contract matrix. The
runtime validator is broader than the tests, but the required rejection cases
are not all directly gated. This is a Phase 1 acceptance gap, not a numerical
correctness finding.

## Findings

### F-1 — HIGH: native-layout negative test coverage is still incomplete

`src/nvfp4_decode_kernel/_decode.py` validates the important native contract:
packed E2M1 query shape/dtype/contiguity, query-scale shape, strides, device,
and storage offset, plus head divisibility and row consistency. The final
correctness test adds exact MHA `(8, 8)`, GQA `(32, 8)`, and MQA `(32, 1)`
FP4-Q/BF16-Q equality checks and several contract failures.

But `tests/kernel/test_fp4_decode_correctness.py` does not independently exercise
the full plan-required negative matrix. In particular, there is no focused test
for wrong query device, wrong query-scale dtype, wrong scale strides/storage
layout beyond the single `.contiguous()` case, or independently mismatched
row/head metadata. The existing shape failures cover some of these indirectly,
but they do not establish each stated rejection contract.

**Required disposition:** add focused pre-launch negative tests for the missing
query device, query-scale dtype/layout, row-count, and head-count cases. Keep the
existing partial, ambiguous, indexed-FP4-Q, dtype, shape, and exact-equality tests.

### F-2 — MEDIUM: final timing provenance is traceable but not HEAD-exact

`docs/perf/phase1/short-case-final.json` records a clean benchmark at
`19edfbd424cb7d633e597b9beb66401fdcd90364` with `dirty=false`, while the
reviewed HEAD is `1e97864`. The intervening HEAD commit is documentation-only,
so the measured implementation is traceable and the result is not inherently
invalid. Nevertheless, the artifact is not a benchmark run from the exact
reviewed HEAD.

For a strict clean-HEAD close, either rerun the short case at `1e97864` and
record that commit, or explicitly retain the current measured-implementation
provenance as an accepted exception in the Phase report. No source-code drift
is visible between `19edfbd` and `1e97864`.

## Areas that PASS

| Area | Result | Review conclusion |
|---|---|---|
| Single public entry | **PASS** | `interface.fp4_decode` remains the only public entry. BF16 `query` and keyword-only `query_fp4 + query_scales` are mutually exclusive; partial FP4-Q input is rejected. |
| FP4-Q correctness | **PASS for approved scope** | The repository quantizer produces the pre-quantized query and native scales; exact `torch.equal` tests cover MHA, GQA, and MQA on pure page-aligned FP4 K/V. |
| Shared core | **PASS** | BF16-Q quantizes then calls `decode_fp4`; FP4-Q calls the same function directly. The compile-cache key does not distinguish the query contract. |
| Native scale layout / no repack | **PASS** | `quantize_query(..., heads_kv=...)` writes the kernel-consumed PackGQA scale layout directly. `_pack_sfq_for_gqa`, `torch.take`, and `index_copy_` are absent from the current implementation. |
| No FP4-Q per-call Q quantization or padded BF16 scratch | **PASS** | The FP4-Q branch does not call `quantize_query()` and sets `query_padded_bf16=None`. |
| Residual deferral | **PASS** | FP4-Q with BF16 residual remains unsupported by design, consistent with the approved Phase 1 ordering. BF16-Q residual handling remains in the existing path and is covered by the recorded 56-test gate. |
| Benchmark separation | **PASS** | `fp4_pure_bf16q` and `fp4_pure_fp4q` are distinct variants; the final artifact labels them `bf16_q_fp4_kv` and `fp4_q_fp4_kv`, respectively, and compares both against the defined FA4 baseline. |
| Provenance quality | **PASS with F-2 caveat** | The artifact records environment, Python/package versions, GPU, arguments, branch, measured commit, and `dirty=false`; the only issue is that the measured commit is the implementation parent rather than reviewed HEAD. |

## Evidence cross-check

- `docs/perf/phase1/test-results.txt` records **56 passed** with no test run
  performed by this review.
- `short-case-final.json` records the representative case
  `batch=1, seqlen=1024, heads_q=32, heads_kv=8`, 10 warmups, 50 iterations,
  CUDA-event timings, and separate FP4-Q/BF16-Q variants.
- Recorded timings are `0.10903 ms` for FP4-Q and `0.32179 ms` for BF16-Q;
  the report's `2.95x` relative improvement is arithmetically consistent with
  those values.
- The final artifact's FP4-Q contract label is correct, unlike the earlier
  pre-fix artifact described by the prior cross-review.

## Close-out

Phase 1 is **NEEDS_FIX** until the missing AC-3 negative tests are added and the
provenance decision is made explicit: exact-HEAD rerun, or documented acceptance
that the timing measures the unchanged implementation commit `19edfbd` while
HEAD `1e97864` adds only evidence documentation.

The lack of FP4-Q + BF16 residual support is **not** a finding; it was explicitly
deferred by the approved Phase 1 execution order. The former per-call scale
repack finding is resolved in the current code.
