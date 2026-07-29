# Phase 1 Alternate-Model Cross-Review

**Reviewed revision:** `4c31c11` (`Add pre-quantized FP4 query decode path`)

**Review inputs:** `docs/plans/phase1-plan.md`, `docs/phase-drafts/phase1.md`,
commit `902a8e3` and its residual-order decision, the implementation/tests/benchmark
changes in `4c31c11`, and the committed `docs/perf/phase1/short-case.json` evidence.

**Scope:** internal cross-review only. No long tests were run and no production code
was edited.

## Verdict

# NEEDS_FIX

The core Phase 1 design is directionally correct and the explicit residual-order
decision is honored: FP4-Q is limited to pure FP4 K/V with page-aligned lengths,
while BF16-Q residual behavior remains available. However, the Phase cannot close
as green on the committed evidence because the performance artifact is not
provenance-valid for `4c31c11`, and the required negative validation gate is not
complete. These are Phase 1 acceptance failures, not merely follow-up polish.

## Findings

### F-1 — **BLOCKER: committed benchmark provenance does not identify the reviewed implementation**

`docs/perf/phase1/short-case.json` records:

- `git.commit = adf98adf...`, not `4c31c11`;
- `git.dirty = true`.

The file is nevertheless committed by `4c31c11` and is used by
`phase1-report.md` as the latest clean timing evidence. Therefore the 10-warmup /
50-iteration result cannot be attributed to the implementation under review. A
dirty-tree result may contain the Phase 1 changes, but it is not reproducible proof
of the committed Phase 1 implementation.

This fails the plan's AC-6 requirement for clean representative timings and makes
the reported FP4-Q speedup non-gating until rerun and recommitted with exact
revision provenance (or an explicitly documented, reproducible source revision).

### F-2 — **HIGH: AC-3 negative validation coverage is incomplete**

The implementation performs substantial runtime validation in `_decode.py`, but the
new contract test only covers:

- neither query contract;
- partial FP4-Q arguments;
- ambiguous BF16 + FP4-Q arguments;
- indexed FP4-Q rejection; and
- a materialized contiguous/wrong-scale-layout case.

The Phase 1 plan explicitly requires negative coverage for FP4-Q wrong dtype, shape,
device, strides, row/head count, and partial arguments. The added test does not
exercise wrong `query_fp4` dtype, wrong `query_fp4` shape, wrong device, wrong
`query_scales` dtype, wrong scale shape, wrong row count, or wrong head count.
Existing older tests do not substitute for these new query-contract cases.

This is a direct unmet acceptance criterion. Add focused pre-launch tests for each
listed rejection and retain the current positive MHA/GQA/MQA exact-equality tests.

### F-3 — **HIGH: benchmark result metadata labels FP4-Q with the wrong input contract**

In `tests/kernel_profile/bench_decode.py`, `run_case()` assigns
`"bf16_q_fp4_kv"` to every non-FA4 variant. Consequently `fp4_pure_fp4q` is
reported as BF16-Q/FP4-KV even though it supplies `query_fp4 + query_scales`.
This makes the JSON self-description incorrect and weakens provenance/comparison
against the separately reported BF16-Q path.

The metadata must distinguish at least:

- `bf16_q_fp4_kv` for `fp4_pure_bf16q`; and
- `fp4_q_fp4_kv` for `fp4_pure_fp4q`.

The committed short-case artifact should then be regenerated.

### F-4 — **MEDIUM: per-path memory evidence cannot support claims about hidden allocations**

`run_case()` resets peak memory once, runs all requested variants, reads one final
`torch.cuda.max_memory_allocated()` value, and writes that same value into every
row's `case_peak_memory_bytes`. Thus the artifact cannot distinguish BF16-Q from
FP4-Q allocation behavior.

The implementation does avoid the prohibited per-call padded BF16 scratch on the
pure FP4-Q path, and it does not call `quantize_query()` on that path. That part is
consistent with the Phase 1 contract. However, the benchmark does not prove it via
per-variant memory accounting. If the report discusses allocation removal, either
remove the unsupported claim or collect isolated per-variant memory deltas.

### F-5 — **MEDIUM: FP4-Q has hidden scale-repacking work and cache-growth allocations**

Every decode call invokes `_pack_sfq_for_gqa()`. On a cache miss it allocates the
packed scale storage, source/destination index tensors, and a gather buffer; on
subsequent calls it still performs `torch.take(..., out=...)` and
`index_copy_(...)` to repack query scale bytes. The cache key includes shape, so a
new row count or layout can grow the process cache and allocate again.

This is not the prohibited query quantization kernel and it is not padded BF16
scratch, so it is not by itself a violation of the residual-order decision. It is,
however, hidden FP4-Q data movement that should be called out in the performance
methodology. The short-case timing includes the steady-state repack copies, while
its report attributes the gain broadly to removing quantization/scratch without
separating this work. Future performance claims should either account for this
explicitly or provide a native scale layout that avoids the repack.

### F-6 — **MEDIUM: compile-cache sharing is implemented but not directly gated**

The code routes BF16-Q and FP4-Q through the same `decode_fp4()` core and uses a
compile-cache key based on device, head configuration, residual mode, and output
scatter mode—not on the query contract. This is the intended sharing behavior.
The same compiled kernel therefore appears to be reusable for both paths.

There is no exact test that calls both contracts in one process and asserts that the
compiled-cache entry count/object is shared. Add a lightweight cache-sharing test
(or an equivalent benchmark assertion) before relying on this as an acceptance
property. This is a testability gap rather than evidence of a current cache-key
bug.

## Criteria assessment

| Area | Assessment | Notes |
|---|---|---|
| Single public API | **PASS with ergonomics caveat** | Existing positional BF16 calls retain their ordering. New FP4-Q calls use keyword arguments naturally. Making all cache arguments default to `None` weakens the signature, but runtime missing-argument errors are clear. |
| Backward compatibility | **PASS** | BF16-Q path still quantizes internally; `query_row_indices`, residual, `out`, and `out_indices` remain on the BF16 path. Existing BF16-Q residual behavior was not intentionally removed. |
| Residual-order decision | **PASS** | Rejecting FP4-Q + BF16 residual in Phase 1 is explicitly approved by `902a8e3` and the Phase 1 plan. It is not a finding. |
| FP4-Q quantization/scratch | **PASS for the narrow prohibition; disclose overhead** | No `quantize_query()` or padded BF16 scratch is created on the pure FP4-Q route. Hidden scale repacking remains (F-5). |
| Validation implementation | **NEEDS_FIX** | Runtime checks are substantial, but required negative tests are incomplete (F-2). |
| Shared decode/compile path | **PASS in code; test gap** | Both contracts call `decode_fp4()`, and the cache key omits the query contract. Add a direct regression gate (F-6). |
| Exact numerical tests | **PASS for current scope, narrow coverage** | Exact `torch.equal` is used for MHA `(8,8)`, GQA `(32,8)`, and MQA `(32,1)` on pure page-aligned data. No FP4-Q residual test is required in this Phase. |
| Benchmark separation | **NEEDS_FIX** | BF16-Q and FP4-Q are separately timed, but FP4-Q's `input_contract` label is wrong and per-path memory evidence is not isolated (F-3/F-4). |
| Benchmark methodology | **PASS in shape, blocked by evidence** | CUDA events, warmups, iterations, and no profiler/IKET are appropriate for the short case. The committed revision/dirty-tree mismatch is blocking (F-1). |
| Provenance | **NEEDS_FIX** | The committed JSON is not attributable to `4c31c11` (F-1). |

## Required close-out actions

1. Rerun the clean short-case benchmark from a clean `4c31c11` checkout (or a
   successor commit containing only approved fixes), and commit JSON whose
   provenance exactly matches the measured revision with `dirty: false`.
2. Correct the `input_contract` metadata for `fp4_pure_fp4q` and regenerate the
   evidence artifact.
3. Add the missing AC-3 negative tests for dtype, shape, device, strides, row
   count, and head count; keep the existing partial/ambiguous/indexed tests.
4. Either add isolated per-path memory accounting or remove any report language
   that treats the shared case peak as proof of allocation removal.
5. Add a lightweight compile-cache-sharing regression test or equivalent explicit
   evidence.

Until items 1–3 are complete, **Phase 1 remains NEEDS_FIX / failed to close** under
the plan's stop-on-failure rule. The lack of FP4-Q residual support is not a failure
because the user-approved execution-order decision deliberately deferred it.
