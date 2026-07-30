# Phase 3 — pure FP4 split-K resumed and validated

## Decision

**PASS at the Phase level.** Pure page-aligned FP4-Q decode now has a
numerically valid split-K producer, a standalone private CuTe combine kernel,
and a conservative production split heuristic. The unchanged `0.99` cosine
and `5e-2` max-error gates pass. BF16 residual and direct scatter continue to
use the existing unsplit path, as required by the user-approved execution
order.

This Phase does **not** claim the final §8.1 gate. That remains a Phase 5
full-grid decision against D0.

## Root cause of the earlier false failure

The implementation in `b28eb38` launched the split producer with
`tuple(query_fp4.shape)` and `tuple(key_pages_fp4.shape)` as logical kernel
shapes. Those packed tensors end in `64` bytes because one byte holds two FP4
values. The core kernel requires the logical FP4 head dimension `128`, exactly
as the existing unsplit launch already passes.

Consequences of the bad metadata:

- Q/K were interpreted as head-dim 64 while V was passed as logical 128;
- every split partial was already wrong before combine;
- a PyTorch LSE-weighted combine therefore also failed, proving the problem was
  not specific to the private CuTe combine;
- compile/file-cache hits obscured several early diagnostic edits until the
  debug shells explicitly used `CUTE_DSL_NO_CACHE=1` and
  `CUTE_DSL_DISABLE_FILE_CACHING=1`.

After passing logical `(rows, 1, heads_q, 128)` and
`(pages, 128, heads_kv, 128)` Q/K/V shapes, each partial agrees with an
independent page-sliced unsplit launch at about `0.999998` cosine.

## Numerical evidence

A PyTorch FP32 oracle combines locally normalized partial O with
`softmax(partial_lse, dim=split)`. The private CuTe combine then independently
reproduces that result. Representative combined-vs-unsplit results:

| Attention | splits=2 | splits=4 | splits=8 | max abs range |
|---|---:|---:|---:|---:|
| GQA `32:8` | 0.99957 | 0.99938 | 0.99926 | 0.00110–0.00143 |
| MHA `16:16` | 0.99957 | 0.99938 | 0.99920 | 0.00110–0.00146 |
| MQA `16:1` | 0.99950 | 0.99934 | 0.99922 | 0.00098–0.00131 |

The new kernel test covers MHA/GQA/MQA with split `{2,4,8}` and uses the
unchanged repository constants `FP4_MIN_COSINE=0.99` and
`FP4_MAX_ABS_ERROR=5e-2`.

## Heuristic and clean performance evidence

The production heuristic:

- leaves contexts below 16K tokens unsplit because fixed combine overhead wins;
- selects from fixed `{2,4,8}` based on `batch * heads_kv`, 148-SM occupancy,
  and available 128-token page blocks;
- preserves the unsplit residual and direct-scatter paths.

A 30-iteration / 10-warmup clean GQA-4 sample (`/tmp/phase3_perf_tuned.json`)
shows the intended shape of the gain:

| batch | seqlen | selected split | prior Phase-5 FP4-Q ms | resumed FP4-Q ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 16384 | 8 | 0.3312 | 0.0920 | 3.60x |
| 1 | 65536 | 8 | 1.1056 | 0.1138 | 9.71x |
| 4 | 16384 | 4 | 0.3467 | 0.0932 | 3.72x |
| 4 | 65536 | 4 | 1.1284 | 0.2162 | 5.22x |
| 16 | 16384 | 2 | 0.3577 | 0.2227 | 1.61x |
| 16 | 65536 | 2 | 1.1224 | 0.9944 | 1.13x |

Short-context samples stay unsplit and remain around the pre-split latency.
The exact numbers are diagnostic Phase evidence, not the final full-grid gate.

## IKET structural evidence

For GQA `batch=1, seqlen=16384`, split=8 changes the main launch grid to
`(1,64,1)`: 64 CTAs traced on 64 distinct SMs, max one CTA per SM, with
`malformed_ranges=0`. The prior Phase-0 `batch=1,seqlen=1024` unsplit trace
used only 16 SMs, so split-K demonstrably raises SM spread 4x for this case.

The main launch remains softmax/correction/P-readiness limited:
`corr_wait_sm` is about 84% of correction lifetime and `mma_wait_p` about 73%
of MMA lifetime. This is consistent with split-K solving occupancy rather than
per-CTA throughput. The combine kernel is a separate fixed launch and is not
used as clean timing evidence under IKET.

## Failed/excluded branches

- The earlier “LSE layout mismatch” diagnosis was downstream of bad logical
  Q/K shapes and is superseded.
- Partial O remains FP32 because the split workspace itself is FP32 and the
  core derives `o_dtype` from that tensor. An extra `partial_output_fp32`
  epilogue flag was redundant and was removed.
- Splitting contexts below 16K loses to combine overhead.
- Blindly maximizing split count regresses higher-batch points; fixed split
  selection is therefore occupancy- and context-aware.
- External `codex exec` / `codex review` paths remain prohibited. Reviews use
  internal agents only; connection failures are recorded separately rather
  than switching back to the external transport.

## Full-grid Phase checkpoint

`docs/perf/phase3/full-grid-clean.json` covers the complete 96-case
MHA/GQA/MQA grid with no missing or skipped cases (10 measured CUDA-event
iterations after 5 warmups). It is a Phase checkpoint, not the Phase 5 final
D3 run. FP4-Q/D0 batch-geomean ratios are:

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 1.377 | 1.765 | 1.538 | 1.492 |
| GQA 32:8 | 1.352 | 1.438 | 1.524 | 1.489 |
| MQA 32:1 | 1.164 | 1.841 | 1.385 | 1.536 |

The diagnostic all-grid geomean is **1.482x slower than D0**, improved from
the prior Phase-5 artifact's **3.322x slower**. Every §8.1 cell remains above
`0.5`; final acceptance is therefore not claimed here. Phase 4 remains intact,
and Phase 5 must be rerun after the Phase 3 green commit exactly as the task
sequence requires.
