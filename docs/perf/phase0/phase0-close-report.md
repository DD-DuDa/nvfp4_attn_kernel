# Phase 0 Close Report

## Outcome

Phase 0 established reliable IKET and clean benchmark infrastructure without changing the decode algorithm. The Phase-level measurement gate is met: numerical tests remain green, clean timing is stable, IKET evidence is non-empty and attributable, and the benchmark records D0/D4/provenance/coverage semantics. The final §8.1 performance gate is intentionally not evaluated until Phase 5.

## Expected versus measured

- Expected: no algorithmic speedup and no clean-run regression attributable to instrumentation, because IKET calls are stripped outside profiling.
- Measured: representative clean runs are stable within 0.36%; the BF16-Q baseline remains in the same broad range as the pre-task baseline. Phase 0 changes are measurement-only.

## IKET decision

- GO: real decode trace, three launches, zero malformed ranges.
- Launch-tail envelope owner: softmax0 for all three baseline launches. This
  is descriptive timing evidence, not a causal dependency proof.
- Structural stall: MMA is strongly producer-starved on P/O-ready dependency; complete zero-overlap serialization is not claimed.
- Baseline grid: 16 CTAs across 16 SMs, one CTA/SM.

## Benchmark evidence

- Historical `full-grid-smoke.json` contains all 96 required batch/seqlen/head-config cases (384 variant/status rows), with zero missing or skipped cases. It is explicitly a dirty-tree one-iteration coverage/representability smoke, not clean final performance evidence.
- D0 selects the faster of FA4 split=1 and FA4 heuristic, preserving varlen pack-GQA.
- D4 records BF16-Q and explicit FP4-Q unavailable-until-Phase-1 status.
- Long case batch=128, seqlen=65536 completes with 48.077 GiB conservative
  whole-case peak allocation; no artificial max-token skip.
- Historical pre-Round-3-schema kernel breakdown is non-empty; Q quantization is 0.00611 ms (11.15% of diagnostic profiler GPU time).
- Provenance records environment, versions, GPU, commit, arguments, timestamp and baseline definition.

## Stability

The following are historical clean timing runs generated during Round 2 on the pre-Round-3 JSON schema; their raw provenance is retained and they are used only for the <3% noise check.


| Variant | Run 1 | Run 2 | Run 3 | Range/geomean |
|---|---:|---:|---:|---:|
| FA4 split=1 | 0.164074 | 0.163996 | 0.164517 | 0.318% |
| FA4 heuristic | 0.164378 | 0.163984 | 0.164397 | 0.251% |
| FP4 BF16-Q | 0.637499 | 0.635958 | 0.638197 | 0.351% |

## Failure ledger

1. Distinct dynamic-branch lifetime tokens failed CuTeDSL scoping; replaced with a branch-safe softmax lifetime range.
2. Installed FA4 moved the heuristic import; corrected to `flash_attn.cute.interface`.
3. IKET plus Torch Profiler conflicted over CUPTI; IKET workload now uses structural-only mode / no profiler timing.
4. Aggregate warp-work was mislabeled critical path; analyzer now separates work from latency envelope.
5. One-shot quantization of 65,536 pages exceeded a launch limit; setup-time chunking now preserves native scale strides.

## Numerical gate

- `tests/kernel` remains 46/46 green.
- Benchmark infrastructure tests are 5/5 green (51/51 combined).
- No numerical thresholds or D0–D8 decisions were changed.

## Next phase

Phase 1 implements D2: single `fp4_decode` entry with optional pre-quantized `query_fp4 + query_scales`, shared core, and new FP4-Q numerical tests. Phase 1 failure is the sole overall-stop exception.
