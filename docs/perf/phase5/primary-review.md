# Phase 5 Primary Review — Stop Audit

## Verdict

**PASS — the Phase 5 stop report is correct.**

This is a correct **performance-gate failure and stop**, not project completion.
The pure FP4-Q §8.1 target failed, so the required disposition is to stop for a
human decision. The residual compatibility phase has correctly not started.

## Evidence audited

- Final report: `docs/perf/phase5/final-report.md`
- Final grid: `docs/perf/phase5/final-grid.json`
- Final reporting commit: `f44fa70`
- Measured clean revision recorded in the grid: `db5326f`, `dirty=false`
- Last implementation/test revision: `738cc2c`; `f44fa70` adds only the Phase 5
  grid and report, so it does not alter the tested kernel state.

The grid has the required 96 unique
`batch × seqlen × head-config` cases
(`8 × 4 × 3`) and 384 rows (`96 × 4` variants).  Every row has `status=ok`;
there are zero requested or Phase-0-required missing/skipped cells.  It uses 10
warmups and 30 CUDA-event iterations, trusted metadata, and no profiler/IKET
timing.

## D0–D8 audit

| Decision | Result | Audit finding |
|---|---|---|
| D0 | PASS | Each FP4 ratio uses the lower measured CUDA-event latency of `fa4_bf16` (split=1) and `fa4_split` (heuristic), both on the required varlen path. Independent recomputation from all 96 cases found zero baseline/ratio mismatches. |
| D1 | PASS | IKET markers remain in the decode source. The final gate is correctly a clean, uninstrumented CUDA-event measurement rather than profiler timing. |
| D2 | PASS | The single `fp4_decode` entry and optional `query_fp4 + query_scales` contract remain unchanged; final reporting introduces no implementation change. |
| D3 | PASS as applied; gate fails | Per-head-config, per-seqlen batch geometric means were recomputed from the eight batches and exactly reproduce the JSON summary. All 12 are greater than 0.5. |
| D4 | PASS | The gate path is `fp4_pure_fp4q` / better-FA4. `fp4_pure_bf16q` is separately timed and reported, not substituted into the gate. |
| D5 | PASS | Work remains on `perf/fp4-decode-2x`. The final reporting commit is documentation-only; the inherited `738cc2c` implementation has the recorded 58-passing `tests/kernel` result. |
| D6 | PASS | This is the final Phase 5 decision. Failure is recorded with causes and excluded branches, followed by a stop rather than additional unsanctioned iteration. |
| D7 | PASS | No performance/numerical threshold, D0 baseline, or aggregation rule was relaxed or replaced. |
| D8 | PASS | Phase 0's IKET go decision and retained markers remain intact; no contrary profiling claim is used for the clean timing gate. |

## Calculation check

Recomputed FP4-Q/D0 batch-geomean ratios:

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 2.558172 | 2.697390 | 3.037884 | 3.554538 |
| GQA 32:8 | 2.526093 | 2.631311 | 3.032854 | 3.587210 |
| MQA 32:1 | 2.496864 | 3.081480 | 4.722336 | 9.233686 |

Thus every required D3 result is slower than D0, not merely above the 0.5
2x-speedup threshold. The all-grid diagnostic geometric mean is `3.322222x`
slower, and the worst point is MQA (`32:1`), batch 2, seqlen 65536:
`1.104113 ms / 0.057719 ms = 19.128954x`. These reproduce the report after
rounding.

The stated additional speedup is also correct: current ratio divided by `0.5`
equals the required factor, giving 4.99x–18.47x (reported as 5x–18x).

The roofline arithmetic is correct under its stated same-bandwidth assumption:
FP4 transfers `1/3.556` of BF16 KV bytes, so the FP4 byte floor is
`FA4_latency / 3.556`; the target is `FA4_latency / 2`; and
`(1/2) / (1/3.556) = 1.778`. Therefore the target remains within this byte
roofline, while the measured kernel remains far from it. This is a feasibility
observation, not a justification to waive the failed gate.

## §6.6, §8.1, and residual deferral

- **§8.1:** Coverage, per-seqlen batch-geomean reporting, diagnostic overall
  geomean, and worst-point reporting are complete. The acceptance threshold
  (`<= 0.5`) fails for every head-config/seqlen group.
- **§6.6:** The report preserves the final green implementation, records failed
  branches, does not change a gate, and stops for a human decision. The final
  report correctly does not describe failure as success or as project COMPLETE.
- **User residual deferral / Phase 6:** Correctly respected. The revision range
  from the last implementation commit through `f44fa70` adds only Phase 5
  planning/reporting artifacts; it contains no FP4-Q + BF16-residual
  implementation. Since Phase 5 did not meet §8.1, starting that phase would
  have violated the explicit ordering rule.

## Required disposition

**Stop here for a new human-approved decision/plan.** No residual phase, gate
relaxation, baseline substitution, or further optimization loop is authorized
by this audit.
