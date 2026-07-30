# Phase 5b — Final gate rerun after the split-K fix

## Decision

**FAIL — §8.1 is still not met.** No gate, baseline, or numerical threshold was
changed. Execution continues rather than stopping, because the evidence now
identifies a specific unexploited lever inside the kernel (see below), which is
a solvable engineering problem rather than a question requiring a human policy
decision.

## Why this run exists

`docs/perf/phase5/` measured commit `db5326f`, where split-K had been reverted
after being misdiagnosed. Commit `01d0b00` fixed the real cause — packed
physical extents were passed where logical head dimensions were required — so
the previous final grid is stale. This run repeats it with identical arguments
(30 iters, 10 warmup, 8.4M token budget, same 96-case MHA/GQA/MQA grid).

## Result

Per-`(head config, seqlen)` batch geomean of FP4-Q against the D0 better-FA4
baseline. D3 requires `<= 0.5`; values above `1.0` mean FP4 is slower.

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 1.376 | 1.855 | 1.511 | 1.415 |
| GQA 32:8 | 1.367 | 1.851 | 1.487 | 1.405 |
| MQA 32:1 | 1.222 | 1.950 | 1.449 | 1.552 |

All-grid geomean: **1.522x slower**, improved from **3.322x** at `db5326f`.
Worst point 2.068x, improved from 19.129x. 96 cases, no skips.

Remaining speedup required to reach the gate: **2.44x–3.90x**, previously
5x–18.5x.

## What changed in the shape of the problem

The bottleneck class has changed, and this matters more than the headline
number.

Before the fix, the failure was dominated by low-batch long-context occupancy:
the worst point was 19.1x and lived at MQA batch 2, seqlen 65536. Split-K has
removed that. Broken out by batch for GQA-4:

| seqlen | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.22 | 1.21 | 1.19 | 1.19 | 1.25 | 1.54 | 1.78 | 1.70 |
| 4096 | 1.96 | 1.96 | 1.97 | 1.91 | 2.07 | 1.72 | 1.60 | 1.67 |
| 16384 | 1.54 | 1.52 | 1.54 | 1.25 | 1.36 | 1.79 | 1.57 | 1.38 |
| 65536 | 1.90 | 1.34 | 1.29 | 1.33 | 1.31 | 1.55 | 1.38 | 1.24 |

Two readings:

1. Ratios now cluster in 1.2–2.1 across the whole grid rather than spanning
   1.2–19x. A uniform penalty independent of batch is a **per-CTA throughput**
   problem, not an occupancy problem.
2. `seqlen=4096` is now the worst column, and it is uniformly bad at every
   batch. It is therefore not explained by the split heuristic leaving contexts
   below 16K unsplit — low and high batch fail equally there.

## The unexploited lever

IKET on the split path (`docs/perf/phase3/split-iket-b1-s16384.txt`) shows five
of sixteen warps doing essentially no work:

| Warps | Role | Mean lifetime |
|---|---|---:|
| 0–3 | softmax0 | 29.71 us |
| 4–7 | softmax1 | **1.42 us** |
| 8–11 | correction | 31.20 us, `corr_wait_sm` 84.5% |
| 12 | MMA | 31.43 us, `mma_wait_p` 72.6% |
| 13 | epilogue | 34.30 us, `epi_wait_corr` 85.8% |
| 14 | load | 26.53 us |
| 15 | empty | **1.42 us** |

The cause is structural rather than incidental. `is_split_kv` forces
`q_stage = 1` (`fp4_decode_kernel.py:144`), and the softmax dispatch is guarded
by `if const_expr(self.q_stage == 2) or stage == 0` (`:2121`). With one Q stage
only stage 0 enters `softmax_loop`, so warps 4–7 fall straight through to the
TMEM-dealloc arrive. Split-K is now the main path for long contexts, so those
four warps are idle exactly where throughput matters most.

Correction, MMA, and epilogue each spend 72–86% of their lifetime waiting, and
the chain terminates at softmax0, which owns 82% of the launch envelope. This
is the same diagnosis BitDecoding reaches for low-bit decode: warps allocated
along the M/q-stage axis are wasted once the decode query length is one.

Amdahl estimate if softmax work were spread over eight warps instead of four:
softmax falls to roughly 14.3 us and the critical path to roughly 20.5 us,
about **1.7x**. That is a substantial fraction of the remaining 2.4–3.9x.

## Constraints on exploiting it

TMEM is the binding resource. With `q_stage = 1`, S occupies columns `[0,128)`,
the unused stage-1 S region `[128,256)` is already reused for SFQ (`:1833`), O
occupies `[256,384)`, and `[384,512)` is free. A second O accumulator fits, but
a second independent S accumulator does not, so a naive duplication of the
stage-1 pipeline is not available.

Phase 2b's failures bound the approach from the other side: expanding a warp
role tuple alone is not safe. Adding warp 15 to `load_warp_ids` produced a CUDA
launch failure, and adding it to `epilogue_warp_ids` passed focused numerics but
stalled the sweep, both because producer counts, barrier arrival counts, and
work partitioning are not derived from the role tuple.

## Next step

Design and gate a change that gives the idle softmax1 warps real work on the
split path, under the unchanged `0.99` cosine and `5e-2` max-error gates. Any
repartition that spreads a row across warps requires cross-warp max/sum
reduction in the same round; BitDecoding's Table III records a variant that was
6.1x faster and numerically wrong, which a performance-only gate would accept.
