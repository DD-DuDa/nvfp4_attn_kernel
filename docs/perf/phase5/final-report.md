# Phase 5 — Final pure FP4-Q performance decision

## Decision

**FAIL — §8.1 is not met. Stop for human decision.**

The clean grid is complete: 96 `(batch,seqlen,head-config)` cases, 384
variant/status rows, zero missing or skipped cells. FP4-Q is compared against
D0's better FA4 split=1/heuristic result. No IKET/profiler timing is used.

D3 requires every per-head-config, per-seqlen batch geomean ratio <= 0.5. All
measured values are >1 (FP4-Q slower than FA4):

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 2.558 | 2.697 | 3.038 | 3.555 |
| GQA 32:8 | 2.526 | 2.631 | 3.033 | 3.587 |
| MQA 32:1 | 2.497 | 3.081 | 4.722 | 9.234 |

Diagnostic all-grid FP4-Q geomean: **3.322x slower**. Worst point: MQA,
batch 2, seqlen 65536, **19.129x slower** than D0.

## Gap to target

Additional speedup required from the current final state to reach the 0.5 gate:

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA | 5.12x | 5.39x | 6.08x | 7.11x |
| GQA | 5.05x | 5.26x | 6.07x | 7.17x |
| MQA | 4.99x | 6.16x | 9.44x | 18.47x |

## Roofline update

FP4 KV uses 1/3.556 the bytes of BF16 KV. Using each cell's measured better-FA4
bandwidth, the FP4 byte floor leaves the D3 target 1.778x above the ideal
same-bandwidth floor. Therefore the 2x target remains inside the byte roofline,
but the current kernel is far from that floor: it needs 5–18x more speed,
especially MQA long-context low batch.

The earlier 3.556x byte roofline is unchanged; Phase 0 measured the same device
and Phase 5 uses per-cell empirical FA4 bandwidth rather than a nominal HBM
number. This confirms remaining room exists physically, but the failed split-K
and cooperative-softmax/warp branches mean no approved in-plan lever remains.

## Evidence from completed/failed phases

- Phase 1 FP4-Q contract removed query quantization/composition cost (~2.95x vs BF16-Q short case).
- Phase 2 true query length retained GQA/MQA gains but introduced an unwaived MHA negative branch.
- Phase 2b warp15-to-load caused launch failure; warp15-to-epilogue caused severe stall; cooperative softmax was not partially implemented.
- Phase 3 split-K core/partial/combine compiled, but combined output failed the unchanged 0.99 cosine gate and was reverted.
- Phase 4 trusted metadata removes host synchronization but does not change attention-kernel throughput.
- High-batch IKET shows a softmax/correction/P-readiness chain; low-batch remains dominated by lack of validated split-K occupancy.

## Required human decision

Per §6.6 and Phase 5 instructions, execution stops here. Gates, D0 baseline,
D3 aggregation, numerical thresholds, and FP4-Q interface have not been
changed. The post-speedup FP4-Q + BF16 residual Phase is **not started** because
the speed target was not achieved.

Potential next directions requiring a new human-approved plan:

1. complete a native cooperative-softmax/P-staging redesign on SM100;
2. debug the split partial/LSE semantic mismatch and revalidate split-K;
3. redesign the kernel around decode-specific CUDA-core/GEMV or a different
   MMA/tiling strategy rather than the inherited FA4 core;
4. revisit the target/baseline only by explicit human decision.
