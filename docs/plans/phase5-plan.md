# Phase 5 Plan — Final clean performance closure

## Goal

Measure the complete pure FP4-Q page-aligned grid without IKET/profiler instrumentation, compare every head configuration to D0's better FA4 baseline, apply D3 batch geometric means per seqlen, and make the only final §8.1 pass/fail decision.

## Acceptance Criteria

- AC-1: Complete batches 1–128 × seqlens 1024–65536 × MHA/GQA/MQA; zero missing/skipped cells.
- AC-2: Clean CUDA-event data, trusted metadata for FP4-Q, FA4 varlen better-of-two baseline, correct provenance.
- AC-3: For every head configuration and each seqlen, FP4-Q/D0 batch geomean <=0.5.
- AC-4: Report per-seqlen gates, diagnostic overall geomean, worst point, BF16-Q comparison, numerical gate and failure branches.
- AC-5: If any gate fails, calculate remaining roofline margin and stop for user; no residual Phase, no threshold change.

## Review

Internal Terra/high primary and alternate cross-review; no external transport.
