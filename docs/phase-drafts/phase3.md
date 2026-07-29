# Phase 3 Draft — pure FP4 split-K + combine

## Constraints

- Current mainline is pure page-aligned FP4-Q; residual integration is post-speedup by user decision.
- Implement partial O/LSE, non-persistent split scheduler, and standalone private combine.
- Low-batch gains must not cause high-batch regression; choose split heuristic accordingly.
- Add numerical tests and IKET SM-spread evidence.
- ≤10 rounds/2h; failed configurations recorded; Terra/high + alternate review.
