# Phase 3 Plan — Low-batch split-K and combine

## Goal Description

Add pure FP4 split-K decode for page-aligned FP4-Q inputs: split the KV block range across CTAs, write FP32 partial O and LSE, combine them in a private standalone CuTe kernel, and select splits only where low-batch occupancy gains exceed launch/combine overhead.

## Acceptance Criteria

- AC-1: Split scheduler produces valid split indices and disjoint KV ranges; partial O/LSE shapes and layouts match the combine contract.
- AC-2: Combined output meets unchanged numerical gates against unsplit FP4 and FA4 for MHA/GQA/MQA and multiple split counts.
- AC-3: Low batch `{1,4}` ratios versus D0 improve significantly, with IKET showing more SMs used and no pathological CTA lifetime imbalance.
- AC-4: High batch does not regress because split selection falls back to unsplit where appropriate; Phase 2 MHA negative branch remains visible.
- AC-5: No residual FP4-Q support is required yet; existing BF16-Q residual tests remain green.
- AC-6: Green committed state, failure/config ledger, clean performance evidence, Terra/high primary and alternate cross-review.

## Sequence

1. Enable split core with SingleTileScheduler, FP32 partial O/LSE and fixed manual split count.
2. Integrate private combine and validate exact/quality results.
3. Benchmark split counts on low/high batch and derive a fixed heuristic.
4. Capture IKET low-batch SM spread and full regression checkpoint.
5. Close green or record failed configurations and continue per §6.6.
