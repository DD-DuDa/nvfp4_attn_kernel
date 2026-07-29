# Phase 2b Plan — Rebalance specialized warp work

## Goal Description

Use the high-batch IKET evidence to reduce the softmax/correction/P-readiness critical chain. First reclaim idle warp 15 with the smallest layout-safe change. Only then attempt a softmax stage redistribution, and only with complete cross-warp row reduction/P staging correctness in the same round.

## Acceptance Criteria

- AC-1: Warp 15 performs attributable useful work or a measured experiment proves it cannot be safely reclaimed; no new idle role is created.
- AC-2: IKET launch-tail/wait fractions improve for the softmax/correction/P chain and warp lifetime dispersion narrows on representative high-batch cases.
- AC-3: Exact numerical gate and full `tests/kernel` pass in the same round as every layout change; no fast-but-wrong intermediate state is committed.
- AC-4: Clean full-table checkpoints show no new low-batch or MHA regression beyond the explicitly carried Phase 2 branch; GQA/MQA gains are preserved.
- AC-5: If softmax rows cross warps, cross-warp rowmax/sum and P operand staging are implemented together; otherwise that branch is rejected/reverted.
- AC-6: At most 10 rounds/2h, failed layouts logged, green commit, Terra/high and alternate-model review.

## Path Boundaries

May modify role IDs, specialized dispatch, softmax/correction coordination, and `_fa4/softmax.py` only if decode correctness requires it. Must not implement split-K, host-sync changes, residual FP4-Q, or relax gates.

## Sequence

1. Record baseline clean/IKET cases and effective role map.
2. Reclaim warp 15 with the minimal safe role expansion; run numerical, clean table checkpoint, IKET.
3. If useful and evidence still points to softmax, design cooperative softmax with reduction/staging as one atomic correctness change.
4. Reject any branch that regresses numerical gates or broad table; document measured failures.
5. Close green with evidence and dual internal review.

## Review Transport

No external Codex transport. Primary GPT-5.6-Terra/high, cross-review another internal model.
