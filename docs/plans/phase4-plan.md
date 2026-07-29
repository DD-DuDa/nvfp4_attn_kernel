# Phase 4 Plan — Trusted metadata fast path

## Goal

Add the S7-compatible `trusted_metadata=False` public option and remove device-to-host validation synchronizations from trusted steady-state calls without weakening standalone/debug validation.

## Acceptance Criteria

- AC-1: `trusted_metadata` passes unchanged through public, composition and decode layers.
- AC-2: When false, all existing device-value errors remain; when true, only `_check_device_values` scans are skipped while shape/dtype/device/stride checks remain.
- AC-3: Trusted and normal legal calls are byte-identical for FP4-Q and BF16-Q representative cases; existing tests remain green.
- AC-4: Torch profiler steady-state trusted FP4-Q records zero `aten::_local_scalar_dense`; wall/gpu ratio improves materially.
- AC-5: Green commit, evidence report, Terra/high plus alternate-model review.

## Scope

Do not add active-row masking in this Phase, alter metadata production, relax gates, or change kernel math.
