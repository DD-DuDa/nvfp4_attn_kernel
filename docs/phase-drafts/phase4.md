# Phase 4 Draft — trusted metadata host-sync cleanup

Implement `trusted_metadata: bool=False` through interface/kernel/decode. Skip only device-value `.item()` checks when true; preserve shape/dtype/device/stride host checks. Current performance mainline is pure page-aligned FP4-Q, but BF16 residual/debug path remains tested. Acceptance: profiler `aten::_local_scalar_dense=0`, trusted/untrusted outputs exact, wall/gpu closer.
