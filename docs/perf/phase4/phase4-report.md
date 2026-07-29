# Phase 4 — Trusted metadata fast path

`trusted_metadata=False` is threaded through the single public entry,
composition layer and decode validation. The default/debug path retains every
device-value validation. The trusted path skips only `_check_device_values`
scans; shape, dtype, device, contiguity and layout checks remain.

## Validation

- trusted and checked pure FP4-Q outputs are byte-identical
- trusted path still rejects invalid host-visible shapes
- full gate: **58 passed**
- Torch profiler steady-state: `aten::_local_scalar_dense` count **0**

## Representative steady-state timing

Batch 1, seqlen 1024, GQA-4, pure FP4-Q, 100 iterations:

| Mode | GPU ms | Wall ms | Wall/GPU |
|---|---:|---:|---:|
| checked | 0.11418 | 0.11468 | 1.004 |
| trusted | **0.04835** | **0.04859** | **1.005** |

The trusted path removes about 2.36x of end-to-end latency in this validation-
dominated cell while keeping wall/GPU near one. This is host/API overhead, not
an attention-kernel throughput change, and Phase 5 GPU gate remains separate.
