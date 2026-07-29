# Phase 0b IKET Go/No-Go

## Environment and case

- Environment: `/apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/BitKV_nvfp4/_local/envs/vllm-nvfp4`
- Device selection: `CUDA_VISIBLE_DEVICES=1` (visible device is reported as `cuda:0`, NVIDIA L20A, SM100)
- Case: `batch=1, seqlen=1024, heads_q=32, heads_kv=8, fp4_pure`
- Reviewer transport: internal sub-agent only; no external `codex exec` / `codex review`

## Attempts

1. **Harness failure, not an IKET failure.** The benchmark tried to use Torch Profiler
   while IKET already owned CUPTI. CUDA activities were absent, `gpu_ms` became zero,
   and bandwidth formatting divided by zero. This excluded JIT/cache/marker placement as
   the immediate cause: the tracker found and patched the real decode kernel.
2. **Go.** Added a CUDA-event timing mode that does not acquire CUPTI and added coarse
   load-side wait ranges requested by independent review. `run-iket` completed both
   passes and emitted a 4.2 MiB non-empty JSON trace. `analyze_trace.py` reported
   `malformed_ranges = 0` for all three launches.

The topology warnings (`gpcId`/`tpcId` unavailable) match the documented non-fatal IKET
failure mode; `smId`, CTA ids, warp ids, and role analysis are present.

## Pre-registered questions

### 1. Critical warp role

Correction is the critical aggregate role. Softmax is within roughly 2–3% of it.
Correction spends about 80% of role lifetime in `corr_wait_sm`, so it is starved by
softmax rather than limited by correction arithmetic. This activates the Phase 2b
precondition (softmax/correction is on the critical path), but Phase 2b still remains
after Phase 2 as specified.

### 2. Softmax to PV serialization

The trace supports the pre-registered serialization hypothesis. MMA spends about 68% of
its role lifetime in `mma_wait_p`, waiting for P/O readiness after softmax and P
quantization; the wait dominates MMA's other recorded waits. This is structural evidence,
not a clean-run performance number.

### 3. CTA and SM spread

The launch grid is `(16, 1, 1)`: 16 CTAs, all 16 traced, spread across 16 SMs at one CTA
per SM. For GQA-8 with `batch=1` and `heads_kv=8`, this is two CTAs per KV head, not the
four-CTA static estimate in §5.1. The estimate is therefore falsified for this compiled
configuration and must be rechecked on the Phase 2 target/high-batch configuration rather
than assumed.

## Additional evidence

- Epilogue spends about 90% in `epi_wait_corr`, downstream of the same critical chain.
- Load lifetime is short relative to correction/softmax, though about 47–48% of it is
  `load_wait_kv`; load is not the launch's critical role for this case.
- Warp 15 remains effectively empty, confirming the idle-warp opportunity registered for
  Phase 2b.

## Decision

**GO: IKET remains the primary timing/structure evidence source; ncu remains the quantity
and resource fallback.** Performance conclusions continue to use clean, uninstrumented
runs only.
