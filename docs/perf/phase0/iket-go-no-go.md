# Phase 0b IKET Go/No-Go

## Environment and case

- Environment: `/apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/BitKV_nvfp4/_local/envs/vllm-nvfp4`
- Device selection: `CUDA_VISIBLE_DEVICES=1` (visible device is reported as `cuda:0`, NVIDIA L20A, SM100)
- Case: `batch=1, seqlen=1024, heads_q=32, heads_kv=8, fp4_pure`
- Reviewer transport: internal sub-agent only; no external `codex exec` / `codex review`

## Attempts

Exact commands are recorded below; both shells began with the mandated PATH and
`CUDA_VISIBLE_DEVICES=1`.

1. **Harness failure, not an IKET failure.** The benchmark tried to use Torch Profiler
   while IKET already owned CUPTI. CUDA activities were absent, `gpu_ms` became zero,
   and bandwidth formatting divided by zero. This excluded JIT/cache/marker placement as
   the immediate cause: the tracker found and patched the real decode kernel.
2. **Go.** Added a CUDA-event timing mode that does not acquire CUPTI and added coarse
   load-side wait ranges requested by independent review. `run-iket` completed both
   passes and emitted a 4.2 MiB non-empty JSON trace. `analyze_trace.py` reported
   `malformed_ranges = 0` for all three launches.

The historical trace was generated at implementation revision `6bef371`.

Attempt 1:

```bash
V=/apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/BitKV_nvfp4/_local/envs/vllm-nvfp4
"$V/bin/run-iket" -o /tmp/iket_fp4_round1_1785332389 --clobber --log-level debug \
  profile --postprocess all -- \
  "$V/bin/python" tests/kernel_profile/bench_decode.py --device 0 \
  --batches 1 --seqlens 1024 --heads-q 32 --heads-kv 8 \
  --iters 1 --warmup 0 --variants fp4_pure \
  --clear-fp4-compile-cache
```

Result: exit 1; `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`, missing CUDA
activities, then zero-derived bandwidth.

Attempt 2:

```bash
V=/apdcephfs_wzc1/share_303541817/hunyuan/dayoudu/dev/BitKV_nvfp4/_local/envs/vllm-nvfp4
"$V/bin/run-iket" -o /tmp/iket_fp4_round1b_1785332527 --clobber --log-level debug \
  profile --postprocess all -- \
  "$V/bin/python" tests/kernel_profile/bench_decode.py --device 0 \
  --batches 1 --seqlens 1024 --heads-q 32 --heads-kv 8 \
  --iters 1 --warmup 0 --variants fp4_pure \
  --clear-fp4-compile-cache --event-timing
```

Result: exit 0; `iket-baseline-b1-s1024.trace.json` is 4,339,958 bytes and
contains three launches. The cache-clear flag was active in both passes, and the
tracker output identified/patched the real decode kernel.

The benchmark CLI evolved after this trace was captured. The current equivalent
structural collection uses `--variants fp4_pure_bf16q --structural-only`;
the historical `--variants fp4_pure --event-timing` spelling above is retained
only to reproduce revision `6bef371`.

The topology warnings (`gpcId`/`tpcId` unavailable) match the documented non-fatal IKET
failure mode; `smId`, CTA ids, warp ids, and role analysis are present.

## Pre-registered questions

### 1. Critical warp role

The original helper incorrectly called the role with the largest sum of warp
lifetimes the critical path; that is aggregate warp-work and is biased by role
warp count. A latency-oriented envelope over the raw trace shows **softmax0 owns
the final warp completion in all three launches**. Correction still spends about
80% of aggregate role lifetime in `corr_wait_sm`, which identifies starvation,
not the launch-tail role. Softmax on the observed tail activates the Phase 2b
precondition, but Phase 2b still remains after Phase 2 as specified.

### 2. Softmax to PV serialization

The trace establishes that MMA is producer-starved on the P/O-ready dependency:
about 68% of aggregate MMA role lifetime is `mma_wait_p`, waiting on the barrier
that P production and correction release. This confirms a strong dependency and
stall point, but the current coarse ranges do **not** prove complete temporal
serialization (zero overlap) between all softmax and PV work. This is structural
evidence, not a clean-run performance number.

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
- `load_wait_res` is statically paired but unobserved in this pure-FP4 case because
  the residual path does not execute.

## Decision

**GO: IKET remains the primary timing/structure evidence source; ncu remains the quantity
and resource fallback.** Performance conclusions continue to use clean, uninstrumented
runs only.
