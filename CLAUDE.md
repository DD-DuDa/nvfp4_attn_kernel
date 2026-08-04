# NVFP4 Decode Kernel

## Purpose

This repository is a standalone SM100 implementation of paged NVFP4 decode
attention. It must not import code from the original `nvfp4_attn` repository,
vLLM, SGLang, or a `third_party` checkout.

The public API is:

```python
from nvfp4_decode_kernel import fp4_decode
```

## Supported Kernel Contract

- NVIDIA SM100 GPUs only.
- Decode query length is one.
- Page size and head dimension are both fixed at 128.
- Q is either BF16 and quantized internally, or pre-quantized E2M1 FP4 with
  E4M3 scales through the same `fp4_decode` entry.
- Full K/V pages are pre-quantized E2M1 FP4 with E4M3 scale factors.
- An optional paged BF16 residual tail is fused into the same decode launch.
- The pre-quantized FP4-Q path initially targets pure page-aligned FP4 K/V;
  FP4-Q plus BF16 residual is implemented only after the performance target.
  The BF16-Q path continues to support residual tails throughout.
- MHA, GQA, and MQA are supported.
- Attention is non-causal because decode only attends to the supplied cache.
- Split-K, fused output scatter, prefill, backward, and serving-framework cache
  bookkeeping are intentionally out of scope.

Important tensor layouts:

- Q: `[rows, heads_q, 128]`
- FP4 K: `[pages, 128, heads_kv, 64]`
- FP4 V: `[pages, heads_kv, 128, 64]`
- Page table: `[rows, max_pages]`, `torch.int32`
- FP4 sequence lengths: `[rows]`, page-aligned `torch.int32`
- BF16 residual K/V: `[pages, 128, heads_kv, 128]`

All four residual arguments must be supplied together. A
`seqused_residual` value of zero must contribute exactly nothing, including in
a batch containing other rows with nonzero residual lengths.

## Code Map

- `interface.py`: stable public `fp4_decode` signature.
- `_kernel.py`: composes Q quantization and decode.
- `_quantize.py`: quantization facade.
- `quantize_q_kernel.py`: CuTeDSL Q quantization.
- `quantize_kv_kernel.py`: CuTeDSL page quantization.
- `_quantize_flashinfer.py`: FlashInfer reference for Q byte equality.
- `_decode.py`: validation, layout adaptation, compilation cache, and launch.
- `fp4_decode_kernel.py`: main SM100 CuTeDSL attention kernel.
- `_fa4/`: private low-level MMA, TMA, softmax, paging, and scheduler helpers.
- `tests/kernel/`: quantization and decode correctness tests.

## Development Rules

1. Keep the public API small and decode-specific.
2. Do not quantize K/V inside `fp4_decode`; serving code owns cache
   quantization.
3. Do not introduce dependencies on the old repository or serving frameworks.
4. Preserve kernel-native K/V and scale layouts. Avoid per-call transposes or
   materialized scale copies.
5. Preserve exact zero-residual semantics and `query_row_indices` behavior.
6. Keep `_fa4` private. Change it only when required by the decode kernel.
7. Do not add experimental environment switches, diagnostic hardcodes,
   historical `Plan-*` comments, or debug dump paths.
   Permanent IKET ranges inside the decode kernel are the sole exception:
   they are stripped from normal builds and must remain covered by
   `tests/kernel`.
8. Comments should explain invariants, layouts, synchronization, or numerical
   behavior—not development history.
9. Validate changes on an SM100 GPU. Import or syntax checks alone are not
   sufficient for CuTeDSL changes.
10. For this repository's Humanize/RLCR workflow, do not invoke the external
    `codex exec` or `codex review` transport. Independent implementation and
    code reviews must be performed by a newly launched internal sub-agent,
    with the usual plan, round contract, committed implementation, summary,
    review-result, fix, and verification artifacts preserved.
    Prefer GPT-5.6-Terra at high reasoning for the primary review and use a
    different internal model for a cross-check.

## Verification

Use `scripts/run_tests.sh`. The `python` on `PATH` cannot run any suite: its
CuTeDSL has no `iket`, which `fp4_decode_kernel.py` imports at module scope,
and it has no compiled vLLM. The script points at an environment that has
both.

```bash
scripts/run_tests.sh kernel          # kernel numerics
scripts/run_tests.sh e2e             # vLLM integration, no soak
scripts/run_tests.sh soak            # slot reuse, many requests
scripts/run_tests.sh all
scripts/run_tests.sh kernel -- -k residual -x   # after --, pytest's own
```

The decode suite checks BF16 FlashAttention quality, hybrid FP4
FlashAttention, the independent PyTorch FP4 oracle, zero-length residual rows,
and indexed query rows. Current numerical gates are defined in the test file;
do not relax them to hide a kernel regression.

### Known failure, deliberately not fixed

`tests/kernel` is not green. `test_prequantized_query_matches_bf16_query_exactly[32-1]`
fails, so a clean run is **1 failed, 72 passed**. Judge a change by whether it
adds a failure, not by whether the suite is green.

The case asserts that feeding a pre-quantized FP4 query reproduces the BF16
query path bitwise. It fails only at MQA (`heads_kv = 1`); the MHA and GQA
parametrizations pass. It is also order-dependent: it passes when run alone and
fails when it runs after the other two, which means something survives between
calls in one process. That state dependence is the part worth chasing, since
CUDA graph capture and persistent scratch buffers both amplify it.

`pytest-randomly` is installed, so test order — and therefore this failure —
varies run to run. Pass `-p no:randomly` whenever comparing two runs, or the
comparison is meaningless.

FlashAttention and FlashInfer are test dependencies only. Production code must
not require either package.
