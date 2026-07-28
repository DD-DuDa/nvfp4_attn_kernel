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
- Q is BF16 and quantized internally for each decode call.
- Full K/V pages are pre-quantized E2M1 FP4 with E4M3 scale factors.
- An optional paged BF16 residual tail is fused into the same decode launch.
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
8. Comments should explain invariants, layouts, synchronization, or numerical
   behavior—not development history.
9. Validate changes on an SM100 GPU. Import or syntax checks alone are not
   sufficient for CuTeDSL changes.

## Verification

Run all tests:

```bash
PYTHONPATH=src python -m pytest -q tests/kernel
```

Run query byte-equality tests:

```bash
PYTHONPATH=src python -m pytest -q tests/kernel/test_quantize_query.py
```

Run decode tests:

```bash
PYTHONPATH=src python -m pytest -q tests/kernel/test_fp4_decode_correctness.py
```

The decode suite checks BF16 FlashAttention quality, hybrid FP4
FlashAttention, the independent PyTorch FP4 oracle, zero-length residual rows,
and indexed query rows. Current numerical gates are defined in the test file;
do not relax them to hide a kernel regression.

FlashAttention and FlashInfer are test dependencies only. Production code must
not require either package.
