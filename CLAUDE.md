# NVFP4 Decode Kernel

## Purpose

This repository holds two packages. `nvfp4_decode_kernel` is a standalone
SM100 implementation of paged NVFP4 decode attention; its public API is:

```python
from nvfp4_decode_kernel import fp4_decode
```

`nvfp4_vllm` is an out-of-tree vLLM V1 attention backend that owns the paged
FP4 cache and calls that kernel. vLLM loads it through the
`vllm.general_plugins` entry point in `pyproject.toml`; a caller selects it
with `attention_config={"backend": "CUSTOM"}` and `kv_cache_dtype="nvfp4"`,
along with the scheduler settings the write path requires.
`nvfp4_vllm.build_llm` binds all of them and turns on the K shift, and is the
way in unless a test is deliberately constructing an engine by hand.

**The import prohibition is about provenance, not about the `vllm` package.**
No source may be copied or vendored in from the original `nvfp4_attn`
repository, vLLM, or SGLang. This repository vendors no vLLM of its own; the
one the tests run against lives in `BitKV_nvfp4/third_party/vllm` and is there
to be read, never to be copied from. `nvfp4_vllm` necessarily imports the
installed `vllm`: it subclasses
`FlashAttentionBackend` and is written against vLLM's public extension
points. `nvfp4_decode_kernel` imports no serving framework at all, and that
is the line to hold — vLLM appears in it only in comments.

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
- `fp4_decode` takes `num_splits`, a positive power of two defaulting to one.
  Production never splits; the parameter is what keeps the split-K path
  reachable through the public entry from tests. A value above one is honoured
  only where that path can serve it — no residual argument at all, or a
  complete residual on the BF16-Q path, and never with `out_indices` — and
  raises anywhere else rather than quietly running one tile.
- The optional query scratch buffers are caller-owned, and two of them must
  arrive zeroed and stay that way: the quantizer writes only the
  `heads_q // heads_kv` scale slots per KV head that carry a query head, and
  the residual MMA's row tile only its first row, so nothing ever clears the
  rest. A scale buffer therefore belongs to one `(heads_q, heads_kv,
  head_dim)` for its life — that ratio is exactly what picks the live slots,
  and no shape check can see the reuse.
- Fused output scatter, prefill, backward, and serving-framework cache
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

## vLLM Backend Contract

Everything above is a way to be silently wrong. This is not: each constraint
here is refused when the engine is built, so breaking one costs a startup
error, never a wrong answer. The enforced list is `guards.py` and the
`supports_*` methods of `backend.py`; the reasons are the table in
`docs/tasks/1.vllm_v1_design.md` §6.1, which is the copy to keep current.

- `kv_cache_dtype="nvfp4"` is what turns any of this on. Under any other cache
  dtype the backend is a FlashAttention pass-through and must stay one.
- `block_size` must be 128, and `max_num_seqs` at most
  `control.MAX_SUPPORTED_SLOTS`, one BF16 tail slot per running request.
  Chunked prefill, prefix caching, speculative decoding, KV connectors and
  offloading, microbatching, and `pipeline_parallel_size > 1` are refused.
- CUDA graphs need no configuration. The builder reports
  `UNIFORM_SINGLE_TOKEN_DECODE` under NVFP4 and vLLM resolves
  `FULL_AND_PIECEWISE` itself: a pure decode step replays one full graph, a
  mixed step falls back to piecewise with our code outside it.
- Within a step the one-token decode rows are sorted to the front
  (`reorder_batch_threshold = 1`) and `builder.decode_split` rechecks that
  rather than trusting it. A graph-dispatched batch is padded to the captured
  width with rows carrying no token, which appear as a trailing run of zeros.
- The tail buffers and query scratch are allocated during the profile run, so
  vLLM sizes the KV cache with their cost already deducted. Anything else that
  has to be charged honestly must be allocated there too.

## Code Map

`src/nvfp4_decode_kernel/`, the kernel:

- `interface.py`: stable public `fp4_decode` signature.
- `_kernel.py`: composes Q quantization and decode.
- `_quantize.py`: quantization facade.
- `quantize_q_kernel.py`: CuTeDSL Q quantization.
- `quantize_kv_kernel.py`: CuTeDSL page quantization, dense or indexed.
- `_quantize_flashinfer.py`: FlashInfer reference for Q byte equality.
- `_decode.py`: validation, layout adaptation, compilation cache, and launch.
- `fp4_decode_kernel.py`: main SM100 CuTeDSL attention kernel.
- `split_k_combine.py`: LSE-weighted reduction over the split-K partials.
- `_fa4/`: private low-level MMA, TMA, softmax, paging, and scheduler helpers.

`src/nvfp4_vllm/`, the vLLM backend:

- `__init__.py`: plugin entry point; registers the backend under `CUSTOM`.
- `backend.py`: declares the NVFP4 page size and flat byte page shape to vLLM,
  plus the refusals the backend selector is allowed to cache.
- `guards.py`: engine configurations the path refuses, each with its reason.
- `builder.py`: the once-per-step `build()`; advances the control plane and
  answers how much of a batch may be captured into a CUDA graph.
- `control.py`: single-CTA Triton kernel that assigns a BF16 tail slot per
  request and splits each row's length into its FP4 and residual parts.
- `metadata.py`: FlashAttention's metadata plus that per-step answer.
- `layout.py`: reads one vLLM block as four NVFP4 regions, block-major.
- `runtime.py`: GPU state one model's layers share — tail buffers, carved
  cache views, query scratch — allocated during the profile run.
- `write.py`: routes this step's K/V into whole FP4 pages or the BF16 tail.
- `impl.py`: per-layer forward; the decode prefix through `fp4_decode`, the
  prefill suffix through cache-free varlen FlashAttention.
- `promote.py`: after the last layer, seals filled tail pages into FP4.

Tests:

- `tests/kernel/`: quantization and decode correctness tests.
- `tests/e2e/`: vLLM integration — guards, control plane, write and read
  paths, promotion, memory accounting, CUDA graph capture. `test_soak.py`
  lives here but is its own suite; `scripts/run_tests.sh e2e` excludes it.
- `tests/kernel_profile/`: benchmarks and probes. `scripts/run_tests.sh` runs
  none of them, though `test_bench_decode.py` there holds the
  `split_k_heuristic` unit test.

## Development Rules

1. Keep the public API small and decode-specific.
2. Do not quantize K/V inside `fp4_decode`; serving code owns cache
   quantization, which here means `nvfp4_vllm.write` and `nvfp4_vllm.promote`.
3. Keep `nvfp4_decode_kernel` free of any serving-framework dependency, and
   both packages free of code copied from the old repository or from the vLLM
   tree. `nvfp4_vllm` depends on vLLM by construction; see Purpose.
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
fails, so a clean run is **1 failed, 85 passed**. Judge a change by whether it
adds a failure, not by whether the suite is green.

The case asserts that feeding a pre-quantized FP4 query reproduces the BF16
query path bitwise. It fails only at MQA (`heads_kv = 1`); the MHA and GQA
parametrizations pass.

The cause is that **`heads_q = 32, heads_kv = 1` on the pure-FP4 path does not
decode reproducibly**: eight repeats of one identical call each differed from
the first. Adding a residual makes the same geometry stable, and `(8, 8)`,
`(32, 8)` and `(16, 1)` are stable over sixteen repeats. So the test is not
wrong and nothing survives between calls — the kernel returns a different
answer each time in that one configuration. Treat it as a kernel bug, not a
flaky test.

Two consequences. A bitwise comparison of two decodes must avoid `(32, 1)`;
`tests/kernel` uses `(16, 1)` for MQA where it needs one. And because the
failure is a coin flip rather than a fixed outcome, and `pytest-randomly` is
installed on top of that, pass `-p no:randomly` whenever comparing two runs —
otherwise a run that happens to come out green reads as a fix.

FlashAttention and FlashInfer are test dependencies only. Production code must
not require either package.
