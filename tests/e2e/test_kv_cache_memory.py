"""Opt-in end-to-end check that the NVFP4 KV cache really is smaller.

Builds two engines with identical configuration, one BF16 and one NVFP4, and
compares how many tokens their caches hold. A cache that silently falls back to
BF16 pages, or a page size declaration that does not reach vLLM's accounting,
shows up here as a ratio near 1.0.

``CUSTOM`` resolves through the ``vllm.general_plugins`` entry point declared
in ``pyproject.toml``, so this package has to be installed in the environment
running the test rather than merely importable.

The test is skipped unless ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import pytest


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.9

# The NVFP4 path refuses chunked prefill, and vLLM then requires a batch wide
# enough that no prompt has to be split. Both engines get the same setting so
# the comparison is not measuring a scheduler difference.
MAX_NUM_BATCHED_TOKENS = MAX_MODEL_LEN

# A BF16 value costs 16 bits; an NVFP4 one costs 4 plus a shared E4M3 scale per
# group of 16, so 4.5. The page size ratio is exactly 16 / 4.5.
EXPECTED_PAGE_SIZE_RATIO = 32 / 9

# Below the exact ratio because the two engines profile free memory
# independently, and the BF16 tail buffer will later take a fixed bite out of
# the NVFP4 side.
MIN_TOKEN_CAPACITY_RATIO = 3.0

# A larger gap means the first engine did not release its cache, so the ratio
# would measure teardown rather than page size.
MAX_KV_BUDGET_SKEW = 0.02


@dataclass(frozen=True)
class CacheReport:
    label: str
    page_size_bytes: int
    num_layers: int
    num_blocks: int
    total_bytes: int

    @property
    def tokens(self) -> int:
        return self.num_blocks * PAGE_SIZE

    def summary(self) -> str:
        return (
            f"{self.label}: page {self.page_size_bytes:,} B/layer, "
            f"{self.num_blocks:,} blocks, {self.tokens:,} tokens, "
            f"{self.total_bytes / 2**30:.2f} GiB"
        )


def _require_sm100() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _model_runner(llm):
    """Reach the worker's model runner through the in-process engine core."""
    core = llm.llm_engine.engine_core
    while hasattr(core, "engine_core"):
        core = core.engine_core
    return core.model_executor.driver_worker.worker.model_runner


def _read_report(label: str, llm) -> CacheReport:
    """Summarize a live engine's KV cache as ints.

    Retaining a spec or a cache tensor would keep the worker alive past
    teardown, and the next engine needs that memory back.
    """
    runner = _model_runner(llm)
    spec = next(iter(runner.get_kv_cache_spec().values()))
    caches = runner.kv_caches
    layer_bytes = caches[0].numel() * caches[0].element_size()
    assert layer_bytes % spec.page_size_bytes == 0
    return CacheReport(
        label=label,
        page_size_bytes=spec.page_size_bytes,
        num_layers=len(caches),
        num_blocks=layer_bytes // spec.page_size_bytes,
        total_bytes=layer_bytes * len(caches),
    )


def _measure(label: str, kv_cache_dtype: str, backend: str) -> CacheReport:
    """Build an engine, read its allocated KV cache, and tear it down."""
    import torch
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": backend},
    )
    try:
        return _read_report(label, llm)
    finally:
        # The next engine sizes its cache from free memory. Startup allocations
        # are behind a gc.freeze(); only shutdown() unfreezes them.
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _free_gib() -> float:
    import torch

    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_nvfp4_kv_cache_holds_more_than_three_times_the_tokens(monkeypatch):
    _require_sm100()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    free_before = _free_gib()
    bf16 = _measure("BF16", "auto", "FLASH_ATTN")
    free_between = _free_gib()
    nvfp4 = _measure("NVFP4", "nvfp4", "CUSTOM")
    print(f"\n{bf16.summary()}\n{nvfp4.summary()}")

    assert free_between >= free_before * 0.95, (
        f"the BF16 engine released only {free_between:.2f} of the "
        f"{free_before:.2f} GiB it started from, so the NVFP4 engine would be "
        "sized against a smaller budget"
    )

    assert bf16.num_layers == nvfp4.num_layers, (
        f"the two engines disagree on layer count ({bf16.num_layers} vs "
        f"{nvfp4.num_layers}), so their caches are not comparable"
    )

    skew = abs(nvfp4.total_bytes - bf16.total_bytes) / bf16.total_bytes
    assert skew <= MAX_KV_BUDGET_SKEW, (
        f"the two engines were given KV budgets {skew * 100:.2f}% apart "
        f"({bf16.summary()} | {nvfp4.summary()}), so a capacity comparison "
        "would measure memory release rather than page size"
    )

    page_ratio = bf16.page_size_bytes / nvfp4.page_size_bytes
    assert page_ratio == pytest.approx(EXPECTED_PAGE_SIZE_RATIO, rel=1e-6), (
        f"declared page size ratio is {page_ratio:.4f}, expected "
        f"{EXPECTED_PAGE_SIZE_RATIO:.4f} ({bf16.summary()} | "
        f"{nvfp4.summary()})"
    )

    token_ratio = nvfp4.tokens / bf16.tokens
    print(
        f"page size ratio {page_ratio:.4f} | token capacity ratio "
        f"{token_ratio:.4f}"
    )
    assert token_ratio > MIN_TOKEN_CAPACITY_RATIO, (
        f"the NVFP4 cache holds only {token_ratio:.4f}x the tokens of the "
        f"BF16 cache, below the {MIN_TOKEN_CAPACITY_RATIO}x floor.\n"
        f"{bf16.summary()}\n{nvfp4.summary()}"
    )
