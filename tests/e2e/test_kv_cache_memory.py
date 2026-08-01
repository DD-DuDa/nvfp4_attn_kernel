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
    tail_bytes: int = 0

    @property
    def tokens(self) -> int:
        return self.num_blocks * PAGE_SIZE

    @property
    def charged_bytes(self) -> int:
        """Everything this cache costs, including what is not in the cache.

        The BF16 tail is the FP4 layout's own overhead: V is packed along the
        token axis a page at a time, so a partial page has to live somewhere
        else. Leaving it out would be the accounting mistake this whole file
        exists to prevent, in the other direction.
        """
        return self.total_bytes + self.tail_bytes

    @property
    def tokens_per_gib(self) -> float:
        return self.tokens / (self.charged_bytes / 2**30)

    def summary(self) -> str:
        tail = (
            f", tail {self.tail_bytes / 2**20:,.0f} MiB" if self.tail_bytes else ""
        )
        return (
            f"{self.label}: page {self.page_size_bytes:,} B/layer, "
            f"{self.num_blocks:,} blocks, {self.tokens:,} tokens, "
            f"{self.total_bytes / 2**30:.2f} GiB{tail}"
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


def _tail_bytes(llm) -> int:
    """What the BF16 tail costs this engine, or zero if it has none.

    Reached through a layer rather than through the runtime module, because
    the runtime is attached to the layers and there is no registry of it.
    """
    runner = _model_runner(llm)
    for module in runner.model.modules():
        runtime = getattr(getattr(module, "impl", None), "runtime", None)
        if runtime is not None:
            return runtime.tail_bytes
    return 0


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
        tail_bytes=_tail_bytes(llm),
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


@pytest.fixture(scope="module")
def reports() -> dict[str, CacheReport]:
    """One pair of engines, read by every test in the file.

    Built once because building them is most of the runtime here, and because
    two engines profiled at different moments would not be comparable.
    """
    _require_sm100()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
    return {"bf16": bf16, "nvfp4": nvfp4}


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_nvfp4_kv_cache_holds_more_than_three_times_the_tokens(reports):
    bf16, nvfp4 = reports["bf16"], reports["nvfp4"]

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


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_the_tail_buffer_does_not_eat_the_saving(reports):
    """The ratio above is gross. This is the same ratio after the rent.

    The FP4 layout cannot store a partial page, so it keeps one BF16 page per
    layer per slot on the side. That memory is not in vLLM's KV accounting and
    it is not free: it scales with ``MAX_SUPPORTED_SLOTS``, which is why
    raising that constant is a memory decision and not only a kernel one.

    Charging it changes the honest headline, and the test is here so the
    charge is visible rather than discovered later by somebody sizing a
    deployment.
    """
    bf16, nvfp4 = reports["bf16"], reports["nvfp4"]

    assert bf16.tail_bytes == 0, (
        "the BF16 arm reported a tail buffer, so it went through the NVFP4 "
        "runtime and the comparison has no baseline"
    )
    assert nvfp4.tail_bytes > 0, (
        "the NVFP4 arm reported no tail buffer, so either the runtime was "
        "never built or this test is reading the wrong attribute, and the "
        "net figure below would flatter us"
    )

    # This engine runs MAX_NUM_SEQS slots, not the ceiling. The tail is linear
    # in slot count, so the worst case is reported alongside the measurement —
    # that is the figure somebody sizing a deployment needs, and quoting the
    # sample instead would understate it fourfold.
    from nvfp4_vllm.control import MAX_SUPPORTED_SLOTS

    per_slot = nvfp4.tail_bytes / MAX_NUM_SEQS
    at_ceiling = per_slot * MAX_SUPPORTED_SLOTS
    gross = nvfp4.tokens / bf16.tokens
    net = nvfp4.tokens_per_gib / bf16.tokens_per_gib
    worst = (
        nvfp4.tokens / ((nvfp4.total_bytes + at_ceiling) / 2**30)
    ) / bf16.tokens_per_gib
    print(
        f"tail {nvfp4.tail_bytes / 2**20:,.0f} MiB at {MAX_NUM_SEQS} slots, "
        f"{per_slot / 2**20:,.1f} MiB per slot, "
        f"{at_ceiling / 2**20:,.0f} MiB at the {MAX_SUPPORTED_SLOTS}-slot "
        f"ceiling ({at_ceiling / nvfp4.total_bytes * 100:.2f}% of the cache)\n"
        f"capacity ratio {gross:.4f} gross, {net:.4f} net, "
        f"{worst:.4f} net at the ceiling"
    )
    assert worst > MIN_TOKEN_CAPACITY_RATIO, (
        f"with the tail buffer at its {MAX_SUPPORTED_SLOTS}-slot ceiling of "
        f"{at_ceiling / 2**20:,.0f} MiB the NVFP4 cache is worth {worst:.4f}x "
        f"the BF16 one per byte, below the {MIN_TOKEN_CAPACITY_RATIO}x floor "
        f"the gross ratio cleared at {gross:.4f}"
    )
