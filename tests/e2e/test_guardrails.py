"""Which engine configurations the NVFP4 path refuses, and that it refuses them.

Two halves that only mean something together. The matrix says which
configurations ``check_supported`` rejects; ``test_engine_refuses_more_rows_
than_tail_slots`` says a live engine reaches it at all. Without the second the
matrix could pass in full while nothing ever calls the function.

The matrix calls ``check_supported`` directly instead of building an engine per
case. Twenty-one engines cost several minutes each and the 8B weights, and the
half of the matrix that asserts a configuration is *accepted* would have to
prove it by starting an engine successfully — including a two-GPU one for the
pipeline parallel case.

Only the wiring test needs a GPU and a model, so only it is gated behind
``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os

import pytest
from vllm.config import CUDAGraphMode, VllmConfig

from nvfp4_vllm.guards import MAX_SLOTS, UnsupportedConfigError, check_supported


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096


def _config(cache_dtype: str = "nvfp4", **overrides) -> VllmConfig:
    """A configuration the NVFP4 path accepts, with fields overridden.

    Overrides are applied by attribute name against whichever sub-config
    declares them, falling back to the top-level config, so a typo raises
    instead of silently testing nothing.
    """
    config = VllmConfig()
    config.cache_config.cache_dtype = cache_dtype
    config.cache_config.enable_prefix_caching = False
    config.cache_config.block_size = PAGE_SIZE
    config.scheduler_config.max_num_seqs = MAX_SLOTS
    config.scheduler_config.enable_chunked_prefill = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    for name, value in overrides.items():
        for section in (
            config.cache_config,
            config.scheduler_config,
            config.parallel_config,
            config.compilation_config,
            config,
        ):
            if hasattr(section, name):
                setattr(section, name, value)
                break
        else:
            raise AttributeError(f"no sub-config declares {name!r}")
    return config


REJECTED = {
    "max_num_seqs": {"max_num_seqs": MAX_SLOTS + 1},
    "prefix_caching": {"enable_prefix_caching": True},
    "chunked_prefill": {"enable_chunked_prefill": True},
    "long_prefill_threshold": {"long_prefill_token_threshold": 512},
    "speculative_decoding": {"speculative_config": object()},
    "kv_offloading": {"kv_offloading_size": 8.0},
    # use_ubatching is a property over these two, so both have to be refused.
    "dual_batch_overlap": {"enable_dbo": True},
    "ubatch_size": {"ubatch_size": 2},
    "pipeline_parallel": {"pipeline_parallel_size": 2},
    "cudagraphs": {"cudagraph_mode": CUDAGraphMode.PIECEWISE},
}


def test_default_configuration_is_accepted():
    check_supported(_config())


@pytest.mark.parametrize("overrides", REJECTED.values(), ids=list(REJECTED))
def test_unsupported_configuration_is_rejected(overrides):
    with pytest.raises(UnsupportedConfigError):
        check_supported(_config(**overrides))


@pytest.mark.parametrize("overrides", REJECTED.values(), ids=list(REJECTED))
def test_bf16_cache_is_left_alone(overrides):
    # With a BF16 cache the backend is a FlashAttention pass-through, so none
    # of the machinery these constraints protect exists to be protected.
    check_supported(_config(cache_dtype="auto", **overrides))


def test_max_num_seqs_at_the_limit_is_accepted():
    check_supported(_config(max_num_seqs=MAX_SLOTS))


def test_rejection_names_the_offending_setting():
    with pytest.raises(UnsupportedConfigError, match="max_num_seqs"):
        check_supported(_config(max_num_seqs=MAX_SLOTS + 1))


def _require_sm100() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _causes(error: BaseException):
    """The exception and everything it was raised from."""
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_engine_refuses_more_rows_than_tail_slots(monkeypatch):
    """A real engine reaches the guardrails.

    ``max_num_seqs`` is the trigger because vLLM is happy to run at any batch
    width, so the rejection can only come from us. The positive control is
    ``test_kv_cache_memory.py``, which starts the same NVFP4 engine within the
    slot limit.
    """
    _require_sm100()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM

    try:
        with pytest.raises(BaseException) as caught:
            LLM(
                model=MODEL,
                dtype="bfloat16",
                kv_cache_dtype="nvfp4",
                tensor_parallel_size=1,
                max_model_len=MAX_MODEL_LEN,
                max_num_seqs=MAX_SLOTS + 8,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                block_size=PAGE_SIZE,
                enable_prefix_caching=False,
                attention_config={"backend": "CUSTOM"},
            )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    # The engine wraps startup failures, so the guardrail is looked for in the
    # chain rather than in the exception the caller sees.
    guardrail = [
        error
        for error in _causes(caught.value)
        if isinstance(error, UnsupportedConfigError)
    ]
    assert guardrail, (
        "the engine failed to start, but not on the NVFP4 guardrail:\n"
        + "\n".join(f"  {type(e).__name__}: {e}" for e in _causes(caught.value))
    )
    assert "max_num_seqs" in str(guardrail[0])
