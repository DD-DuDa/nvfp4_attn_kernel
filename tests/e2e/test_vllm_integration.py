"""Opt-in end-to-end coverage for the vendored vLLM integration."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = REPOSITORY_ROOT / "third_party" / "vllm"


def _require_vllm_fork():
    if not (VLLM_ROOT / "vllm" / "__init__.py").is_file():
        pytest.fail(
            "the vLLM submodule is not initialized; run "
            "`git submodule update --init --recursive`"
        )

    try:
        import vllm
    except ImportError:
        pytest.fail(
            "the vendored vLLM fork is not installed; install "
            "`third_party/vllm` in the test environment"
        )

    source = Path(vllm.__file__).resolve()
    if not source.is_relative_to(VLLM_ROOT):
        pytest.fail(
            "tests must import the vendored vLLM fork, but imported "
            f"{source}"
        )
    return vllm


def _require_sm100():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")
    return torch


def _long_prompt() -> str:
    # The standalone kernel is a decode kernel. Exceed one 128-token page so
    # the test cannot pass through a residual-only BF16 attention path.
    context = (
        "Remember this fact: Paris is the capital of France. "
        * 24
    )
    return (
        context
        + "\nQuestion: What is the capital of France? "
        "Answer with only the city name:"
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_vllm_llama_generates_with_standalone_fp4_decode(monkeypatch):
    torch = _require_sm100()

    # Keep the worker in this process so the public-kernel spy observes every
    # layer invocation and teardown is deterministic.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import nvfp4_decode_kernel

    original_decode = nvfp4_decode_kernel.fp4_decode
    observed_seqused_fp4 = []

    def traced_fp4_decode(*args, **kwargs):
        seqused_fp4 = kwargs.get("seqused_fp4")
        if seqused_fp4 is None and len(args) >= 7:
            seqused_fp4 = args[6]
        assert seqused_fp4 is not None
        observed_seqused_fp4.append(seqused_fp4.detach().clone())
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(
        nvfp4_decode_kernel,
        "fp4_decode",
        traced_fp4_decode,
    )

    _require_vllm_fork()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        block_size=128,
        attention_config={
            "backend": "FLASH_ATTN",
            "nvfp4_kv_mode": "unified_v2",
        },
    )
    try:
        prompt = _long_prompt()
        outputs = llm.generate(
            [prompt],
            SamplingParams(max_tokens=8, temperature=0.0),
            use_tqdm=False,
        )

        assert len(outputs) == 1
        assert len(outputs[0].prompt_token_ids) > 128
        assert outputs[0].outputs
        assert outputs[0].outputs[0].token_ids
        generated_text = outputs[0].outputs[0].text.strip().lower()
        assert "paris" in generated_text, (
            f"expected a reasonable factual answer, got {generated_text!r}"
        )
        assert observed_seqused_fp4, (
            "vLLM generation did not call nvfp4_decode_kernel.fp4_decode"
        )
        assert any(
            bool(torch.any(lengths > 0).item())
            for lengths in observed_seqused_fp4
        ), "vLLM never dispatched a row with an FP4 prefix"
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
