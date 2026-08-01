"""The point of the whole project, measured as something a user would notice.

``test_kv_cache_memory.py`` shows the FP4 cache holds 3.56 times the tokens.
That is the ratio, and it is the honest headline, but it is a number about
blocks. This file spends it: at a long enough context, the extra tokens are the
difference between running a batch and queueing half of it.

The setup is chosen so the two arms cannot both win. At 65536 tokens of context
the BF16 cache on this machine holds about eighteen sequences and the FP4 cache
about sixty-four, so a batch of thirty-two sits between them. The BF16 engine
has to preempt; ours does not. Both are given the same ``max_num_seqs``, so what
differs is what the cache can hold rather than what the scheduler was allowed to
try.

Nothing here is a microbenchmark. It is the first time this path runs a
sequence long enough for a block table to have five hundred columns and for a
page offset into the FP4 cache to pass 2^31 — the wrap that ``175761c`` fixed
and that no other test reaches. So the outputs are checked too: a run that
preempts nothing and returns nothing usable has not demonstrated anything.

Requires ``NVFP4_RUN_VLLM_E2E=1``, and enough free memory for two engines in
sequence at high utilization.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import pytest
import torch


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
CONTEXT = 65536
CONCURRENCY = 32
PROMPT_TOKENS = CONTEXT - 512
GENERATED_TOKENS = 8
GPU_MEMORY_UTILIZATION = 0.9


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass(frozen=True)
class Arm:
    label: str
    cache_tokens: int
    finite_outputs: int
    total_outputs: int
    text_sample: str

    @property
    def sequences_at_context(self) -> float:
        return self.cache_tokens / CONTEXT

    def summary(self) -> str:
        return (
            f"{self.label}: {self.cache_tokens:,} cache tokens = "
            f"{self.sequences_at_context:.1f} sequences at {CONTEXT:,}"
        )


def _model_runner(llm):
    core = llm.llm_engine.engine_core
    while hasattr(core, "engine_core"):
        core = core.engine_core
    return core.model_executor.driver_worker.worker.model_runner


def _cache_tokens(llm) -> int:
    runner = _model_runner(llm)
    spec = next(iter(runner.get_kv_cache_spec().values()))
    cache = runner.kv_caches[0]
    layer_bytes = cache.numel() * cache.element_size()
    return (layer_bytes // spec.page_size_bytes) * PAGE_SIZE


def _prompt(row: int) -> list[int]:
    # Token ids, so the prompt is exactly as long as intended. Rows differ so
    # that no two sequences produce the same attention pattern.
    return [1000 + (row * 31 + i * 7) % 20000 for i in range(PROMPT_TOKENS)]


def _run_arm(label: str, kv_cache_dtype: str) -> Arm:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=CONTEXT,
        max_num_seqs=CONCURRENCY,
        # Chunked prefill is refused by the NVFP4 path, and vLLM then requires
        # a budget wide enough that no prompt is split. Both arms take the same
        # setting so neither is measuring a scheduler difference.
        max_num_batched_tokens=CONTEXT,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "CUSTOM"},
    )
    try:
        tokens = _cache_tokens(llm)
        outputs = llm.generate(
            [
                TokensPrompt(prompt_token_ids=_prompt(row))
                for row in range(CONCURRENCY)
            ],
            SamplingParams(
                temperature=0.0, max_tokens=GENERATED_TOKENS, ignore_eos=True
            ),
            use_tqdm=False,
        )
        produced = [
            output.outputs[0]
            for output in outputs
            if output.outputs and output.outputs[0].token_ids
        ]
        return Arm(
            label=label,
            cache_tokens=tokens,
            finite_outputs=sum(
                1 for o in produced if len(o.token_ids) == GENERATED_TOKENS
            ),
            total_outputs=len(outputs),
            text_sample=produced[0].text if produced else "",
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def arms() -> dict[str, Arm]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    result = {
        "nvfp4": _run_arm("NVFP4", "nvfp4"),
        "bf16": _run_arm("BF16", "auto"),
    }
    print("\n" + "\n".join(arm.summary() for arm in result.values()))
    return result


def test_the_fp4_cache_holds_the_whole_batch_and_the_bf16_one_does_not(arms):
    """The claim, at the width a scheduler would feel it.

    Stated as capacity rather than as an observed preemption count, because
    preemption is the scheduler's decision and it has more than one way to
    make room. What the cache can hold is the property this project changed.
    """
    nvfp4, bf16 = arms["nvfp4"], arms["bf16"]
    assert bf16.sequences_at_context < CONCURRENCY, (
        f"the BF16 cache holds {bf16.sequences_at_context:.1f} sequences at "
        f"{CONTEXT:,} tokens, which is already the whole batch of "
        f"{CONCURRENCY}. This comparison needs a context where BF16 runs out."
    )
    assert nvfp4.sequences_at_context >= CONCURRENCY, (
        f"the NVFP4 cache holds {nvfp4.sequences_at_context:.1f} sequences at "
        f"{CONTEXT:,} tokens, short of the batch of {CONCURRENCY}.\n"
        f"{nvfp4.summary()}\n{bf16.summary()}"
    )


def test_every_sequence_came_back_from_a_context_this_long(arms):
    """A page offset past 2^31, a block table five hundred columns wide.

    The wrap this reaches was a real bug in the quantizer's addressing, found
    by a spike and fixed in ``175761c``; the kernel test that pins it builds a
    two gibibyte allocation by hand. This is the same arithmetic arrived at
    the way a user would arrive at it.
    """
    nvfp4 = arms["nvfp4"]
    assert nvfp4.finite_outputs == nvfp4.total_outputs == CONCURRENCY, (
        f"{nvfp4.finite_outputs} of {nvfp4.total_outputs} sequences returned "
        f"{GENERATED_TOKENS} tokens at a context of {CONTEXT:,}"
    )
    assert nvfp4.text_sample.strip(), (
        "the FP4 arm returned empty text, so the run says nothing about "
        "whether a context this long is served correctly"
    )
