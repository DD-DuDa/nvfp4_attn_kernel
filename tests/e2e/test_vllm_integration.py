"""Opt-in end-to-end check that the CUSTOM backend is a pass-through.

Scores the same GSM8K subset on stock ``FLASH_ATTN`` and on ``CUSTOM``, both
with a BF16 KV cache and no quantization anywhere. The two arms run identical
arithmetic, so greedy decoding must produce the same token ids exactly. The
BF16 arm additionally has to clear an absolute accuracy floor, or two arms
agreeing on garbage would satisfy the equality check.

``CUSTOM`` resolves through the ``vllm.general_plugins`` entry point declared
in ``pyproject.toml``. The test must not register the backend itself, or that
declaration would stop being covered.

The test is skipped unless ``NVFP4_RUN_VLLM_E2E=1``, and it needs the
``datasets`` package and access to ``openai/gsm8k``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
- ``NVFP4_GSM8K_N``: number of GSM8K test examples to score.
- ``NVFP4_GSM8K_MAX_NUM_SEQS``: decode batch width.
"""

from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass

import pytest


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128

BF16_ATTENTION_CONFIG = {"backend": "FLASH_ATTN"}
CUSTOM_ATTENTION_CONFIG = {"backend": "CUSTOM"}

GSM8K_NUM_EXAMPLES = int(os.environ.get("NVFP4_GSM8K_N", "32"))
GSM8K_NUM_FEWSHOT = 5
GSM8K_MAX_NEW_TOKENS = 256
GSM8K_MAX_MODEL_LEN = 4096
# FlashAttention is not batch invariant, so both arms have to decode at the
# same batch width to be comparable.
GSM8K_MAX_NUM_SEQS = int(os.environ.get("NVFP4_GSM8K_MAX_NUM_SEQS", "8"))
# Wide enough that every prompt prefills in one execution.
GSM8K_MAX_NUM_BATCHED_TOKENS = 16384

# Llama-3.1-8B-Instruct scores about 0.78 here; the floor leaves room for the
# few-sample noise of greedy GSM8K at N=32.
GSM8K_MIN_ACCURACY = 0.60


@dataclass(frozen=True)
class Completion:
    token_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class Score:
    correct: int
    total: int
    mismatches: tuple[tuple[str, str], ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self, label: str) -> str:
        return (
            f"{label} {self.correct}/{self.total} "
            f"(accuracy {self.accuracy:.4f})"
        )


def _require_sm100() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _run_gsm8k(attention_config: dict, prompts: list[str]) -> list[Completion]:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=GSM8K_MAX_MODEL_LEN,
        max_num_seqs=GSM8K_MAX_NUM_SEQS,
        max_num_batched_tokens=GSM8K_MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        # The few-shot prompts share a prefix, and caching it would leave the
        # write path seeing only the per-example delta.
        enable_prefix_caching=False,
        attention_config=attention_config,
    )
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=GSM8K_MAX_NEW_TOKENS,
            ),
            use_tqdm=False,
        )
        return [
            Completion(
                token_ids=tuple(output.outputs[0].token_ids),
                text=output.outputs[0].text,
            )
            for output in outputs
        ]
    finally:
        # The next engine sizes its cache from free memory. Startup allocations
        # are behind a gc.freeze(); only shutdown() unfreezes them.
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _load_gsm8k(num_examples: int) -> tuple[list[str], list[str]]:
    """Build 5-shot prompts and their reference answers."""
    try:
        from datasets import load_dataset
    except ImportError:
        pytest.fail(
            "the GSM8K gate needs the `datasets` package; install the test "
            "extra to run it"
        )

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    prefix = "".join(
        f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
        for example in train.select(range(GSM8K_NUM_FEWSHOT))
    )
    examples = test.select(range(num_examples))
    prompts = [
        prefix + f"Question: {example['question']}\nAnswer:"
        for example in examples
    ]
    references = [_reference_answer(example["answer"]) for example in examples]
    return references, prompts


def _reference_answer(answer: str) -> str:
    """Return the final number that follows GSM8K's ``####`` marker."""
    match = re.search(r"####\s*([-\d,\.]+)", answer)
    return match.group(1).replace(",", "").strip() if match else ""


def _predicted_answer(text: str) -> str:
    """Return the last number the model emitted for the current question.

    A greedy 5-shot completion usually runs on into a new ``Question:`` block,
    so only the first block is scored.
    """
    block = text.split("Question:")[0]
    numbers = re.findall(r"-?[\d,]*\.?\d+", block)
    return numbers[-1].replace(",", "") if numbers else ""


def _same_number(reference: str, prediction: str) -> bool:
    if not reference or not prediction:
        return False
    try:
        return abs(float(reference) - float(prediction)) < 1e-6
    except ValueError:
        return reference == prediction


def _score(references: list[str], completions: list[Completion]) -> Score:
    assert len(references) == len(completions)

    correct = 0
    mismatches = []
    for reference, completion in zip(references, completions):
        prediction = _predicted_answer(completion.text)
        if _same_number(reference, prediction):
            correct += 1
        else:
            mismatches.append((reference, prediction))
    return Score(
        correct=correct,
        total=len(references),
        mismatches=tuple(mismatches),
    )


def _mismatch_report(score: Score, limit: int = 5) -> str:
    lines = [
        f"  expected {reference!r}, predicted {prediction!r}"
        for reference, prediction in score.mismatches[:limit]
    ]
    if len(score.mismatches) > limit:
        lines.append(f"  ... {len(score.mismatches) - limit} more")
    return "\n".join(lines)


def _divergence_report(
    index: int, baseline: Completion, candidate: Completion
) -> str:
    """Report a window around where two completions first differ."""
    position = next(
        (
            i
            for i, (left, right) in enumerate(
                zip(baseline.token_ids, candidate.token_ids)
            )
            if left != right
        ),
        min(len(baseline.token_ids), len(candidate.token_ids)),
    )
    start = max(0, position - 8)
    stop = position + 8
    return (
        f"request {index}: first difference at token {position} of "
        f"{len(baseline.token_ids)}/{len(candidate.token_ids)}\n"
        f"  FLASH_ATTN[{start}:{stop}] = "
        f"{list(baseline.token_ids[start:stop])}\n"
        f"  CUSTOM    [{start}:{stop}] = "
        f"{list(candidate.token_ids[start:stop])}"
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_custom_backend_matches_flash_attn(monkeypatch):
    _require_sm100()
    # In-process engine core, so the first arm's teardown is deterministic.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    references, prompts = _load_gsm8k(GSM8K_NUM_EXAMPLES)

    baseline = _run_gsm8k(BF16_ATTENTION_CONFIG, prompts)
    candidate = _run_gsm8k(CUSTOM_ATTENTION_CONFIG, prompts)

    bf16 = _score(references, baseline)
    custom = _score(references, candidate)
    print(f"GSM8K {bf16.summary('FLASH_ATTN')} | {custom.summary('CUSTOM')}")

    assert bf16.accuracy >= GSM8K_MIN_ACCURACY, (
        "the BF16 baseline is too weak to gate against: "
        f"{bf16.summary('FLASH_ATTN')}. Check the model and the GSM8K split.\n"
        f"{_mismatch_report(bf16)}"
    )

    divergences = [
        _divergence_report(index, left, right)
        for index, (left, right) in enumerate(zip(baseline, candidate))
        if left.token_ids != right.token_ids
    ]
    assert not divergences, (
        f"{len(divergences)} of {len(baseline)} requests decoded differently "
        "under the CUSTOM backend, which means it is not a pass-through or "
        "the two engines were scheduled differently.\n"
        f"{bf16.summary('FLASH_ATTN')} | {custom.summary('CUSTOM')}\n"
        + "\n".join(divergences[:3])
    )
