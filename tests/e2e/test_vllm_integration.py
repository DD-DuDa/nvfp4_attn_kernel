"""Opt-in end-to-end check of the CUSTOM backend, on a BF16 and an FP4 cache.

The same GSM8K subset is scored three times. ``FLASH_ATTN`` and ``CUSTOM``
both run a BF16 KV cache with no quantization anywhere, so they run identical
arithmetic and greedy decoding must produce the same token ids exactly. The
BF16 arm additionally has to clear an absolute accuracy floor, or two arms
agreeing on garbage would satisfy the equality check.

The third arm is ``CUSTOM`` over a real NVFP4 cache, which cannot be held to
token identity: it is 4-bit. It is held to answering as well, within a margin.
At 256 new tokens every request fills several pages, so this is the only gate
here that exercises the write path, the FP4 read path and promotion together
on a workload nobody wrote for them.

Be honest about its power. The margin is a quarter of the way to chance at
N=32, which is four questions, and the two arms answer the same questions, so
the standard deviation of the difference is around 0.06. It is a catastrophe
alarm — it catches a page written to the wrong block, a layer written into
another layer, a page of history dropped — and it does not catch a gradual
five-point decay. What guards against silently losing a page is the byte
equality in ``test_promotion.py``, not this.

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
# How far the FP4 cache may fall behind the BF16 one. Four questions at N=32.
GSM8K_MAX_ACCURACY_DROP = 0.125


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


def _run_gsm8k(
    attention_config: dict,
    prompts: list[str],
    kv_cache_dtype: str = "auto",
) -> list[Completion]:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
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
        # The FP4 arm cannot take a split prompt, since the second chunk would
        # resume mid-page. Set on every arm rather than only that one: chunking
        # changes how the batch is composed, and arms composed differently are
        # not comparable either for token identity or for accuracy.
        enable_chunked_prefill=False,
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


@dataclass(frozen=True)
class Arm:
    label: str
    completions: tuple[Completion, ...]
    score: Score


@pytest.fixture(scope="module")
def arms() -> tuple[Arm, Arm, Arm]:
    """The three arms, scored once for both tests below.

    Three 8B engines is most of the cost of this file, and the BF16 baseline
    is the comparison both gates are against, so it is built once rather than
    once per gate.
    """
    _require_sm100()
    # In-process engine core, so each arm's teardown is deterministic.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    references, prompts = _load_gsm8k(GSM8K_NUM_EXAMPLES)
    plan = (
        ("FLASH_ATTN", BF16_ATTENTION_CONFIG, "auto"),
        ("CUSTOM", CUSTOM_ATTENTION_CONFIG, "auto"),
        ("CUSTOM/nvfp4", CUSTOM_ATTENTION_CONFIG, "nvfp4"),
    )
    scored = []
    for label, attention_config, kv_cache_dtype in plan:
        completions = _run_gsm8k(attention_config, prompts, kv_cache_dtype)
        scored.append(
            Arm(
                label=label,
                completions=tuple(completions),
                score=_score(references, completions),
            )
        )
    print(
        "\nGSM8K "
        + " | ".join(arm.score.summary(arm.label) for arm in scored)
    )
    return tuple(scored)


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_custom_backend_matches_flash_attn(arms):
    """With a BF16 cache the backend changes nothing, so nothing may change."""
    flash, custom, _ = arms

    assert flash.score.accuracy >= GSM8K_MIN_ACCURACY, (
        "the BF16 baseline is too weak to gate against: "
        f"{flash.score.summary(flash.label)}. Check the model and the GSM8K "
        f"split.\n{_mismatch_report(flash.score)}"
    )

    divergences = [
        _divergence_report(index, left, right)
        for index, (left, right) in enumerate(
            zip(flash.completions, custom.completions)
        )
        if left.token_ids != right.token_ids
    ]
    assert not divergences, (
        f"{len(divergences)} of {len(flash.completions)} requests decoded "
        "differently under the CUSTOM backend, which means it is not a "
        "pass-through or the two engines were scheduled differently.\n"
        f"{flash.score.summary(flash.label)} | "
        f"{custom.score.summary(custom.label)}\n"
        + "\n".join(divergences[:3])
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_the_fp4_cache_answers_about_as_well_as_the_bf16_one(arms):
    """The debt the write path opened when it stopped being readable.

    Every other test of the FP4 path compares against something that was
    computed the same way — the same quantizer, the same oracle. This one is
    the only place the whole path is asked whether the model still works, on a
    workload that was not written for it.
    """
    flash, _, nvfp4 = arms
    drop = flash.score.accuracy - nvfp4.score.accuracy

    assert drop <= GSM8K_MAX_ACCURACY_DROP, (
        f"the FP4 cache lost {drop:.4f} against BF16, over the "
        f"{GSM8K_MAX_ACCURACY_DROP} allowed.\n"
        f"{flash.score.summary(flash.label)} | "
        f"{nvfp4.score.summary(nvfp4.label)}\n"
        f"{_mismatch_report(nvfp4.score)}"
    )
    assert nvfp4.score.accuracy >= GSM8K_MIN_ACCURACY, (
        "the FP4 arm is below the absolute floor, so a baseline that also "
        f"fell would have hidden it: {nvfp4.score.summary(nvfp4.label)}\n"
        f"{_mismatch_report(nvfp4.score)}"
    )
