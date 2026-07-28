"""Opt-in end-to-end coverage for the vendored vLLM integration.

``test_gsm8k_subset_accuracy_tracks_bf16`` proves that the FP4 KV cache still
answers correctly, by scoring a GSM8K subset against a BF16 baseline taken from
the same engine configuration in the same process.

Attribution and accuracy are both required. Accuracy alone cannot distinguish
a correct FP4 path from a silent fallback to BF16 attention, and attribution
alone cannot distinguish a running kernel from a numerically broken one.

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
from dataclasses import dataclass, field
from pathlib import Path

import pytest


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = REPOSITORY_ROOT / "third_party" / "vllm"

PAGE_SIZE = 128

BF16_ATTENTION_CONFIG = {"backend": "FLASH_ATTN"}
NVFP4_ATTENTION_CONFIG = {
    "backend": "FLASH_ATTN",
    "nvfp4_kv_mode": "tagged",
}

GSM8K_NUM_EXAMPLES = int(os.environ.get("NVFP4_GSM8K_N", "32"))
GSM8K_NUM_FEWSHOT = 5
GSM8K_MAX_NEW_TOKENS = 256
GSM8K_MAX_MODEL_LEN = 4096
GSM8K_MAX_NUM_SEQS = int(os.environ.get("NVFP4_GSM8K_MAX_NUM_SEQS", "1"))
# Wide enough that every prompt prefills in one execution. Unaligned chunked
# prefill is a separate contract and is deliberately not exercised here.
GSM8K_MAX_NUM_BATCHED_TOKENS = 16384

# Llama-3.1-8B-Instruct scores about 0.78 on this harness with a BF16 cache and
# about 0.75 with an NVFP4 cache, and greedy GSM8K shifts by up to two samples
# at N=32. A 0.125 budget covers that band while still failing on the class of
# integration faults that drop accuracy to 0.6 or below.
GSM8K_MAX_ACCURACY_DROP = 0.125
GSM8K_MIN_ACCURACY = 0.60

# Reading a device tensor synchronizes, so only the first few decode calls are
# inspected; later calls only advance the counter.
DECODE_CALLS_INSPECTED = 8


@dataclass(frozen=True)
class DecodeCall:
    """One observed call into the standalone decode kernel."""

    rows: int
    max_seqused_fp4: int
    max_seqused_residual: int


@dataclass
class DecodeSpy:
    calls: int = 0
    inspected: list[DecodeCall] = field(default_factory=list)

    @property
    def saw_fp4_prefix(self) -> bool:
        return any(call.max_seqused_fp4 > 0 for call in self.inspected)


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


def _require_vllm_fork() -> None:
    if not (VLLM_ROOT / "vllm" / "__init__.py").is_file():
        pytest.fail(
            "the vLLM submodule is not initialized; run "
            "`git submodule update --init third_party/vllm`"
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


def _require_sm100() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _spy_on_fp4_decode(monkeypatch) -> DecodeSpy:
    """Record every decode call the engine makes.

    ``interface.fp4_decode`` resolves its implementation on each call, so
    patching the implementation also catches callers that bound the public
    ``fp4_decode`` name at import time.
    """
    from nvfp4_decode_kernel import _kernel

    spy = DecodeSpy()
    original = _kernel.fp4_decode_impl

    def traced(**kwargs):
        if spy.calls < DECODE_CALLS_INSPECTED:
            seqused_fp4 = kwargs["seqused_fp4"]
            seqused_residual = kwargs.get("seqused_residual")
            spy.inspected.append(
                DecodeCall(
                    rows=seqused_fp4.numel(),
                    max_seqused_fp4=int(seqused_fp4.max()),
                    max_seqused_residual=(
                        0
                        if seqused_residual is None
                        else int(seqused_residual.max())
                    ),
                )
            )
        spy.calls += 1
        return original(**kwargs)

    monkeypatch.setattr(_kernel, "fp4_decode_impl", traced)
    return spy


def _run_gsm8k(attention_config: dict, prompts: list[str]) -> list[str]:
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
        # Few-shot prompts share a prefix. Prefix caching would skip prefill
        # for the shared tokens, so the FP4 write path would only ever see the
        # per-example delta.
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
        return [output.outputs[0].text for output in outputs]
    finally:
        # Both backends run back to back in one process, so this engine must
        # release its KV cache before the next one is built.
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


def _score(references: list[str], completions: list[str]) -> Score:
    assert len(references) == len(completions)

    correct = 0
    mismatches = []
    for reference, completion in zip(references, completions):
        prediction = _predicted_answer(completion)
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


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_gsm8k_subset_accuracy_tracks_bf16(monkeypatch):
    _require_sm100()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _require_vllm_fork()

    references, prompts = _load_gsm8k(GSM8K_NUM_EXAMPLES)
    spy = _spy_on_fp4_decode(monkeypatch)

    bf16 = _score(references, _run_gsm8k(BF16_ATTENTION_CONFIG, prompts))
    assert spy.calls == 0, (
        f"the BF16 baseline reached the decode kernel {spy.calls} times; it "
        "must not share the FP4 path"
    )
    assert bf16.accuracy >= GSM8K_MIN_ACCURACY, (
        "the BF16 baseline is too weak to gate against: "
        f"{bf16.summary('BF16')}. Check the model and the GSM8K split.\n"
        f"{_mismatch_report(bf16)}"
    )

    nvfp4 = _score(references, _run_gsm8k(NVFP4_ATTENTION_CONFIG, prompts))
    print(f"GSM8K {bf16.summary('BF16')} | {nvfp4.summary('NVFP4')}")

    assert spy.calls > 0, (
        "the NVFP4 configuration never reached the standalone decode kernel, "
        "so the accuracy above only measures a fallback path"
    )
    assert spy.saw_fp4_prefix, (
        "every inspected decode call had an empty FP4 prefix, so the run only "
        "exercised the BF16 residual path"
    )
    assert nvfp4.accuracy >= GSM8K_MIN_ACCURACY, (
        f"{nvfp4.summary('NVFP4')} is below the "
        f"{GSM8K_MIN_ACCURACY:.2f} floor.\n{_mismatch_report(nvfp4)}"
    )

    drop = bf16.accuracy - nvfp4.accuracy
    assert drop <= GSM8K_MAX_ACCURACY_DROP, (
        f"NVFP4 lost {drop * 100:.2f} pp against BF16 "
        f"({bf16.summary('BF16')}, {nvfp4.summary('NVFP4')}), more than the "
        f"{GSM8K_MAX_ACCURACY_DROP * 100:.2f} pp budget.\n"
        f"{_mismatch_report(nvfp4)}"
    )
