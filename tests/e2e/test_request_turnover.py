"""Two hundred requests through one engine, and the second hundred agree.

Nothing in this package is reset between requests. A tail slot keeps the tokens
its last tenant left there, the slot table keeps its keys and its clock, and the
FP4 blocks keep whatever the previous occupant wrote until vLLM hands the block
to somebody else. That is deliberate — reclaiming eagerly is what loses a tail —
but it means every request runs on state the requests before it left behind, and
a mistake in that state does not announce itself. It comes back as an answer
that is slightly worse than it should have been.

So the same hundred questions are asked twice, as requests 1-100 and again as
requests 101-200, and the two passes have to produce the same token ids. By the
second pass every slot has changed hands many times and every block has been
recycled, so if any of that leaks the two passes disagree. Comparing accuracy
instead would not do: a hundred questions with different difficulty in each half
cannot tell a two-point drop from a two-point coincidence, and token identity
has no such noise floor.

Identity between two identically scheduled passes is a strong claim to make of
an engine, and it is only made here because it was already observed: the BF16
arms of ``test_vllm_integration.py`` agree token for token across two different
attention backends. Two passes through one engine ask for less than that.

An accuracy floor comes along for the ride, since two passes could agree on
nonsense.

Requires ``NVFP4_RUN_VLLM_E2E=1`` and access to ``openai/gsm8k``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
- ``NVFP4_TURNOVER_N``: questions per pass. Two passes, so requests are twice
  this.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)
QUESTIONS = int(os.environ.get("NVFP4_TURNOVER_N", "100"))

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8
MAX_NEW_TOKENS = 256
MIN_ACCURACY = 0.60


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass
class Turnover:
    first: list[tuple[int, ...]]
    second: list[tuple[int, ...]]
    accuracy: float
    error_code: int
    handovers_per_slot: list[int]
    num_slots: int

    @property
    def requests(self) -> int:
        return len(self.first) + len(self.second)


def _integration_module():
    """The GSM8K prompt builder and scorer, loaded by path.

    ``tests/`` has no package structure, and copying the scorer here would let
    the two files disagree about what a correct answer is.
    """
    path = Path(__file__).resolve().parent / "test_vllm_integration.py"
    spec = importlib.util.spec_from_file_location("_integration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def turnover() -> Turnover:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import gc

    from vllm import LLM, SamplingParams

    from nvfp4_vllm.control import ControlPlane

    integration = _integration_module()
    references, prompts = integration._load_gsm8k(QUESTIONS)

    held: dict[str, ControlPlane] = {}
    # A slot changes hands when its key changes. Counted on the device against
    # a shadow copy, because the alternative — reading the table each step —
    # is the host synchronization the whole read path is built to avoid, and
    # it would also perturb what this file is trying to observe.
    state: dict[str, torch.Tensor] = {}
    original_prepare = ControlPlane.prepare

    def watched_prepare(self, *args, **kwargs):
        outputs = original_prepare(self, *args, **kwargs)
        held["plane"] = self
        previous = state.get("keys")
        if previous is None:
            state["keys"] = self.slot_keys.clone()
            state["handovers"] = torch.zeros_like(self.slot_keys)
        else:
            state["handovers"] += (self.slot_keys != previous).to(torch.int32)
            previous.copy_(self.slot_keys)
        return outputs

    ControlPlane.prepare = watched_prepare
    try:
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            kv_cache_dtype="nvfp4",
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_MODEL_LEN * 4,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            block_size=PAGE_SIZE,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            attention_config={"backend": "CUSTOM"},
        )
        try:
            sampling = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=MAX_NEW_TOKENS
            )

            def run() -> list:
                return llm.generate(prompts, sampling, use_tqdm=False)

            first = run()
            second = run()
            plane = held["plane"]
            yield Turnover(
                first=[tuple(o.outputs[0].token_ids) for o in first],
                second=[tuple(o.outputs[0].token_ids) for o in second],
                accuracy=integration._score(
                    references,
                    [
                        integration.Completion(
                            token_ids=tuple(o.outputs[0].token_ids),
                            text=o.outputs[0].text,
                        )
                        for o in second
                    ],
                ).accuracy,
                # Read once, at the end. The flags are sticky for the life of
                # the engine, so one read covers all two hundred requests.
                error_code=int(plane.error_code.item()),
                handovers_per_slot=state["handovers"].tolist(),
                num_slots=plane.num_slots,
            )
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ControlPlane.prepare = original_prepare


def test_the_second_hundred_answers_exactly_like_the_first(turnover: Turnover):
    """The gate. Nothing a request leaves behind reaches the next one."""
    differing = [
        index
        for index, (a, b) in enumerate(zip(turnover.first, turnover.second))
        if a != b
    ]
    assert not differing, (
        f"{len(differing)} of {len(turnover.first)} questions answered "
        f"differently the second time through the same engine, first at index "
        f"{differing[0]}. The engine carried something across requests.\n"
        f"  first:  {turnover.first[differing[0]][:24]}\n"
        f"  second: {turnover.second[differing[0]][:24]}"
    )


def test_the_run_was_worth_comparing(turnover: Turnover):
    """Two passes agreeing on nonsense would satisfy the test above."""
    assert turnover.requests == 2 * QUESTIONS
    assert turnover.accuracy >= MIN_ACCURACY, (
        f"the engine scored {turnover.accuracy:.4f} on GSM8K, below the "
        f"{MIN_ACCURACY} floor, so agreement between the passes says only "
        "that it was consistently wrong"
    )


def test_the_slot_table_reported_nothing(turnover: Turnover):
    """Two hundred requests through eight slots, with no invariant broken.

    ``error_code`` is sticky, so this one read covers every step of the run.
    Its flags are the ones reclamation would trip: a row that came back to
    find its tail gone, a slot whose recorded length does not continue, a
    needy row with nowhere to go.
    """
    from nvfp4_vllm.control import ERROR_NAMES

    named = [text for bit, text in ERROR_NAMES.items() if turnover.error_code & bit]
    assert turnover.error_code == 0, (
        f"the slot table reported error_code {turnover.error_code} over "
        f"{turnover.requests} requests: {'; '.join(named)}"
    )


def test_every_slot_changed_hands_many_times(turnover: Turnover):
    """Otherwise the run never reused a slot and proves nothing about reuse.

    Two hundred requests through eight slots is roughly twenty-five tenants
    each. The floor is well under that, because the scheduler is free to
    distribute them unevenly; what it rules out is a slot that sat idle while
    the others did the work.
    """
    per_slot = turnover.handovers_per_slot
    assert len(per_slot) == turnover.num_slots
    floor = turnover.requests // (4 * turnover.num_slots)
    assert min(per_slot) >= floor, (
        f"handovers per slot were {per_slot} over {turnover.requests} "
        f"requests; every slot was expected to take at least {floor} tenants"
    )
