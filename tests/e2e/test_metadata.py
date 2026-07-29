"""The attention metadata the NVFP4 path hands to the layers.

Two halves. The first checks the one thing ``build()`` does beyond calling
FlashAttention's builder and launching the control plane: it splices the
control plane's answer onto FlashAttention's metadata without losing a field.
That needs neither a GPU nor a model.

The second half is one live engine. It exists for a fact no unit test can
establish: that ``build()`` runs exactly once per scheduler step. Everything
about the slot table depends on that — the LRU ordering, the continuity check,
the tail lengths — and vLLM decides it, not us. Generating a known number of
tokens and watching the control plane's step counter move is the direct
measurement.

The engine half is skipped unless ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os
from dataclasses import fields

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from nvfp4_vllm.control import ControlOutputs
from nvfp4_vllm.metadata import NVFP4Metadata


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8


# --- field splicing --------------------------------------------------------


class _Marker:
    """Stands in for a field value, distinguishable from every other one."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


def _markers(cls) -> dict:
    return {field.name: _Marker(field.name) for field in fields(cls)}


# Deliberately not spliced: reading error_code costs a host synchronization and
# nothing inside a step acts on it. A new output here is a decision, not an
# omission, which is what the field-set assertion below forces.
NOT_SPLICED = {"error_code"}

# The other source of appended fields: the page work table, which build()
# produces alongside the control plane's outputs.
WORK_TABLE = ("source_tokens", "destination_pages")


def _splice():
    """One spliced object plus the two things it was spliced from."""
    base = FlashAttentionMetadata(**_markers(FlashAttentionMetadata))
    outputs = ControlOutputs(**_markers(ControlOutputs))
    work_table = tuple(_Marker(name) for name in WORK_TABLE)
    return NVFP4Metadata.from_flash(base, outputs, work_table), base, outputs


def test_the_splice_keeps_every_flash_attention_field():
    # Values are markers rather than tensors because this is plumbing: a field
    # that is dropped, defaulted, or filled from the wrong source shows up as
    # an identity mismatch whatever its declared type.
    spliced, base, _ = _splice()
    for field in fields(FlashAttentionMetadata):
        assert getattr(spliced, field.name) is getattr(base, field.name), field.name


def test_the_splice_keeps_every_control_plane_output():
    spliced, _, outputs = _splice()
    added = {field.name for field in fields(NVFP4Metadata)} - {
        field.name for field in fields(FlashAttentionMetadata)
    }
    from_control = {field.name for field in fields(ControlOutputs)} - NOT_SPLICED
    assert added == from_control | set(WORK_TABLE)
    for name in from_control:
        assert getattr(spliced, name) is getattr(outputs, name), name
    for name in WORK_TABLE:
        assert getattr(spliced, name).name == name


def test_the_result_is_still_flash_attention_metadata():
    # Until the decode kernel lands, every layer still runs FlashAttention and
    # reads these objects through the base class.
    spliced, _, _ = _splice()
    assert isinstance(spliced, FlashAttentionMetadata)


# --- one build per step ----------------------------------------------------


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _control_plane(llm):
    """The one control plane belonging to the engine's attention group."""
    core = llm.llm_engine.engine_core
    while hasattr(core, "engine_core"):
        core = core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    groups = [group for per_cache in runner.attn_groups for group in per_cache]
    assert len(groups) == 1, (
        f"{len(groups)} attention groups, each with its own control plane; the "
        "slot table assumes one"
    )
    return groups[0].get_metadata_builder(0).plane


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_the_control_plane_advances_once_per_decode_step(monkeypatch):
    _require_sm100()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "CUSTOM"},
    )
    try:
        plane = _control_plane(llm)
        # Sizing the KV cache and warming the kernels both push batches through
        # build() before any request exists, so the counter does not start at
        # zero. What has to hold is that those batches pass through the slot
        # table cleanly: the dummy batch keys on the null block and must be read
        # as no requests rather than as a batch of eight sharing one block.
        assert int(plane.error_code.item()) == 0, (
            f"engine startup left error_code {int(plane.error_code.item())} "
            "before a single request was served"
        )

        def advance(prompts: list[str], tokens: int) -> int:
            before = int(plane.step.item())
            llm.generate(
                prompts,
                SamplingParams(
                    max_tokens=tokens, ignore_eos=True, temperature=0.0
                ),
            )
            return int(plane.step.item()) - before

        # One prefill step, which produces the first token, then one step per
        # token after it.
        for tokens in (16, 4):
            advanced = advance(["The capital of France is"], tokens)
            assert advanced == tokens, (
                f"the control plane advanced {advanced} times over {tokens} "
                "scheduler steps. Every slot decision — the LRU ordering, the "
                "continuity check, the tail lengths — assumes these are the "
                "same number."
            )

        # Three requests in one batch still take one step each, which is what
        # separates a per-step counter from a per-sequence one.
        tokens = 12
        advanced = advance(
            [
                "The capital of France is",
                "The tallest mountain is",
                "In the year 1900",
            ],
            tokens,
        )
        assert advanced == tokens, (
            f"three concurrent requests advanced the control plane {advanced} "
            f"times over {tokens} scheduler steps"
        )

        assert int(plane.error_code.item()) == 0, (
            f"the control plane reported error_code "
            f"{int(plane.error_code.item())} over plain generation"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)
def test_a_bf16_engine_has_no_control_plane(monkeypatch):
    # The backend is chosen per engine, not per cache dtype, so the NVFP4
    # builder is also what a BF16 run gets. It has to stay a pass-through.
    _require_sm100()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        attention_config={"backend": "CUSTOM"},
    )
    try:
        assert _control_plane(llm) is None
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
