"""The attention metadata the NVFP4 path hands to the layers.

Three groups, in order of what they need to run. The first checks the one
thing ``build()`` does beyond calling FlashAttention's builder and launching
the control plane: it splices the control plane's answer onto FlashAttention's
metadata without losing a field. That, and the decode-prefix arithmetic after
it, need neither a GPU nor a model.

The second builds a metadata builder from a model configuration, which is as
far as the cache dtype has to travel to decide whether there is a control
plane at all.

The third is one live engine. It exists for a fact no unit test can establish:
that ``build()`` runs exactly once per scheduler step. Everything about the
slot table depends on that — the LRU ordering, the continuity check, the tail
lengths — and vLLM decides it, not us. Generating a known number of tokens and
watching the control plane's step counter move is the direct measurement.

Everything past the first group is skipped unless ``NVFP4_RUN_VLLM_E2E=1``.

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

# The other sources of appended fields: the page work table, and what build()
# derives once per step from the CPU-side batch description.
APPENDED = (
    "source_tokens",
    "destination_pages",
    "decode_prefix_rows",
    "decode_prefix_tokens",
    "prefill_query_start_loc",
    "decode_page_columns",
)


def _splice():
    """One spliced object plus the two things it was spliced from."""
    base = FlashAttentionMetadata(**_markers(FlashAttentionMetadata))
    outputs = ControlOutputs(**_markers(ControlOutputs))
    spliced = NVFP4Metadata.from_flash(
        base, outputs, **{name: _Marker(name) for name in APPENDED}
    )
    return spliced, base, outputs


def test_the_splice_keeps_every_flash_attention_field():
    # Values are markers rather than tensors because this is plumbing: a field
    # that is dropped, defaulted, or filled from the wrong source shows up as
    # an identity mismatch whatever its declared type.
    spliced, base, _ = _splice()
    for field in fields(FlashAttentionMetadata):
        assert getattr(spliced, field.name) is getattr(base, field.name), field.name


def test_the_splice_keeps_every_control_plane_output():
    # Computing the added set by subtraction also catches a field that shadows
    # one of FlashAttention's: it would drop out of `added` here, and the
    # splice itself would pass the same keyword twice.
    spliced, _, outputs = _splice()
    added = {field.name for field in fields(NVFP4Metadata)} - {
        field.name for field in fields(FlashAttentionMetadata)
    }
    from_control = {field.name for field in fields(ControlOutputs)} - NOT_SPLICED
    assert added == from_control | set(APPENDED)
    for name in from_control:
        assert getattr(spliced, name) is getattr(outputs, name), name
    for name in APPENDED:
        assert getattr(spliced, name).name == name


def test_the_result_is_still_flash_attention_metadata():
    # Prefill rows still run FlashAttention and read these objects through the
    # base class.
    spliced, _, _ = _splice()
    assert isinstance(spliced, FlashAttentionMetadata)


# --- the decode prefix -----------------------------------------------------


def _batch(query_lens: list[int]):
    """The little of CommonAttentionMetadata that the split reads."""
    from types import SimpleNamespace

    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    return SimpleNamespace(
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        max_query_len=max(query_lens),
        num_reqs=len(query_lens),
        num_actual_tokens=starts[-1],
        is_prefilling=None,
    )


def test_the_decode_prefix_is_measured_in_rows_and_tokens():
    from nvfp4_vllm.builder import decode_split

    # A pure decode batch is all prefix; a pure prefill batch has none.
    assert decode_split(_batch([1, 1, 1])) == (3, 3)
    assert decode_split(_batch([300, 40])) == (0, 0)
    # Mixed: the two numbers stop agreeing past the boundary, which is why the
    # token count is derived rather than assumed equal to the row count.
    assert decode_split(_batch([1, 1, 300, 40])) == (2, 2)


def test_a_batch_that_was_not_reordered_is_refused():
    from nvfp4_vllm.builder import decode_split

    # Everything downstream slices by the decode prefix, so a prefill row in
    # front of a decode row has to stop the step rather than silently hand the
    # kernel a row it cannot serve.
    with pytest.raises(ValueError, match="front of the batch"):
        decode_split(_batch([300, 1, 1]))


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


# --- the pass-through under a BF16 cache -----------------------------------


def _builder(kv_cache_dtype: str):
    """A metadata builder for one attention layer, with no engine behind it.

    Its constructor only reads configuration, so a configuration is all it
    takes. An engine would load the weights to reach a branch that does not
    depend on them.
    """
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from nvfp4_vllm.builder import NVFP4MetadataBuilder

    config = EngineArgs(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    ).create_engine_config()
    model_config, parallel_config = config.model_config, config.parallel_config
    spec = FullAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=torch.bfloat16,
    )
    return NVFP4MetadataBuilder(
        spec, ["layer.0"], config, torch.device("cuda", 0)
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to reach the model configuration",
)
def test_a_bf16_cache_gets_no_control_plane():
    # The backend is chosen per engine, not per cache dtype, so the NVFP4
    # builder is also what a BF16 run gets. It has to stay a pass-through.
    _require_sm100()
    assert _builder("auto").plane is None


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to reach the model configuration",
)
def test_an_nvfp4_cache_gets_one():
    # The other half of the branch, so a builder that stopped building control
    # planes altogether cannot pass the test above.
    _require_sm100()
    builder = _builder("nvfp4")
    assert builder.plane is not None
    assert builder.reorder_batch_threshold == 1
