"""Many requests through one engine, watching for a decode that goes non-finite.

The failures this is built for do not show up in a single generation. A tail
slot outlives the request that filled it, so a fault only becomes visible when
the slot changes hands — and only when the new tenant's sequence is shorter
than the old one's, since that is what leaves the previous tenant's tokens
sitting past the new length where the mask multiplies rather than replaces
them. One long generation never reuses a slot and never sees any of it.

So the prompts here differ in length on purpose, and there are far more
requests than slots. Every decode output of every layer of every step is
checked, which is the part a text-level assertion cannot do: a single
non-finite lane in an early layer is laundered into plausible-looking text by
the layers above it.

Four of the six lengths also fill their page during the run, which puts the
other way a slot's contents move — promotion wrapping the tail back to its
start — into the same mixture as the handovers, and at different steps for
each. So a slot is handed on having been wrapped once or twice mid-generation
rather than only ever grown, which is a state neither this file nor
``test_promotion.py`` reached before. The wrap and the handover are still
several steps apart, since a request only leaves at the end of its own
generation; a slot reused on the very next step would need requests that stop
at different lengths, and nothing here does that yet.

The counting is done on the device and read once at the end. Checking a step's
output on the host would synchronize on every step, which would change the
timing the soak is meant to sample.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
- ``NVFP4_SOAK_ROUNDS``: batches of requests to run. The default is enough to
  reuse every slot several times; raise it for a campaign.
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
ROUNDS = int(os.environ.get("NVFP4_SOAK_ROUNDS", "6"))

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8

# Tails that start at very different offsets, since a slot handed from a long
# tail to a short one is the case that exposes stale tokens. The offsets are
# 0, 40, 72, 100, 118 and 124 tokens into a page, so four of the six fill their
# page within the thirty generated: 1024 during its own prefill, then 1148 on
# step 4, 1142 on step 10 and 1124 on step 28. 1064 and 1096 never cross, so
# both kinds of row are in every batch.
PROMPT_TOKENS = (1024, 1064, 1096, 1124, 1142, 1148)
GENERATED_TOKENS = 30


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass
class Soak:
    """What a run of many requests left behind."""

    bad_steps_per_layer: list[int]
    decode_steps: int
    requests: int
    slots_used: int
    pages_sealed: int
    prompt_lengths: list[int]
    texts: list[str]
    token_counts: list[int]


def _read_path_module():
    """The read-path test's prompt builder, loaded by path.

    ``tests/`` has no package structure. Loading by path keeps one definition
    of "natural text of exactly n tokens" rather than a copy that could drift.
    """
    path = Path(__file__).resolve().parent / "test_read_path.py"
    spec = importlib.util.spec_from_file_location("_read_path", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def soak() -> Soak:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )

    import nvfp4_vllm.impl as impl_module
    import nvfp4_vllm.promote as promote_module

    device = torch.device("cuda")
    # [layer] count of steps whose output held a non-finite value, so a failure
    # names the layer it started in instead of only saying the run went bad.
    tally = torch.zeros(256, dtype=torch.int32, device=device)
    # [slot] whether the run ever put a row there, so the reuse this file
    # depends on is measured rather than assumed.
    slots_seen = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=device)
    # Pages promotion actually sealed, so the crossings the lengths were chosen
    # for are counted rather than assumed. Accumulated on the device for the
    # same reason as the two above.
    sealed = torch.zeros((), dtype=torch.int32, device=device)
    counters = {"decode_steps": 0}

    original_decode = impl_module.NVFP4Impl._decode
    original_update = impl_module.NVFP4Impl.do_kv_cache_update
    original_launch = promote_module.launch

    def watch_decode(self, rows, query, kv_cache, attn_metadata, output):
        original_decode(self, rows, query, kv_cache, attn_metadata, output)
        tally[self.layer_index] += (
            (~output[:rows].isfinite()).any().to(torch.int32)
        )
        counters["decode_steps"] += 1

    def watch_update(self, layer, key, value, kv_cache, slot_mapping):
        original_update(self, layer, key, value, kv_cache, slot_mapping)
        if self.layer_index != 0 or self.runtime is None:
            return
        metadata, _, _, _ = get_attention_context(layer.layer_name)
        if metadata is None:
            return
        # A padding row carries -1 and contributes a zero, which amax cannot
        # take back a slot it already marked. Reading the slots here instead
        # would synchronize on every step and change the timing being sampled.
        rows = metadata.row_to_slot
        slots_seen.scatter_reduce_(
            0,
            rows.clamp(min=0).long(),
            (rows >= 0).to(torch.int32),
            reduce="amax",
        )

    def watch_launch(metadata, runtime):
        original_launch(metadata, runtime)
        sealed.add_((metadata.promotion_pages >= 0).sum().to(torch.int32))

    impl_module.NVFP4Impl._decode = watch_decode
    impl_module.NVFP4Impl.do_kv_cache_update = watch_update
    promote_module.launch = watch_launch
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
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
        build = _read_path_module()._prompt_of_exact_length
        prompts = {length: build(llm, length) for length in PROMPT_TOKENS}
        sampling = SamplingParams(
            max_tokens=GENERATED_TOKENS, ignore_eos=True, temperature=0.0
        )
        # Startup runs the engine on shapes of its own choosing; only what the
        # rounds below seal is being counted.
        sealed.zero_()

        texts: list[str] = []
        token_counts: list[int] = []
        lengths: list[int] = []
        requests = 0
        for round_index in range(ROUNDS):
            # The batch's size changes from round to round as well as its
            # lengths, so rows finish at different times and slots are handed
            # on in a different order each time rather than settling into a
            # fixed row-to-slot mapping.
            width = 2 + round_index % (MAX_NUM_SEQS - 1)
            batch = [
                PROMPT_TOKENS[(round_index + i) % len(PROMPT_TOKENS)]
                for i in range(width)
            ]
            completions = llm.generate(
                [
                    TokensPrompt(prompt_token_ids=prompts[length])
                    for length in batch
                ],
                sampling,
            )
            requests += len(batch)
            lengths.extend(batch)
            for completion in completions:
                output = completion.outputs[0]
                texts.append(output.text)
                token_counts.append(len(output.token_ids))

        yield Soak(
            bad_steps_per_layer=tally.tolist(),
            decode_steps=counters["decode_steps"],
            requests=requests,
            slots_used=int(slots_seen.sum()),
            pages_sealed=int(sealed),
            prompt_lengths=lengths,
            texts=texts,
            token_counts=token_counts,
        )
    finally:
        impl_module.NVFP4Impl._decode = original_decode
        impl_module.NVFP4Impl.do_kv_cache_update = original_update
        promote_module.launch = original_launch
        llm.llm_engine.engine_core.shutdown()
        del llm


def test_the_run_reused_its_slots(soak: Soak):
    """Without reuse the rest of this file proves nothing."""
    assert soak.requests > soak.slots_used, (
        f"{soak.requests} requests over {soak.slots_used} slots is not enough "
        "reuse for a slot to have changed hands"
    )
    assert soak.decode_steps > 0, "no decode step ran"


def test_the_run_sealed_the_pages_its_lengths_call_for(soak: Soak):
    """The other premise: slots were wrapped by promotion as well as reused.

    Counted rather than asserted to be positive, because the interesting
    mixture is quantitative — a change to the lengths that leaves one crossing
    a run would still pass a "greater than zero" and would no longer be putting
    handover and wrapping together.
    """
    expected = sum(
        1
        for length in soak.prompt_lengths
        for seq in range(length, length + GENERATED_TOKENS)
        if seq % PAGE_SIZE == 0
    )
    assert soak.pages_sealed == expected, (
        f"{soak.pages_sealed} pages were sealed over {soak.requests} requests, "
        f"and the lengths call for {expected}"
    )
    assert expected >= soak.requests // 2, (
        f"only {expected} of {soak.requests} requests cross a page boundary, "
        "so the run is mostly the no-crossing case again"
    )


def test_no_decode_step_went_non_finite(soak: Soak):
    bad = [
        (layer, count)
        for layer, count in enumerate(soak.bad_steps_per_layer)
        if count
    ]
    assert not bad, (
        f"non-finite decode output on {sum(c for _, c in bad)} of "
        f"{soak.decode_steps} layer-steps, first at layer {bad[0][0]}: "
        f"(layer, steps) {bad[:10]}"
    )


def test_every_request_produced_sane_text(soak: Soak):
    """A NaN that reaches the sampler is not the only way a run can rot.

    A stale tail that is finite still answers with a previous request's
    context, and that shows up as text rather than as an arithmetic fault.
    """
    assert len(soak.texts) == soak.requests
    short = [count for count in soak.token_counts if count != GENERATED_TOKENS]
    assert not short, (
        f"{len(short)} of {soak.requests} requests stopped early: {short[:10]}"
    )
    empty = [index for index, text in enumerate(soak.texts) if not text.strip()]
    assert not empty, f"requests {empty[:10]} produced no text"
