"""What a decode step is allowed to cost the host.

The read path is built so that a steady-state decode step never asks the device
a question. Every branch that would need an answer was moved into a kernel: the
slot assignment, which rows filled a page, which pages to quantize. The payoff
is that the host stays a step ahead of the device, and the cost of losing it is
invisible — nothing breaks, the step just takes longer, and it stays that way
until somebody profiles it. So it is asserted rather than trusted.

Three things are checked, and the division of labour between them matters.

``set_sync_debug_mode`` is the sharp one. It is armed only while this package's
own code is on the stack, which is what makes the answer attributable: vLLM
synchronizes every step in its sampler, and that is not ours to fix. When it
fires it names the file and line, so the report is a fix rather than a hunt.
PyTorch calls it a prototype and says it does not detect every synchronizing
operation, which is why it is not the only check here.

The launch counts are the second. They are taken by counting calls, not by
reading a trace: an assertion that depends on parsing profiler output fails for
reasons that have nothing to do with the property being asserted.

The profiler is the third, and it is the one that covers what the other two
cannot — a copy that does not synchronize. Its count is only meaningful as a
difference, since vLLM copies on its own account every step, so a BF16 engine
of the same shape is run beside ours and the two are compared.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field

import pytest
import torch


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 4
PROMPT_TOKENS = 300
GENERATED_TOKENS = 24

SYNC_MESSAGE = "synchronizing CUDA operation"


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass
class Violation:
    where: str
    filename: str
    lineno: int


@dataclass
class Watch:
    """Arms the sync detector only while this package's code is running."""

    armed: bool = False
    in_decode_step: bool = False
    decode_steps: int = 0
    violations: list[Violation] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    @contextlib.contextmanager
    def watching(self, where: str):
        if not (self.armed and self.in_decode_step):
            yield
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                yield
            finally:
                torch.cuda.set_sync_debug_mode("default")
        for entry in caught:
            # set_sync_debug_mode announces itself as a prototype on every
            # call, and the CuTeDSL stack deprecates something on most of
            # them. Neither is what this is looking for.
            if SYNC_MESSAGE in str(entry.message):
                self.violations.append(
                    Violation(where, entry.filename, entry.lineno)
                )

    def tally(self, name: str) -> None:
        if self.armed and self.in_decode_step:
            self.counts[name] += 1


def _memcpy_counts(prof) -> Counter:
    """Device-side copies in a profile, by direction."""
    counts: Counter = Counter()
    for entry in prof.key_averages():
        if entry.key.startswith("Memcpy"):
            direction = entry.key.split()[1] if " " in entry.key else entry.key
            counts[direction] += entry.count
    return counts


@dataclass
class Session:
    watch: Watch
    num_layers: int
    nvfp4_memcpy: Counter
    nvfp4_decode_steps: int
    bf16_memcpy: Counter
    bf16_decode_steps: int


def _sampling(module):
    return module.SamplingParams(
        max_tokens=GENERATED_TOKENS, ignore_eos=True, temperature=0.0
    )


def _prompts(count: int) -> list[list[int]]:
    # Token ids rather than text, so the prompt is exactly as long as intended
    # and every row in the batch is the same length. What this file measures is
    # the shape of a step, not what the model says.
    return [
        [1000 + (row * 7 + i) % 20000 for i in range(PROMPT_TOKENS)]
        for row in range(count)
    ]


def _engine(kv_cache_dtype: str, backend: str):
    from vllm import LLM

    return LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": backend},
    )


def _profile_decode(llm, module, decode_counter: dict) -> tuple[Counter, int]:
    """Profile one generation and return its copies and its decode steps."""
    from torch.profiler import ProfilerActivity, profile

    decode_counter["steps"] = 0
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate(
            [module.TokensPrompt(prompt_token_ids=ids) for ids in _prompts(2)],
            _sampling(module),
        )
    return _memcpy_counts(prof), decode_counter["steps"]


@pytest.fixture(scope="module")
def session() -> Session:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import gc

    import vllm
    from vllm.inputs import TokensPrompt

    import nvfp4_vllm.builder as builder_module
    import nvfp4_vllm.impl as impl_module
    import nvfp4_vllm.layout as layout_module
    import nvfp4_vllm.promote as promote_module
    from nvfp4_vllm.control import ControlPlane

    vllm.TokensPrompt = TokensPrompt

    watch = Watch()
    # vLLM's own steps, counted the same way for both engines so the copy
    # counts below are per step rather than per run.
    decode_counter = {"steps": 0}

    original_build = builder_module.NVFP4MetadataBuilder.build
    original_forward = impl_module.NVFP4Impl.forward
    original_update = impl_module.NVFP4Impl.do_kv_cache_update
    original_prepare = ControlPlane.prepare
    original_carve = layout_module.carve
    original_write = impl_module.write_kv
    original_launch = promote_module.launch

    def watched_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        meta = common_attn_metadata
        # Every row contributing one token is what a decode step looks like
        # from here, and it is the only shape this file has anything to say
        # about. Counted for both engines, since they are compared per step;
        # only armed for the one that has an NVFP4 path to police.
        is_decode = meta.num_actual_tokens == meta.num_reqs
        if is_decode:
            decode_counter["steps"] += 1
        watch.in_decode_step = self.nvfp4 and is_decode
        if watch.in_decode_step:
            watch.decode_steps += 1
        with watch.watching("build"):
            return original_build(
                self, common_prefix_len, common_attn_metadata, fast_build
            )

    def watched_forward(self, *args, **kwargs):
        with watch.watching("forward"):
            return original_forward(self, *args, **kwargs)

    def watched_update(self, *args, **kwargs):
        with watch.watching("do_kv_cache_update"):
            return original_update(self, *args, **kwargs)

    def counted_prepare(self, *args, **kwargs):
        watch.tally("control")
        return original_prepare(self, *args, **kwargs)

    def counted_carve(*args, **kwargs):
        watch.tally("carve")
        return original_carve(*args, **kwargs)

    def counted_write(*args, **kwargs):
        watch.tally("write_kv")
        return original_write(*args, **kwargs)

    def counted_launch(metadata, runtime):
        watch.tally("promotion")
        watch.counts["num_layers"] = runtime.num_layers
        return original_launch(metadata, runtime)

    builder_module.NVFP4MetadataBuilder.build = watched_build
    impl_module.NVFP4Impl.forward = watched_forward
    impl_module.NVFP4Impl.do_kv_cache_update = watched_update
    ControlPlane.prepare = counted_prepare
    layout_module.carve = counted_carve
    impl_module.write_kv = counted_write
    promote_module.launch = counted_launch

    try:
        llm = _engine("nvfp4", "CUSTOM")
        try:
            # Unarmed. Startup compiles kernels and carves the cache views, so
            # the first pass through any of this is not a steady state and has
            # no business being measured as one.
            llm.generate(
                [
                    TokensPrompt(prompt_token_ids=ids)
                    for ids in _prompts(MAX_NUM_SEQS)
                ],
                _sampling(vllm),
            )
            watch.armed = True
            watch.decode_steps = 0
            nvfp4_memcpy, nvfp4_steps = _profile_decode(llm, vllm, decode_counter)
        finally:
            watch.armed = False
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        # The BF16 arm runs through this same backend, which is a pass-through
        # under any other cache dtype. That makes the cache dtype the only
        # difference between the two profiles, so what the subtraction leaves
        # is the NVFP4 path's own copies and nothing else.
        bf16 = _engine("auto", "CUSTOM")
        try:
            bf16.generate(
                [
                    TokensPrompt(prompt_token_ids=ids)
                    for ids in _prompts(MAX_NUM_SEQS)
                ],
                _sampling(vllm),
            )
            bf16_memcpy, bf16_steps = _profile_decode(bf16, vllm, decode_counter)
        finally:
            bf16.llm_engine.engine_core.shutdown()
            del bf16
            gc.collect()
            torch.cuda.empty_cache()

        yield Session(
            watch=watch,
            num_layers=watch.counts["num_layers"],
            nvfp4_memcpy=nvfp4_memcpy,
            nvfp4_decode_steps=nvfp4_steps,
            bf16_memcpy=bf16_memcpy,
            bf16_decode_steps=bf16_steps,
        )
    finally:
        builder_module.NVFP4MetadataBuilder.build = original_build
        impl_module.NVFP4Impl.forward = original_forward
        impl_module.NVFP4Impl.do_kv_cache_update = original_update
        ControlPlane.prepare = original_prepare
        layout_module.carve = original_carve
        impl_module.write_kv = original_write
        promote_module.launch = original_launch


def test_a_decode_step_never_asks_the_device_a_question(session: Session):
    """The gate. Nothing in this package synchronizes during a decode step."""
    watch = session.watch
    assert watch.decode_steps > 1, (
        f"only {watch.decode_steps} decode steps ran, so the steady state was "
        "never reached and this proves nothing"
    )
    assert not watch.violations, "\n".join(
        [f"{len(watch.violations)} synchronizing calls in a decode step:"]
        + [
            f"  in {v.where}: {v.filename}:{v.lineno}"
            for v in watch.violations[:20]
        ]
    )


def test_the_step_costs_the_launches_it_is_supposed_to(session: Session):
    """One control kernel and one promotion a step, whatever the batch holds.

    The failure this is built for is a regression to per-layer work: a
    promotion loop over layers, or a control plane rebuilt for each of them.
    Both would still be correct, and both would cost several milliseconds a
    step against roughly one of attention.
    """
    counts = session.watch.counts
    steps = session.watch.decode_steps
    assert counts["control"] == steps, (
        f"the control kernel ran {counts['control']} times over {steps} decode "
        "steps; it is meant to run once a step, before any layer"
    )
    assert counts["promotion"] == steps, (
        f"promotion launched {counts['promotion']} times over {steps} decode "
        "steps; it is meant to launch once a step whether or not a row crossed"
    )
    assert counts["write_kv"] == steps * session.num_layers, (
        f"the write path ran {counts['write_kv']} times over {steps} steps of "
        f"{session.num_layers} layers, which is not once per layer per step"
    )


def test_the_cache_views_are_carved_once_and_not_again(session: Session):
    """``layout.carve`` is address arithmetic, but it builds Python objects.

    Once per layer per step would be thirty-two allocations of tuples and
    views on a path that is meant to touch the host as little as possible. The
    runtime keys them on the cache pointer for exactly this reason.
    """
    assert session.watch.counts["carve"] == 0, (
        f"the cache was carved {session.watch.counts['carve']} times during "
        "decode; the views are meant to be built once and kept"
    )


def test_we_copy_no_more_than_the_bf16_engine_does(session: Session):
    """The cross-check the sync detector cannot do: copies that do not block.

    Only meaningful as a difference. vLLM copies on its own account every step
    — the sampled token has to reach the host somehow — so the number to hold
    at zero is ours on top of that, which is what a BF16 engine of the same
    shape measures.
    """
    ours = session.nvfp4_memcpy
    theirs = session.bf16_memcpy
    steps = session.nvfp4_decode_steps
    assert steps and steps == session.bf16_decode_steps, (
        f"the two engines ran {steps} and {session.bf16_decode_steps} decode "
        "steps on the same workload, so their copy counts are not comparable"
    )
    print(
        f"\ncopies over {steps} decode steps (nvfp4 | bf16): "
        + ", ".join(
            f"{d} {ours[d]} | {theirs[d]}"
            for d in sorted(set(ours) | set(theirs))
        )
    )
    excess = {
        direction: ours[direction] - theirs[direction]
        for direction in set(ours) | set(theirs)
        if ours[direction] > theirs[direction]
    }
    assert not excess, (
        f"the NVFP4 path copies more than the BF16 one over the same "
        f"{steps} decode steps: {excess}. Totals are nvfp4={dict(ours)}, "
        f"bf16={dict(theirs)}."
    )


def test_the_gate_measured_the_configuration_people_serve(session: Session):
    """``NVFP4_DEBUG`` would have invalidated everything above.

    It reads the sticky error word every step, which is exactly the host
    synchronization the rest of this file asserts the absence of. Off by
    default; this makes sure a stray export did not quietly turn the gate into
    a measurement of a configuration nobody runs. What the switch itself does
    is covered in ``test_control.py``, which does not need an engine for it.
    """
    from nvfp4_vllm.builder import DEBUG_ENV

    assert os.environ.get(DEBUG_ENV) != "1", (
        f"{DEBUG_ENV} was set while the sync gate ran, so the gate was "
        "measuring a configuration nobody serves"
    )
