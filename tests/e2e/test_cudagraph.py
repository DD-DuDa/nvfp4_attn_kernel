"""The NVFP4 path under CUDA graph capture and replay.

A replayed step runs no Python at all, so almost everything the rest of
``tests/e2e`` asserts by hooking this package stops being observable the moment
graphs are on. What is still observable is what this file uses: the metadata
builder runs on the host every step, before the graph is replayed, so the batch
each step was dispatched with can be recorded; the control plane's sticky error
word survives the run; and the tokens the engine produced are the same evidence
they always were.

Two properties matter and neither is about speed.

The first is that capture happens and serving still works afterwards. Capture
runs a batch whose every row keys on the null block, which the control plane
judges not live, so the graph is recorded over a step in which nothing is a
request. If that recording were wrong, the failure would arrive later, in a
replay that serves real requests against it.

The second is the padding contract. vLLM replays at the width it captured, so a
batch narrower than the nearest captured size arrives with rows that carry no
token, a zero length and the null block. Those rows must contribute exactly
nothing. The comparison here holds the captured width fixed at four and varies
only whether the fourth row is a request or padding, so the model runs the same
shapes both times and the live rows have to answer identically.

Both are checked with graphs on and with graphs off. The eager arm is a control:
it pins that padding is something graphs introduce and not something the
scheduler does anyway, and it holds the same correctness gates on a path where
every step is Python.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os
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
MAX_NUM_SEQS = 8
# Four rows because four is one of the captured widths at this ``max_num_seqs``.
# A batch of four decodes replays with no padding at all and a batch of three
# replays into the same graph with one padding row, which is the only
# difference the comparison below is allowed to have.
BATCH = 4
# One whole FP4 page plus a partial tail, and far enough from the next boundary
# that decoding crosses it. That puts a page promotion inside the captured
# region rather than only in the parts of a step that stay in Python.
PROMPT_TOKENS = 250
GENERATED_TOKENS = 32
# Enough replayed steps in a row that a graph which only works once would show.
MIN_DECODE_STEPS = 8


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass(frozen=True)
class Step:
    """One ``build()``, as the batch looked to the metadata builder."""

    num_reqs: int
    query_lens: tuple[int, ...]

    @property
    def is_decode(self) -> bool:
        return bool(self.query_lens) and all(
            length <= 1 for length in self.query_lens
        )

    @property
    def padding_rows(self) -> int:
        return sum(1 for length in self.query_lens if length == 0)


@dataclass
class Run:
    """One ``generate()`` call: what it was asked, and what it did."""

    steps: list[Step] = field(default_factory=list)
    replays: int = 0
    completions: tuple[tuple[int, ...], ...] = ()

    @property
    def decode_steps(self) -> int:
        return sum(1 for step in self.steps if step.is_decode)

    @property
    def padded_steps(self) -> int:
        return sum(1 for step in self.steps if step.padding_rows)


@dataclass
class Session:
    graphs: bool
    cudagraph_mode: str
    full_cudagraphs: bool
    captured: int
    capture_builds: int
    resets: int
    reset_after_builds: tuple[int, ...]
    decodes_under_capture: int
    untrusted_under_capture: int
    compiles_under_capture: int
    error_code_at_reset: int
    error_code_after_startup: int
    error_code_after_serving: int
    unpadded: Run
    padded: Run


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _model_runner(llm):
    """Reach the worker's model runner through the in-process engine core."""
    core = llm.llm_engine.engine_core
    while hasattr(core, "engine_core"):
        core = core.engine_core
    return core.model_executor.driver_worker.worker.model_runner


def _metadata_builder(runner):
    builders = [
        group.get_metadata_builder(0)
        for groups in runner.attn_groups
        for group in groups
    ]
    assert len(builders) == 1, f"expected one attention group, found {builders}"
    return builders[0]


def _prompts() -> list[list[int]]:
    # Token ids rather than text, so every row is exactly the same length and
    # the page arithmetic above holds. What is being compared is which tokens
    # come out, not whether they mean anything.
    return [
        [1000 + (row * 31 + i * 7) % 20000 for i in range(PROMPT_TOKENS)]
        for row in range(BATCH)
    ]


def _engine(graphs: bool):
    from vllm import LLM

    return LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
        enforce_eager=not graphs,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "CUSTOM"},
    )


def _generate(llm, prompts, budgets: list[int]) -> tuple[tuple[int, ...], ...]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        [
            SamplingParams(temperature=0.0, max_tokens=budget, ignore_eos=True)
            for budget in budgets
        ],
        use_tqdm=False,
    )
    return tuple(tuple(output.outputs[0].token_ids) for output in outputs)


def _forget_the_graph_pool() -> None:
    """Drop vLLM's memo of the CUDA graph memory pool this engine captured into.

    vLLM asks PyTorch for one graph pool per process and keeps the handle on
    the platform class forever. Destroying an engine that captured graphs
    releases the pool, and the next engine to capture then hands PyTorch a
    handle to a pool that no longer exists, which fails inside the caching
    allocator rather than anywhere either engine can see. Nothing in a served
    process builds a second engine, so vLLM has no reason to reset this; a test
    session builds a dozen, and this file is the first of them to tear down an
    engine that captured anything.
    """
    from vllm.platforms import current_platform

    for klass in type(current_platform).__mro__:
        if "_global_graph_pool" in vars(klass):
            klass._global_graph_pool = None


def _serve(graphs: bool, probe: dict, current: dict) -> Session:
    """Build one engine, serve both batches, and return ints and token ids.

    Nothing the engine owns may outlive this call. The next engine sizes its
    KV cache from free memory, and a surviving reference to the runner or to
    any of its tensors keeps the weights and the cache allocated, so the whole
    engine is confined to this frame and named locals are dropped before the
    allocator is asked to give the memory back.
    """
    from vllm.compilation.counter import compilation_counter

    captured_before = compilation_counter.num_cudagraph_captured
    llm = _engine(graphs)
    try:
        runner = _model_runner(llm)
        plane = _metadata_builder(runner).plane
        mode = runner.compilation_config.cudagraph_mode
        # Read before serving, so it reports capture rather than capture plus
        # everything after it. Costs a host synchronization, which is what
        # keeps this out of the engine itself.
        error_after_startup = int(plane.error_code.item())

        prompts = _prompts()
        # Every row a request, so the batch is exactly a captured width.
        current["run"] = unpadded = Run()
        unpadded.completions = _generate(
            llm, prompts, [GENERATED_TOKENS] * BATCH
        )
        # The same four prompts and the same captured width, with the last row
        # retiring after its first token. Every decode step below is then
        # three requests and one padding row.
        current["run"] = padded = Run()
        padded.completions = _generate(
            llm, prompts, [GENERATED_TOKENS] * (BATCH - 1) + [1]
        )

        print(
            f"\n{'graphs' if graphs else 'eager'}: cudagraph_mode="
            f"{mode.name}, "
            f"{compilation_counter.num_cudagraph_captured - captured_before} "
            f"graphs captured over {probe['capture_builds']} capture builds "
            f"and {probe['resets']} slot table resets; "
            f"{unpadded.replays} replays over {unpadded.decode_steps} decode "
            f"steps with no padding, {padded.replays} over "
            f"{padded.decode_steps} of which {padded.padded_steps} padded"
        )
        return Session(
            graphs=graphs,
            cudagraph_mode=mode.name,
            full_cudagraphs=mode.has_full_cudagraphs(),
            captured=(
                compilation_counter.num_cudagraph_captured - captured_before
            ),
            capture_builds=probe["capture_builds"],
            resets=probe["resets"],
            reset_after_builds=tuple(probe["reset_after_builds"]),
            decodes_under_capture=probe["decodes_under_capture"],
            untrusted_under_capture=probe["untrusted_under_capture"],
            compiles_under_capture=probe["compiles_under_capture"],
            error_code_at_reset=probe["error_code_at_reset"],
            error_code_after_startup=error_after_startup,
            error_code_after_serving=int(plane.error_code.item()),
            unpadded=unpadded,
            padded=padded,
        )
    finally:
        # Startup allocations sit behind a gc.freeze(); only shutdown()
        # unfreezes them.
        llm.llm_engine.engine_core.shutdown()
        runner = plane = llm = None
        gc.collect()
        _forget_the_graph_pool()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[True, False], ids=["graphs", "eager"])
def session(request) -> Session:
    """One engine per arm, serving the same two batches.

    Everything the tests below read is collected here, because the engine is
    nearly all the cost and because the two batches have to be served by the
    same engine to be comparable.
    """
    _require_sm100()
    # In-process engine core, so the worker and its builder are reachable and
    # teardown is deterministic.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import nvfp4_vllm.builder as builder_module
    import nvfp4_vllm.impl as impl_module
    from nvfp4_decode_kernel._decode import (
        _decode_compile_cache,
        _split_decode_compile_cache,
    )
    from nvfp4_vllm.control import ControlPlane

    probe = {
        "capture_builds": 0,
        "resets": 0,
        "reset_after_builds": [],
        "decodes_under_capture": 0,
        "untrusted_under_capture": 0,
        "compiles_under_capture": 0,
        "error_code_at_reset": 0,
    }
    current: dict = {"run": Run()}

    original_build = builder_module.NVFP4MetadataBuilder.build
    original_capture_build = (
        builder_module.NVFP4MetadataBuilder.build_for_cudagraph_capture
    )
    original_reset = ControlPlane.reset
    original_fp4_decode = impl_module.fp4_decode
    original_replay = torch.cuda.CUDAGraph.replay

    def recorded_build(
        self, common_prefix_len, common_attn_metadata, fast_build=False
    ):
        meta = common_attn_metadata
        starts = meta.query_start_loc_cpu
        lengths = starts[1 : meta.num_reqs + 1] - starts[: meta.num_reqs]
        current["run"].steps.append(
            Step(num_reqs=meta.num_reqs, query_lens=tuple(lengths.tolist()))
        )
        return original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )

    def counted_capture_build(self, common_attn_metadata):
        probe["capture_builds"] += 1
        return original_capture_build(self, common_attn_metadata)

    def watched_reset(self):
        # The reset after capture clears the error word along with the slots,
        # so this is the only place capture's own verdict can still be read.
        # It costs a synchronization, at startup, once.
        probe["resets"] += 1
        # How many batches capture had built by the time this fired. A reset
        # worth having lands after the last of them; one that lands earlier is
        # undone by the capture build that follows it.
        probe["reset_after_builds"].append(probe["capture_builds"])
        probe["error_code_at_reset"] |= int(self.error_code.item())
        return original_reset(self)

    def watched_fp4_decode(**kwargs):
        # Both of the things a capturing stream cannot survive are checked
        # here rather than reasoned about: a device-value scan, which is what
        # trusted_metadata=False would ask for, and a JIT compilation, which
        # allocates and synchronizes. The compile cache is keyed on the
        # attention shape, so a call that grows it compiled something.
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            probe["decodes_under_capture"] += 1
            if kwargs.get("trusted_metadata") is not True:
                probe["untrusted_under_capture"] += 1
        compiled = len(_decode_compile_cache) + len(_split_decode_compile_cache)
        result = original_fp4_decode(**kwargs)
        if capturing and (
            len(_decode_compile_cache) + len(_split_decode_compile_cache)
        ) != compiled:
            probe["compiles_under_capture"] += 1
        return result

    def counted_replay(self):
        current["run"].replays += 1
        return original_replay(self)

    builder_module.NVFP4MetadataBuilder.build = recorded_build
    builder_module.NVFP4MetadataBuilder.build_for_cudagraph_capture = (
        counted_capture_build
    )
    ControlPlane.reset = watched_reset
    impl_module.fp4_decode = watched_fp4_decode
    torch.cuda.CUDAGraph.replay = counted_replay
    try:
        return _serve(request.param, probe, current)
    finally:
        builder_module.NVFP4MetadataBuilder.build = original_build
        builder_module.NVFP4MetadataBuilder.build_for_cudagraph_capture = (
            original_capture_build
        )
        ControlPlane.reset = original_reset
        impl_module.fp4_decode = original_fp4_decode
        torch.cuda.CUDAGraph.replay = original_replay


def test_the_engine_settles_on_the_graph_mode_it_should(session: Session):
    """Nothing has to be configured for a decode step to be captured whole.

    ``UNIFORM_SINGLE_TOKEN_DECODE`` is what this backend declares, and vLLM's
    resolver turns that into ``FULL_AND_PIECEWISE`` on its own — a full graph
    for the pure decode steps and a piecewise one for everything else. If this
    ever came back ``PIECEWISE``, the engine would still be correct and the
    host cost the graphs exist to remove would be back.
    """
    if not session.graphs:
        assert session.cudagraph_mode == "NONE", (
            f"an engine built with enforce_eager resolved to "
            f"{session.cudagraph_mode}"
        )
        return
    assert session.full_cudagraphs, (
        f"cudagraph_mode resolved to {session.cudagraph_mode}, which captures "
        "no full graph, so a decode step still pays the host for every launch"
    )


def test_capture_ran_and_the_engine_served_afterwards(session: Session):
    """Capture is a batch of rows that are not requests, recorded and kept."""
    if not session.graphs:
        assert session.captured == 0 and session.capture_builds == 0, (
            f"an eager engine captured {session.captured} graphs over "
            f"{session.capture_builds} capture builds"
        )
        assert session.resets == 0, (
            f"an eager engine reset the slot table {session.resets} times, "
            "which only capture is supposed to provoke"
        )
        return

    assert session.captured > 0, "no CUDA graph was captured"
    assert session.capture_builds > 0, (
        "no metadata was built for capture, so build_for_cudagraph_capture is "
        "not on the path and the reset after capture never fires"
    )
    # Once, and after the last batch capture built. Both halves matter. A
    # reset that fires while capture is still going gets undone by the next
    # capture build, which leaves serving to start on capture's leavings —
    # the very thing this is here to rule out — while still counting as a
    # reset that happened.
    assert session.reset_after_builds == (session.capture_builds,), (
        f"{session.capture_builds} batches were built for capture and the "
        f"slot table was reset after {session.reset_after_builds} of them, "
        "rather than exactly once after the last"
    )
    assert all(len(ids) == GENERATED_TOKENS for ids in session.unpadded.completions), (
        "the engine did not serve the full batch after capturing: token counts "
        f"were {[len(ids) for ids in session.unpadded.completions]}"
    )


def test_every_decode_step_replayed_a_graph(session: Session):
    """Several replays in a row, not one lucky first step.

    The prompts are far longer than the largest captured token count, so the
    prefill step cannot be a graph and every replay counted here is a decode
    step replaying the full graph.
    """
    run = session.unpadded
    assert run.decode_steps >= MIN_DECODE_STEPS, (
        f"only {run.decode_steps} decode steps ran, which is too few to say "
        "anything about replaying"
    )
    if not session.graphs:
        assert run.replays == 0, (
            f"an eager engine replayed {run.replays} graphs"
        )
        return
    assert run.replays == run.decode_steps, (
        f"{run.replays} graph replays over {run.decode_steps} decode steps; "
        "every pure decode step is meant to be one full graph replay"
    )


def test_the_control_plane_reported_nothing(session: Session):
    """The slot table's sticky word, read across capture and across replay.

    Reading it costs a host synchronization, which is why nothing on the
    serving path does — see ``raise_for_errors``. A test can afford it, and it
    is the only channel through which a replayed step can say that a slot went
    missing or that two rows claimed one tail.
    """
    from nvfp4_vllm.control import ERROR_NAMES

    def named(code: int) -> str:
        return "; ".join(text for bit, text in ERROR_NAMES.items() if code & bit)

    assert session.error_code_at_reset == 0, (
        "the control plane reported "
        f"{session.error_code_at_reset} over the batches capture recorded: "
        f"{named(session.error_code_at_reset)}"
    )
    assert session.error_code_after_startup == 0, (
        "the control plane reported "
        f"{session.error_code_after_startup} during startup and capture: "
        f"{named(session.error_code_after_startup)}"
    )
    assert session.error_code_after_serving == 0, (
        "the control plane reported "
        f"{session.error_code_after_serving} while serving: "
        f"{named(session.error_code_after_serving)}"
    )


def test_padding_rows_change_nothing(session: Session):
    """The padding contract, with the captured width held fixed.

    Both batches are four prompts and both replay the four-wide graph. In one
    of them the fourth row is a request; in the other it retired after its
    first token and vLLM refills its place with a row that has no token, a
    zero length and the null block. Rows are independent everywhere in a
    decode step, so the three rows they share have to answer identically —
    unless a padding row took a tail slot from one of them, wrote its own K/V
    into one of their tails, or leaked a value across.
    """
    if not session.graphs:
        # The control. Without graphs vLLM has no width to reach, so a batch
        # that lost a row is simply narrower, and there is no padding contract
        # to test here — nor may the two runs be compared, since the model
        # then runs at three rows against four.
        assert session.unpadded.padded_steps == 0
        assert session.padded.padded_steps == 0, (
            "an eager engine padded a batch, which nothing in this file "
            f"expects: {session.padded.steps}"
        )
        return

    assert session.unpadded.padded_steps == 0, (
        "the batch that should have needed no padding was padded on "
        f"{session.unpadded.padded_steps} of {len(session.unpadded.steps)} "
        "steps, so the comparison below has no padding to be about"
    )
    assert session.padded.padded_steps >= MIN_DECODE_STEPS, (
        f"only {session.padded.padded_steps} steps carried a padding row, out "
        f"of {len(session.padded.steps)}"
    )
    assert all(
        step.num_reqs == BATCH
        for step in session.padded.steps
        if step.is_decode
    ), (
        "the padded batch did not replay at the same width as the unpadded "
        f"one: {[step.num_reqs for step in session.padded.steps]}"
    )

    divergent = [
        row
        for row in range(BATCH - 1)
        if session.unpadded.completions[row] != session.padded.completions[row]
    ]
    assert not divergent, (
        f"rows {divergent} decoded differently once the fourth row of the "
        "batch became padding, so a padding row is contributing something\n"
        + "\n".join(
            f"  row {row}: without padding "
            f"{list(session.unpadded.completions[row][:12])}, with padding "
            f"{list(session.padded.completions[row][:12])}"
            for row in divergent[:3]
        )
    )


def test_nothing_was_compiled_or_checked_while_capturing(session: Session):
    """The two things a capturing stream cannot survive.

    A CuTeDSL compile allocates and synchronizes, so one that happens during
    capture either fails outright or records something that was never meant to
    be a step. And ``trusted_metadata`` is what keeps the decode kernel from
    scanning ``seqused_*`` on the device for validation, which would be a
    device-to-host read per layer — fatal under capture and merely expensive
    outside it.
    """
    if not session.graphs:
        pytest.skip("nothing is captured with graphs off")
    assert session.decodes_under_capture > 0, (
        "the decode kernel never ran on a capturing stream, so this file has "
        "not shown that anything of ours went into a graph"
    )
    assert session.compiles_under_capture == 0, (
        f"{session.compiles_under_capture} decode calls compiled a kernel "
        "while the stream was capturing; the warmup before capture is meant "
        "to have compiled every shape capture needs"
    )
    assert session.untrusted_under_capture == 0, (
        f"{session.untrusted_under_capture} decode calls under capture asked "
        "the kernel to validate its metadata, which reads device values back"
    )
