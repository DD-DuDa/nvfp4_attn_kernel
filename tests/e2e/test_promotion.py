"""What a filled BF16 tail page becomes once it is sealed into the cache.

The main check is byte equality, for the same reason the write path's is: both
sides run the same quantizer, so the only thing that can differ is placement —
which layer, which block, which region. It runs over all thirty-two layers
rather than a sample, because what promotion newly relies on is a table of
per-layer base addresses, and the way a table goes wrong is by being shifted
or scrambled, which leaves the ends right and the middle wrong.

Bytes are not the whole story, though. A page can land correctly and still be
invisible to the next step, if the FP4 prefix did not grow or the tail did not
wrap back to its start. Neither shows up in a byte comparison, so the decode
that follows a crossing is also compared against the PyTorch oracle, on a few
layers — the lengths that would be wrong are per-step, not per-layer, so the
layer coverage that matters is the byte check's.

The prompts are chosen so that two rows cross on the same step, from different
tails into different blocks. One row crossing at a time would let a launch
that mixed up its rows pass.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_pages,
    quantize_value_pages,
)


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8

# 1024 seals a page during its own prefill, which is the path a prompt that is
# an exact multiple of the page size takes: the work table quantizes only
# (L-1)//P pages and promotion finishes the last one. 1022 and 894 are both
# 126 short of a boundary, so they cross together on decode step 2, out of
# different tail slots into different blocks. 1000 crosses alone later. Every
# row but the last crosses twice within the generation.
PROMPT_LENGTHS = (1024, 1022, 894, 1000)
GENERATED_TOKENS = 140

# The oracle is checked on these layers only. What a crossing can get wrong
# beyond the bytes is the step's own lengths, which every layer shares, so
# spending thirty-two forward passes of oracle on it buys nothing the byte
# comparison does not already cover. First, last, and two in between.
CHECKED_LAYERS = (0, 1, 15, 31)

# Same floor as tests/kernel and test_read_path.py: both sides quantize
# independently from the same BF16.
MIN_COSINE = 0.99

# Two launches a step. Measured at about 160 microseconds; a per-layer loop
# would be about 3 milliseconds, which is what this is really watching for.
MAX_HOST_MICROS = 500.0


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


def _expected_crossings() -> int:
    """How many pages the prompt lengths above have to seal, counted out.

    A request is at its prompt length on the step that prefills it, and one
    token longer on each step after, so over a generation of N tokens it takes
    every length from its prompt through its prompt plus N-1. It crosses
    whenever one of those is a multiple of the page size. Spelling it out means
    a change to the lengths cannot quietly leave the run with fewer crossings
    than the tests below believe they are inspecting.
    """
    return sum(
        1
        for length in PROMPT_LENGTHS
        for seq in range(length, length + GENERATED_TOKENS)
        if seq % PAGE_SIZE == 0
    )


def _prompt_of(state: "StepState", step: int, row: int) -> int:
    """Which request is sitting in a row, named by the prompt it started from.

    Row numbers are not identity: vLLM renumbers the batch between the prefill
    step and the decodes that follow, so row 0 is one request on one step and
    another on the next. What does not move is how far along a request is —
    every request here is scheduled on every step and grows by exactly one
    token, so subtracting the step number from the row's length gives back the
    prompt length it started at, and the four prompt lengths are distinct.
    ``test_every_request_was_scheduled_on_every_step`` checks the premise.
    """
    return state.seq_lens[row] - step


@dataclass
class Sealed:
    """One page promotion wrote, and whether it holds the right bytes."""

    step: int
    row: int
    prompt: int
    layer: int
    block: int
    regions_equal: tuple[bool, bool, bool, bool]


@dataclass
class StepState:
    """The lengths one step reported, per row."""

    seq_lens: list[int]
    seqused_fp4: list[int]
    seqused_residual: list[int]
    promotion_pages: list[int]
    promotion_sources: list[int]


@dataclass
class Session:
    sealed: list[Sealed]
    states: list[StepState]
    num_layers: int
    launches: int
    host_micros: list[float]
    device_micros: list[float]
    decodes: list["Decode"]
    history_key: dict[int, dict[int, torch.Tensor]]
    history_value: dict[int, dict[int, torch.Tensor]]


@dataclass
class Decode:
    """One layer's decode on one step, with what it needs to be redone."""

    step: int
    layer: int
    row: int
    seq_len: int
    softmax_scale: float
    query: torch.Tensor
    output: torch.Tensor

    @property
    def prompt(self) -> int:
        """Which request this was, whatever row the step put it in."""
        return self.seq_len - self.step

    @property
    def seqused_fp4(self) -> int:
        return ((self.seq_len - 1) // PAGE_SIZE) * PAGE_SIZE

    @property
    def residual(self) -> int:
        return self.seq_len - self.seqused_fp4


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _oracle():
    """The kernel suite's decode oracle, loaded by path.

    ``tests/`` has no package structure, so the two directories cannot import
    each other. Loading by path keeps one definition of the oracle rather than
    a copy that could drift from the kernel it pins down.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "kernel"
        / "test_fp4_decode_correctness.py"
    )
    spec = importlib.util.spec_from_file_location("_fp4_decode_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_sealed_pages(
    step: int, lengths: list[int], metadata, runtime
) -> list[Sealed]:
    """Compare every page this launch sealed against the packed quantizer.

    Called straight after the launch, while the tail still holds what was
    quantized out of it — the next step's write is the first thing that
    overwrites it. Both K and V for every layer are quantized in one call each,
    so the comparison costs two launches rather than sixty-four.
    """
    rows = (metadata.promotion_pages >= 0).nonzero().flatten().tolist()
    if not rows:
        return []
    blocks = metadata.promotion_pages.tolist()
    slots = [
        source // PAGE_SIZE for source in metadata.promotion_source_tokens.tolist()
    ]
    layers = range(runtime.num_layers)
    pairs = [(layer, row) for layer in layers for row in rows]

    tails_key = torch.stack(
        [runtime.tail_key[layer, slots[row]] for layer, row in pairs]
    ).contiguous()
    tails_value = torch.stack(
        [runtime.tail_value[layer, slots[row]] for layer, row in pairs]
    ).contiguous()
    expected = (
        *quantize_key_pages(tails_key),
        *quantize_value_pages(tails_value),
    )

    sealed = []
    for index, (layer, row) in enumerate(pairs):
        regions = runtime.layer_regions(layer)
        sealed.append(
            Sealed(
                step=step,
                row=row,
                prompt=lengths[row] - step,
                layer=layer,
                block=blocks[row],
                regions_equal=tuple(
                    torch.equal(region[blocks[row]], want[index])
                    for region, want in zip(regions, expected)
                ),
            )
        )
    return sealed


@pytest.fixture(scope="module")
def session() -> Session:
    _require_sm100()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )

    import nvfp4_vllm.impl as impl_module
    import nvfp4_vllm.promote as promote_module

    sealed: list[Sealed] = []
    states: list[StepState] = []
    decodes: list[Decode] = []
    writes: dict[int, list[torch.Tensor]] = {}
    starts: list[torch.Tensor] = []
    host_micros: list[float] = []
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    counters = {"launches": 0, "num_layers": 0}

    original_update = impl_module.NVFP4Impl.do_kv_cache_update
    original_decode = impl_module.NVFP4Impl._decode
    original_launch = promote_module.launch

    def record_update(self, layer, key, value, kv_cache, slot_mapping):
        original_update(self, layer, key, value, kv_cache, slot_mapping)
        metadata, _, _, _ = get_attention_context(layer.layer_name)
        if self.runtime is None or metadata is None:
            return
        tokens = metadata.num_actual_tokens
        if self.layer_index in CHECKED_LAYERS:
            # Off the device immediately: keeping the whole prefill for four
            # layers would be most of a gibibyte against a cache that took
            # ninety percent of the card.
            writes.setdefault(self.layer_index, []).append(
                (key[:tokens].cpu(), value[:tokens].cpu())
            )
        if self.layer_index == 0:
            starts.append(metadata.query_start_loc.clone())

    def record_decode(self, rows, query, kv_cache, attn_metadata, output):
        original_decode(self, rows, query, kv_cache, attn_metadata, output)
        if self.layer_index not in CHECKED_LAYERS:
            return
        step = len(starts) - 1
        for row in range(rows):
            decodes.append(
                Decode(
                    step=step,
                    layer=self.layer_index,
                    row=row,
                    seq_len=0,  # filled in once the lengths come back
                    softmax_scale=self.scale,
                    query=query[row].clone(),
                    output=output[row].clone(),
                )
            )

    def record_launch(metadata, runtime):
        # Timed before anything that reads a device tensor: the checks below
        # synchronize, and the point of the measurement is what a step that
        # does no checking pays.
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        started = time.perf_counter()
        original_launch(metadata, runtime)
        host_micros.append((time.perf_counter() - started) * 1e6)
        end.record()
        events.append((begin, end))
        counters["launches"] += 1
        counters["num_layers"] = runtime.num_layers

        step = len(starts) - 1
        lengths = metadata.seq_lens.tolist()
        states.append(
            StepState(
                seq_lens=lengths,
                seqused_fp4=metadata.seqused_fp4.tolist(),
                seqused_residual=metadata.seqused_residual.tolist(),
                promotion_pages=metadata.promotion_pages.tolist(),
                promotion_sources=metadata.promotion_source_tokens.tolist(),
            )
        )
        sealed.extend(_check_sealed_pages(step, lengths, metadata, runtime))

    impl_module.NVFP4Impl.do_kv_cache_update = record_update
    impl_module.NVFP4Impl._decode = record_decode
    promote_module.launch = record_launch
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
        for recorder in (sealed, states, decodes, starts, host_micros, events):
            recorder.clear()
        writes.clear()
        counters["launches"] = 0
        llm.generate(
            [
                TokensPrompt(prompt_token_ids=list(range(10, 10 + length)))
                for length in PROMPT_LENGTHS
            ],
            SamplingParams(
                max_tokens=GENERATED_TOKENS, ignore_eos=True, temperature=0.0
            ),
        )
        torch.cuda.synchronize()

        boundaries = [start.tolist() for start in starts]
        for decode in decodes:
            decode.seq_len = states[decode.step].seq_lens[decode.row]
        yield Session(
            sealed=sealed,
            states=states,
            num_layers=counters["num_layers"],
            launches=counters["launches"],
            host_micros=host_micros,
            device_micros=[
                begin.elapsed_time(end) * 1e3 for begin, end in events
            ],
            decodes=decodes,
            history_key=_per_request(writes, boundaries, states, 0),
            history_value=_per_request(writes, boundaries, states, 1),
        )
    finally:
        impl_module.NVFP4Impl.do_kv_cache_update = original_update
        impl_module.NVFP4Impl._decode = original_decode
        promote_module.launch = original_launch
        llm.llm_engine.engine_core.shutdown()
        del llm


def _per_request(
    writes: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    boundaries: list[list[int]],
    states: list[StepState],
    side: int,
) -> dict[int, dict[int, torch.Tensor]]:
    """Regroup the K or V a layer wrote, from per step to per request.

    Each step arrives as one flat run of tokens; ``query_start_loc`` says which
    stretch of it belongs to which row. The stretches are filed under the
    request's prompt length rather than its row number, because the row number
    changes under it — see ``_prompt_of``.
    """
    grouped: dict[int, dict[int, list[torch.Tensor]]] = {}
    for layer, steps in writes.items():
        parts = grouped.setdefault(layer, {})
        for step, written in enumerate(steps):
            for row in range(len(states[step].seq_lens)):
                start = boundaries[step][row]
                stop = boundaries[step][row + 1]
                parts.setdefault(_prompt_of(states[step], step, row), []).append(
                    written[side][start:stop]
                )
    return {
        layer: {
            prompt: torch.cat(pieces) for prompt, pieces in parts.items()
        }
        for layer, parts in grouped.items()
    }


# --- the pages themselves --------------------------------------------------


def test_two_rows_seal_a_page_on_the_same_step(session: Session):
    """The interesting case has to happen, not just be arranged for.

    With one row crossing at a time, a launch that read the wrong row's slot
    or wrote the wrong row's block would still produce the right bytes, since
    there is only one of each to choose from.
    """
    per_step: dict[int, set[int]] = {}
    for page in session.sealed:
        per_step.setdefault(page.step, set()).add(page.row)
    widest = max((len(rows) for rows in per_step.values()), default=0)
    assert widest > 1, (
        f"no step sealed more than one row's page: {per_step}"
    )


def test_every_sealed_page_holds_the_quantizer_s_own_bytes(session: Session):
    """The launch honoured its index tensors, in every layer's own allocation.

    Both sides read the same two index tensors — the slot from
    ``promotion_source_tokens``, the block from ``promotion_pages`` — so a
    control plane that named the wrong slot would move both together and go
    unseen here. That half belongs to ``test_control.py``, which checks those
    tensors against a reference, and to the oracle below, which would not
    reconcile against a page built from the wrong tokens.

    What is left is the half this is for: given the right indices, did the
    bytes land in the right layer, block and region. Every layer is compared
    rather than a sample, because the address table promotion reaches the
    layers through is exactly the kind of thing that goes wrong in the middle
    while the ends stay right.
    """
    assert session.sealed, "no page was ever sealed, so this checked nothing"
    names = ("packed K", "K scales", "packed V", "V scales")
    wrong = [
        f"step {page.step} row {page.row} layer {page.layer} "
        f"block {page.block}: {names[index]}"
        for page in session.sealed
        for index, equal in enumerate(page.regions_equal)
        if not equal
    ]
    assert not wrong, (
        f"{len(wrong)} of {4 * len(session.sealed)} promoted regions differ "
        f"from the packed quantizer: {wrong[:12]}"
    )

    layers = {page.layer for page in session.sealed}
    assert layers == set(range(session.num_layers)), (
        f"layers {sorted(layers)} were compared, out of "
        f"{session.num_layers} the model has"
    )


def test_a_prompt_that_ends_on_a_boundary_seals_its_last_page(
    session: Session,
):
    """A prompt of exactly N pages leaves the last one to promotion.

    The write path quantizes ``(L-1)//P`` pages and puts the rest in the tail,
    so an exact multiple of the page size arrives with a full tail page and no
    tokens to spare. Quantizing ``L/P`` pages there instead would look like a
    shortcut and would seal a page the next step still has to append to.
    """
    first = session.states[0]
    exact = [
        row
        for row, length in enumerate(first.seq_lens)
        if length % PAGE_SIZE == 0
    ]
    assert exact, f"no row arrived on a boundary: {first.seq_lens}"
    for row in exact:
        assert first.promotion_pages[row] >= 0, (
            f"row {row} arrived with {first.seq_lens[row]} tokens and sealed "
            "nothing"
        )
        assert first.seqused_fp4[row] == first.seq_lens[row] - PAGE_SIZE
        assert first.seqused_residual[row] == PAGE_SIZE


# --- the state a sealed page leaves behind ---------------------------------


def test_the_fp4_prefix_takes_the_page_over(session: Session):
    """After a crossing the next step must count the page as FP4, once.

    This is the half of correctness the bytes cannot show: a page written into
    the cache that the following step still believes is in the tail is a page
    nobody reads, and the tail it reads instead is about to be overwritten.
    """
    advanced = 0
    for step, (before, after) in enumerate(
        zip(session.states, session.states[1:])
    ):
        later = {
            _prompt_of(after, step + 1, row): (
                after.seqused_fp4[row],
                after.seqused_residual[row],
            )
            for row in range(len(after.seq_lens))
        }
        for row in range(len(before.seq_lens)):
            crossed = before.promotion_pages[row] >= 0
            prefix, residual = later[_prompt_of(before, step, row)]
            grew = prefix - before.seqused_fp4[row]
            assert grew == (PAGE_SIZE if crossed else 0), (
                f"step {step} length {before.seq_lens[row]}: FP4 prefix moved "
                f"by {grew} with crossed={crossed}"
            )
            if crossed:
                # The tail wraps to its start, holding only the new token.
                assert residual == 1, (
                    f"the row at length {before.seq_lens[row]} kept a "
                    f"{residual}-token tail after its page was sealed"
                )
                advanced += 1
    assert advanced == _expected_crossings(), (
        f"{advanced} crossings were followed by a step, wanted "
        f"{_expected_crossings()}"
    )


def test_the_lengths_agree_with_the_page_split_all_the_way_through(
    session: Session,
):
    """``seqused_fp4`` is whole pages, and the two parts sum to the length.

    A property of the control plane rather than of promotion, kept here
    because it is the invariant the tests above read the lengths under, and a
    run where it stopped holding would make them report something else.
    """
    for index, state in enumerate(session.states):
        for row, length in enumerate(state.seq_lens):
            assert state.seqused_fp4[row] == ((length - 1) // PAGE_SIZE) * PAGE_SIZE, (
                f"step {index} row {row}: length {length} split at "
                f"{state.seqused_fp4[row]}"
            )
            assert (
                state.seqused_fp4[row] + state.seqused_residual[row] == length
            )


def test_a_row_crosses_twice(session: Session):
    """Once is a special case; the second time reuses everything.

    The slot still holds the previous page's tokens under the new ones, the
    block table has moved on a column, and the FP4 prefix has to advance from
    a value it already advanced to once.
    """
    blocks: dict[int, list[int]] = {}
    for step, state in enumerate(session.states):
        for row in range(len(state.seq_lens)):
            page = state.promotion_pages[row]
            if page >= 0:
                blocks.setdefault(_prompt_of(state, step, row), []).append(page)
    assert max((len(used) for used in blocks.values()), default=0) >= 2, (
        f"no request sealed two pages during the run: {blocks}"
    )
    for prompt, used in blocks.items():
        assert len(set(used)) == len(used), (
            f"the request from {prompt} tokens sealed two pages into the same "
            f"block: {used}"
        )


# --- what the next decode reads --------------------------------------------


def test_decode_after_a_crossing_matches_the_torch_oracle(session: Session):
    """The page has to be readable, not merely present.

    The oracle rebuilds the whole history from the BF16 the model wrote and
    quantizes it itself, so it is only right about the step after a crossing
    if the FP4 prefix really did take the sealed page over. A promotion that
    wrote correct bytes into a block nothing reads fails here and nowhere
    else.
    """
    oracle = _oracle()
    wanted = {
        (step + 1, _prompt_of(state, step, row))
        for step, state in enumerate(session.states[:-1])
        for row in range(len(state.seq_lens))
        if state.promotion_pages[row] >= 0
    }
    assert len(wanted) >= 4, f"only {len(wanted)} steps follow a crossing"

    cosines: list[tuple[float, int, int, int]] = []
    for decode in session.decodes:
        if (decode.step, decode.prompt) not in wanted:
            continue
        expected = _replay(oracle, session, decode)
        cosines.append(
            (
                F.cosine_similarity(
                    decode.output.float().flatten(),
                    expected.float().flatten(),
                    dim=0,
                ).item(),
                decode.layer,
                decode.step,
                decode.prompt,
            )
        )

    cosines.sort()
    print(
        f"\nworst cosines over {len(cosines)} post-crossing layer-steps: "
        + ", ".join(
            f"{c:.4f} (layer {layer}, step {step}, prompt {prompt})"
            for c, layer, step, prompt in cosines[:5]
        )
    )
    assert len(cosines) == len(wanted) * len(CHECKED_LAYERS), (
        f"{len(cosines)} comparisons for {len(wanted)} steps across "
        f"{len(CHECKED_LAYERS)} layers"
    )
    below = [entry for entry in cosines if entry[0] < MIN_COSINE]
    assert not below, (
        f"{len(below)} of {len(cosines)} are below {MIN_COSINE}, worst "
        f"{below[0][0]:.6f} at layer {below[0][1]} step {below[0][2]} "
        f"prompt {below[0][3]}"
    )


def _replay(oracle, session: Session, decode: Decode) -> torch.Tensor:
    """The oracle's answer for one recorded decode.

    The cache is rebuilt from the BF16 the model wrote: whole pages the oracle
    quantizes itself, and a last page it keeps in BF16, which is the tail.
    """
    seq = decode.seq_len
    pages = -(-seq // PAGE_SIZE)
    device = decode.query.device

    def paged(history: torch.Tensor) -> torch.Tensor:
        tokens = history[:seq].to(device)
        padding = pages * PAGE_SIZE - seq
        if padding:
            tokens = F.pad(tokens, (0, 0, 0, 0, 0, padding))
        return tokens.reshape(pages, PAGE_SIZE, *tokens.shape[1:])

    def ints(*values: int) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    with torch.no_grad():
        expected = oracle._torch_fp4_decode(
            decode.query.unsqueeze(0).unsqueeze(0),
            paged(session.history_key[decode.layer][decode.prompt]),
            paged(session.history_value[decode.layer][decode.prompt]),
            torch.arange(pages, dtype=torch.int32, device=device).reshape(
                1, pages
            ),
            ints(seq),
            ints(pages - 1),
            ints(decode.residual),
            decode.softmax_scale,
        )
    return expected[:, 0]


# --- what it cost ----------------------------------------------------------


def test_promotion_is_launched_once_a_step_whatever_the_batch(
    session: Session,
):
    """Fixed launch count is the design's whole reason for tolerating waste.

    One call per step, on the step after the last layer, whether or not any
    row crossed. Anything that scaled with the batch or the layer count would
    have to ask the device a question first, and asking costs a step.
    """
    assert session.launches == len(session.states) == len(session.host_micros)
    # The prefill step emits the first token, so N tokens take N steps.
    assert session.launches == GENERATED_TOKENS, (
        f"{session.launches} launches over a generation of "
        f"{GENERATED_TOKENS} tokens"
    )
    idle = sum(
        1
        for state in session.states
        if all(page < 0 for page in state.promotion_pages)
    )
    assert idle > session.launches // 2, (
        f"only {idle} of {session.launches} steps promoted nothing, so the "
        "fixed launch was never exercised on an empty step"
    )


def test_only_the_rows_that_crossed_carried_a_destination(session: Session):
    """A fixed launch shape must not turn into fixed work.

    This counts destinations, not writes. That the kernel leaves an idle page
    alone is the kernel suite's to prove, and
    ``test_a_base_table_sends_each_layer_to_its_own_allocation`` does it there
    by poisoning the pages nobody should touch.
    """
    payload = sum(
        1
        for state in session.states
        for page in state.promotion_pages
        if page >= 0
    )
    assert payload == len({(page.step, page.row) for page in session.sealed})
    assert payload == _expected_crossings(), (
        f"{payload} rows were promoted over {session.launches} steps, and the "
        f"prompt lengths allow exactly {_expected_crossings()}"
    )


def test_promotion_costs_a_step_far_less_than_a_layer_loop_would(
    session: Session,
):
    """Recorded for the speed accounting, and gated against the way back.

    The number that matters is the host cost: the two launches are what a step
    waits for before it can go on, and a per-layer loop would put sixty-four
    of them there.
    """
    host = sorted(session.host_micros)
    device = sorted(session.device_micros)
    middle = len(host) // 2
    print(
        f"\npromotion per step: host median {host[middle]:.1f} us, "
        f"p95 {host[int(len(host) * 0.95)]:.1f} us, max {host[-1]:.1f} us; "
        f"device median {device[middle]:.1f} us, max {device[-1]:.1f} us"
    )
    assert host[middle] < MAX_HOST_MICROS, (
        f"promotion costs {host[middle]:.1f} us of host time a step, over "
        f"the {MAX_HOST_MICROS:.0f} us a two-launch design should need"
    )


def test_every_request_was_scheduled_on_every_step(session: Session):
    """The premise everything above is regrouped under.

    Rows are matched to requests by subtracting the step number from the row's
    length, which only names a request if all four are scheduled on every step
    and each grows by exactly one token. Were a request ever left out of a
    step, the concatenated history would splice one request's tokens into
    another's and the oracle would be comparing against a sequence that never
    existed.
    """
    assert len(set(PROMPT_LENGTHS)) == len(PROMPT_LENGTHS), (
        f"two requests start from the same length: {PROMPT_LENGTHS}. Their "
        "histories would be filed under one key and spliced together."
    )
    widths = {len(state.seq_lens) for state in session.states}
    assert widths == {len(PROMPT_LENGTHS)}, (
        f"the batch changed width during the run: {widths}"
    )
    for step, state in enumerate(session.states):
        prompts = sorted(
            _prompt_of(state, step, row) for row in range(len(state.seq_lens))
        )
        assert prompts == sorted(PROMPT_LENGTHS), (
            f"step {step} held {state.seq_lens}, which is not the four prompts "
            f"advanced by {step} tokens"
        )
