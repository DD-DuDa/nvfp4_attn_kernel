"""The tail-slot control plane, against a serial reference.

The kernel resolves slot ownership without branching: it ranks the rows that
need a slot, ranks the unclaimed slots by age, and pairs the ranks. That is not
how anyone would describe the rule, so the reference below implements the
description instead — a dict-shaped table walked row by row, taking the oldest
free slot each time. Agreement between the two is the point of the file.

Three kinds of input drive it. Scripted steps pin the behaviours the design
turns on, and read as documentation of what a step means. A scheduler
simulator generates the combinations nobody thought to script: rows condensed
after a finish, requests left out of a batch and returning at a different row,
block ids recycled into new requests. A separate pair of tests covers the two
properties that are not about output values at all — that a step performs no
host synchronization, and that a steady-state stream never loses a slot.

Needs a GPU, but no model and no vLLM engine, so it runs unconditionally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from nvfp4_vllm.control import (
    ERR_CONTINUATION_PREFILL,
    ERR_DUPLICATE_KEY,
    ERR_NO_FREE_SLOT,
    ERR_PROMOTION_COLUMN,
    ERR_SLOT_LOST,
    ERR_STALE_SLOT_HISTORY,
    FREE_KEY,
    INACTIVE_ROW,
    MAX_SUPPORTED_SLOTS,
    NULL_BLOCK,
    PAGE_SIZE,
    ControlPlane,
)


# The width the scripted tests below are written at. Deliberately narrow: they
# assert on whole rows of output, and eight of anything is readable. The
# ceiling is exercised by the randomized cross-check at the end of the file.
NUM_SLOTS = 8
MAX_TOKENS = 4096
# Columns in the block table the steps below hand to the kernel. Wide enough
# that a promotion column is in range for the lengths used here, and the one
# test that wants it out of range says so.
BLOCK_COLUMNS = 8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@dataclass(frozen=True)
class Step:
    """One scheduler step as the attention metadata describes it.

    ``seq_lens`` counts the tokens this step will have stored once it is done,
    which is how vLLM's model runner computes it.
    """

    block0: list[int]
    seq_lens: list[int]
    query_lens: list[int]
    columns: int = BLOCK_COLUMNS

    @property
    def query_start_loc(self) -> list[int]:
        starts = [0]
        for length in self.query_lens:
            starts.append(starts[-1] + length)
        return starts

    @property
    def block_table(self) -> list[list[int]]:
        """Column 0 is the request's key; the rest are distinct made-up ids.

        Distinct because promotion answers with one of them, and a table of
        zeros would let a wrong column pass for the right one.
        """
        return [
            [key] + [key * 1000 + column for column in range(1, self.columns)]
            for key in self.block0
        ]


def decodes(block0: list[int], seq_lens: list[int]) -> Step:
    """A step in which every row emits one token."""
    return Step(block0, seq_lens, [1] * len(block0))


def prefills(block0: list[int], prompt_lens: list[int]) -> Step:
    """A step in which every row runs a whole prompt."""
    return Step(block0, list(prompt_lens), list(prompt_lens))


class ReferenceSlotTable:
    """What the kernel is supposed to compute, written out the long way."""

    def __init__(self, num_slots: int = NUM_SLOTS, page: int = PAGE_SIZE) -> None:
        self.num_slots = num_slots
        self.page = page
        self.keys = [FREE_KEY] * num_slots
        self.last_seq = [-1] * num_slots
        self.last_seen = [-1] * num_slots
        self.step_index = 0
        self.errors = 0

    def prepare(self, step: Step) -> dict:
        self.step_index += 1
        num_reqs = len(step.block0)
        # A padded row carries a stale block id under a zero length; a dummy
        # batch carries a nonzero length over the null block. Neither is a
        # request, and it takes both tests to say so.
        live = [
            length > 0 and key != NULL_BLOCK
            for length, key in zip(step.seq_lens, step.block0)
        ]
        slot_of = [INACTIVE_ROW] * num_reqs
        taken: set[int] = set()

        seen: set[int] = set()
        for row, key in enumerate(step.block0):
            if not live[row]:
                continue
            if key in seen:
                self.errors |= ERR_DUPLICATE_KEY
            seen.add(key)

        for row, key in enumerate(step.block0):
            if not live[row]:
                continue
            computed = step.seq_lens[row] - step.query_lens[row]
            if step.query_lens[row] > 1 and computed > 0:
                self.errors |= ERR_CONTINUATION_PREFILL
            for slot in range(self.num_slots):
                if self.keys[slot] == key:
                    slot_of[row] = slot
                    taken.add(slot)
                    if computed > 0 and self.last_seq[slot] != computed:
                        self.errors |= ERR_STALE_SLOT_HISTORY
                    break

        for row in range(num_reqs):
            if not live[row] or slot_of[row] != INACTIVE_ROW:
                continue
            if step.seq_lens[row] - step.query_lens[row] > 0:
                self.errors |= ERR_SLOT_LOST
            free = [s for s in range(self.num_slots) if s not in taken]
            if not free:
                self.errors |= ERR_NO_FREE_SLOT
                continue
            victim = min(free, key=lambda s: (self.last_seen[s], s))
            slot_of[row] = victim
            taken.add(victim)

        written: set[int] = set()
        for row, slot in enumerate(slot_of):
            if slot == INACTIVE_ROW or slot in written:
                continue
            written.add(slot)
            self.keys[slot] = step.block0[row]
            self.last_seq[slot] = step.seq_lens[row]
            self.last_seen[slot] = self.step_index

        fp4 = [
            ((length - 1) // self.page) * self.page if alive else 0
            for length, alive in zip(step.seq_lens, live)
        ]
        token_to_slot = [
            slot_of[row] if live[row] else 0
            for row in range(num_reqs)
            for _ in range(step.query_lens[row])
        ]

        # Promotion is answered for the whole table, not just the batch, so
        # that its launch shape does not follow the batch.
        table = step.block_table
        sources = [0] * self.num_slots
        pages = [-1] * self.num_slots
        for row in range(num_reqs):
            if not live[row]:
                continue
            sources[row] = slot_of[row] * self.page
            if step.seq_lens[row] % self.page:
                continue
            column = step.seq_lens[row] // self.page - 1
            if column >= len(table[row]):
                self.errors |= ERR_PROMOTION_COLUMN
                continue
            pages[row] = table[row][column]

        return {
            "row_to_slot": slot_of,
            "token_to_slot": token_to_slot,
            "seqused_fp4": fp4,
            "seqused_residual": [
                length - base if alive else 0
                for length, base, alive in zip(step.seq_lens, fp4, live)
            ],
            "promotion_source_tokens": sources,
            "promotion_pages": pages,
            "error_code": self.errors,
        }


def run(plane: ControlPlane, step: Step) -> dict:
    """Push one step through the kernel and bring the answer back to host."""
    device = plane.device

    def to_gpu(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    # Several columns wide so the kernel's strided reads are exercised rather
    # than accidentally landing on a contiguous vector.
    outputs = plane.prepare(
        block_table=to_gpu(step.block_table),
        seq_lens=to_gpu(step.seq_lens),
        query_start_loc=to_gpu(step.query_start_loc),
        num_reqs=len(step.block0),
        num_actual_tokens=sum(step.query_lens),
    )
    return {
        "row_to_slot": outputs.row_to_slot.tolist(),
        "token_to_slot": outputs.token_to_slot.tolist(),
        "seqused_fp4": outputs.seqused_fp4.tolist(),
        "seqused_residual": outputs.seqused_residual.tolist(),
        "promotion_source_tokens": outputs.promotion_source_tokens.tolist(),
        "promotion_pages": outputs.promotion_pages.tolist(),
        "error_code": int(outputs.error_code.item()),
    }


@pytest.fixture
def plane() -> ControlPlane:
    return ControlPlane(
        max_num_seqs=NUM_SLOTS,
        max_num_batched_tokens=MAX_TOKENS,
        device="cuda",
    )


@pytest.fixture
def reference() -> ReferenceSlotTable:
    return ReferenceSlotTable()


def check(plane: ControlPlane, reference: ReferenceSlotTable, *steps: Step) -> dict:
    """Run a sequence through both implementations, returning the last answer."""
    actual = {}
    for index, step in enumerate(steps):
        actual = run(plane, step)
        expected = reference.prepare(step)
        assert actual == expected, f"step {index + 1} diverged"
    return actual


# --- scripted behaviour ----------------------------------------------------


def test_lengths_split_at_the_last_whole_page(plane, reference):
    # 300 tokens is two whole pages plus 44, and those 44 are the tail. A
    # sequence landing exactly on a page boundary keeps a full tail page rather
    # than an empty one, because the FP4 side is only written on promotion.
    answer = check(plane, reference, prefills([11, 12, 13], [300, 128, 256]))
    assert answer["seqused_fp4"] == [256, 0, 128]
    assert answer["seqused_residual"] == [44, 128, 128]
    # Rows 1 and 2 fill their tail page exactly, so promotion has somewhere to
    # put it: the block holding logical page 0 for row 1, page 1 for row 2.
    # Both ids are the row's own, from the column its length picks out.
    assert answer["promotion_pages"][:3] == [-1, 12, 13001]
    assert answer["promotion_source_tokens"][:3] == [0, 128, 256]


def test_a_slot_follows_its_request_across_a_row_move(plane, reference):
    # vLLM condenses its batch when a request finishes: the last row slides
    # into the hole. Keying on the row index would hand the mover somebody
    # else's tail, which is why the key is block_table[row, 0].
    check(plane, reference, prefills([11, 22, 33], [10, 20, 30]))
    answer = check(plane, reference, decodes([11, 33], [11, 31]))
    assert answer["row_to_slot"] == [0, 2]


def test_a_request_left_out_of_a_step_keeps_its_slot(plane, reference):
    # The scheduler drops a live request when its token budget runs out.
    # Treating absence as death would discard the only copy of that request's
    # most recent keys and values.
    check(plane, reference, prefills([11, 22], [10, 20]))
    check(plane, reference, decodes([11], [11]))
    check(plane, reference, decodes([11], [12]))
    answer = check(plane, reference, decodes([11, 22], [13, 21]))
    assert answer["row_to_slot"] == [0, 1]
    assert answer["error_code"] == 0


def test_a_new_request_takes_the_slot_unseen_longest(plane, reference):
    # The table fills, then key 1 sits out two steps and key 2 only one, so
    # key 1's slot is the one unseen longest when a ninth request arrives.
    check(plane, reference, prefills([1, 2, 3, 4, 5, 6, 7, 8], [10] * 8))
    check(plane, reference, decodes([3, 4, 5, 6, 7, 8], [11] * 6))
    check(plane, reference, decodes([2, 3, 4, 5, 6, 7, 8], [11] + [12] * 6))
    answer = check(plane, reference, prefills([99], [64]))
    assert answer["row_to_slot"] == [0]


def test_recycled_block_ids_do_not_confuse_a_new_request(plane, reference):
    # A finished request's physical block goes back to vLLM's pool and can be
    # handed to a new one. The new owner has no history, so inheriting the slot
    # is correct and is not reported.
    check(plane, reference, prefills([11, 22], [10, 20]))
    answer = check(plane, reference, Step([11, 22], [11, 40], [1, 40]))
    assert answer["row_to_slot"] == [0, 1]
    assert answer["error_code"] == 0


def test_every_token_is_labelled_with_its_row_slot(plane, reference):
    answer = check(plane, reference, Step([11, 22], [3, 300], [3, 300]))
    assert answer["token_to_slot"] == [0, 0, 0] + [1] * 300


def test_dead_rows_read_as_attending_to_nothing(plane):
    # The buffers are full width, so a consumer that ignores num_reqs must
    # still see something inert past the live rows.
    run(plane, prefills([11, 22, 33, 44], [10, 20, 30, 40]))
    run(plane, decodes([11], [11]))
    assert plane.row_to_slot.tolist()[1:] == [INACTIVE_ROW] * (NUM_SLOTS - 1)
    assert plane.seqused_fp4.tolist()[1:] == [0] * (NUM_SLOTS - 1)
    assert plane.seqused_residual.tolist()[1:] == [0] * (NUM_SLOTS - 1)
    assert plane.promotion_pages.tolist()[1:] == [INACTIVE_ROW] * (NUM_SLOTS - 1)
    assert plane.promotion_source_tokens.tolist()[1:] == [0] * (NUM_SLOTS - 1)


def test_a_padded_row_is_not_a_request(plane, reference):
    # Replaying a full cuda graph means padding the batch out to the captured
    # width. vLLM zeroes the sequence lengths of the rows it adds but leaves
    # their block table entries alone, so row 3 below still reads as the
    # request that finished last step. Only the length says the row is empty.
    check(plane, reference, prefills([11, 22, 33], [10, 20, 30]))
    answer = check(
        plane, reference, Step([11, 22, 33, 33], [11, 21, 0, 0], [1, 1, 0, 0])
    )
    assert answer["row_to_slot"] == [0, 1, INACTIVE_ROW, INACTIVE_ROW]
    assert answer["error_code"] == 0
    # 33's slot survives untouched, so the request could still be resumed.
    answer = check(plane, reference, decodes([11, 22, 33], [12, 22, 31]))
    assert answer["row_to_slot"] == [0, 1, 2]
    assert answer["error_code"] == 0


def test_a_dummy_batch_over_the_null_block_is_not_a_request(plane, reference):
    # vLLM sizes the KV cache by running one maximum-width batch through the
    # model and watching the memory. That batch reports a full slate of long
    # sequences over a block table nobody has written, so every row keys on
    # block 0. Reading those rows as requests would take every slot and, since
    # they all carry the same key, report a duplicate that never happened.
    check(plane, reference, prefills([11, 22], [10, 20]))
    answer = check(plane, reference, prefills([NULL_BLOCK] * 8, [512] * 8))
    assert answer["row_to_slot"] == [INACTIVE_ROW] * 8
    assert answer["error_code"] == 0
    # The real requests keep their slots and their history across it.
    answer = check(plane, reference, decodes([11, 22], [11, 21]))
    assert answer["row_to_slot"] == [0, 1]
    assert answer["error_code"] == 0


# --- scripted failures -----------------------------------------------------


def test_a_split_prompt_is_reported(plane, reference):
    # Guarded against at configuration time by requiring chunked prefill off.
    # If that ever stops holding, the second chunk shows up here.
    check(plane, reference, prefills([11], [256]))
    answer = check(plane, reference, Step([11], [512], [256]))
    assert answer["error_code"] & ERR_CONTINUATION_PREFILL


def test_a_lost_slot_is_reported(plane, reference):
    # Eight live requests, one of them sitting out long enough to become the
    # oldest, then a ninth arrives and takes its slot. When the victim returns
    # it has history and nowhere to put it.
    check(plane, reference, prefills([1, 2, 3, 4, 5, 6, 7, 8], [10] * 8))
    for length in range(11, 15):
        check(plane, reference, decodes([2, 3, 4, 5, 6, 7, 8], [length] * 7))
    check(plane, reference, Step([2, 3, 4, 5, 6, 7, 8, 99], [15] * 7 + [8], [1] * 7 + [8]))
    answer = check(plane, reference, decodes([1, 2, 3, 4, 5, 6, 7], [11] + [16] * 6))
    assert answer["error_code"] & ERR_SLOT_LOST


def test_a_promotion_column_past_the_block_table_is_reported(plane, reference):
    # Not reachable through vLLM, which allocated the block holding the row's
    # last token before the step ran. It is checked because the gather is
    # masked, not bounds-checked: a column one past the end reads the next
    # row's block id, and promotion would then write a whole page of one
    # request's history over another's. Nothing downstream could tell.
    answer = check(
        plane, reference, Step([11], [1024], [1024], columns=4)
    )
    assert answer["error_code"] & ERR_PROMOTION_COLUMN
    assert answer["promotion_pages"][0] == INACTIVE_ROW


def test_a_recycled_key_carrying_history_is_reported(plane, reference):
    # Not reachable through vLLM: a matched key means either the same request,
    # whose length is continuous by construction, or a recycled block, whose
    # new owner has no history. The step below is fabricated, and exists so the
    # detector is known to work if either assumption stops holding.
    check(plane, reference, prefills([11], [10]))
    answer = check(plane, reference, decodes([11], [50]))
    assert answer["error_code"] & ERR_STALE_SLOT_HISTORY


def test_two_live_rows_sharing_a_block_are_reported(plane, reference):
    # Also not reachable through vLLM, which never gives one physical block to
    # two unfinished requests. The matching reduction answers with the lower
    # slot rather than the sum of both, so the outputs stay in range and the
    # step is attributable instead of scribbling on an unrelated slot.
    check(plane, reference, prefills([11, 22], [10, 20]))
    answer = check(plane, reference, decodes([11, 11], [11, 11]))
    assert answer["row_to_slot"] == [0, 0]
    assert answer["error_code"] == ERR_DUPLICATE_KEY


def test_errors_are_sticky_until_reset(plane, reference):
    check(plane, reference, prefills([11], [256]))
    check(plane, reference, Step([11], [512], [256]))
    answer = check(plane, reference, decodes([11], [513]))
    assert answer["error_code"] & ERR_CONTINUATION_PREFILL
    plane.reset()
    assert int(plane.error_code.item()) == 0


def test_more_rows_than_slots_is_refused_on_the_host(plane):
    # Cheaper to catch here than to encode in the kernel: the caller cannot do
    # anything useful with a device-side flag, and the guardrails already cap
    # max_num_seqs at the table width.
    with pytest.raises(ValueError, match="tail slots"):
        run(plane, decodes(list(range(NUM_SLOTS + 1)), [1] * (NUM_SLOTS + 1)))


# --- properties ------------------------------------------------------------


def test_a_step_does_not_synchronize(plane):
    # Any host synchronization inside prepare() would drain the pipeline every
    # decode step. The kernel is the only place a read-back could hide, so the
    # inputs are staged on device before the mode is armed.
    device = plane.device
    step = decodes([11, 22, 33], [10, 20, 30])
    block_table = torch.tensor(
        [[key, 0, 0] for key in step.block0], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(step.seq_lens, dtype=torch.int32, device=device)
    starts = torch.tensor(step.query_start_loc, dtype=torch.int32, device=device)

    plane.prepare(  # compile before arming; compilation itself synchronizes
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=starts,
        num_reqs=3,
        num_actual_tokens=3,
    )
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(4):
            plane.prepare(
                block_table=block_table,
                seq_lens=seq_lens,
                query_start_loc=starts,
                num_reqs=3,
                num_actual_tokens=3,
            )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_a_step_can_be_captured_in_a_cuda_graph(plane):
    # Not used yet — cudagraph_mode is guarded to NONE — but capture is the
    # strictest available proof that a step is pure device work, and it is why
    # this kernel is Triton rather than a chain of torch ops.
    device = plane.device
    block_table = torch.tensor([[11, 0, 0], [22, 0, 0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    def step() -> None:
        # num_reqs and num_actual_tokens are host scalars, so a graph bakes
        # them in. Enabling graphs would mean moving them onto the device.
        plane.prepare(
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=starts,
            num_reqs=2,
            num_actual_tokens=2,
        )

    step()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        step()

    before = int(plane.step.item())
    graph.replay()
    torch.cuda.synchronize()
    assert int(plane.step.item()) == before + 1
    assert plane.row_to_slot.tolist()[:2] == [0, 1]


def test_a_steady_stream_never_loses_a_slot(plane):
    # The reclamation rule is lazy, so slots are only ever taken from requests
    # that have gone quiet. When every live request is scheduled every step —
    # the normal case — nothing can be taken from a live request, however many
    # times the table turns over.
    simulator = Simulator(random.Random(0), skip_probability=0.0)
    for _ in range(400):
        step = simulator.step()
        if step.block0:
            run(plane, step)
    assert int(plane.error_code.item()) == 0
    assert simulator.retired > 3 * NUM_SLOTS, "the table never turned over"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("width", [NUM_SLOTS, MAX_SUPPORTED_SLOTS])
def test_the_kernel_agrees_with_the_reference_under_random_scheduling(seed, width):
    """The one test that runs the table at its declared ceiling.

    Width is the kernel's most load-bearing constexpr: it sets BLOCK, and
    every pairwise matrix in there is BLOCK by BLOCK. The scripted tests above
    all sit at eight, where a rank-pairing bug has only a few places to hide,
    so the ceiling gets its coverage here — against the reference model, over
    random scheduling, with the block pool widened to match so the table
    actually fills up.
    """
    plane = ControlPlane(
        max_num_seqs=width,
        max_num_batched_tokens=MAX_TOKENS,
        device="cuda",
    )
    reference = ReferenceSlotTable(num_slots=width)
    simulator = Simulator(
        random.Random(seed),
        skip_probability=0.25,
        num_slots=width,
        block_pool=width + 4,
    )
    crossings = 0
    busiest = 0
    for index in range(300):
        step = simulator.step()
        if not step.block0:
            continue
        answer = run(plane, step)
        assert answer == reference.prepare(step), f"step {index} diverged"
        crossings += sum(page >= 0 for page in answer["promotion_pages"])
        busiest = max(busiest, len(step.block0))
    assert crossings, "no row filled a tail page, so promotion agreed vacuously"
    assert busiest > width // 2, (
        f"the widest step held {busiest} of {width} rows, so most of the "
        "table was never asked to do anything"
    )


class Simulator:
    """A scheduler stripped down to what the slot table can observe.

    Requests arrive, run their prompt in one step, decode for a while and
    finish. Finishing condenses the batch the way ``InputBatch.condense`` does
    and returns the physical block to a pool small enough that ids come back
    around. A running request can be left out of a step, which is how vLLM
    behaves when its token budget runs out.
    """

    def __init__(
        self,
        rng: random.Random,
        *,
        skip_probability: float,
        num_slots: int = NUM_SLOTS,
        block_pool: int = 12,
        max_prompt: int = 400,
        max_output: int = 40,
    ) -> None:
        self.rng = rng
        self.skip_probability = skip_probability
        self.num_slots = num_slots
        self.max_prompt = max_prompt
        self.max_output = max_output
        # Block 0 is vLLM's null block and never belongs to a request.
        self.free_blocks = list(range(1, block_pool + 1))
        self.rows: list[dict] = []
        self.retired = 0

    def step(self) -> Step:
        for row in reversed(range(len(self.rows))):
            if self.rows[row]["remaining"] == 0:
                self.free_blocks.append(self.rows[row]["block0"])
                self.rows[row] = self.rows[-1]
                self.rows.pop()
                self.retired += 1

        while (
            len(self.rows) < self.num_slots
            and self.free_blocks
            and self.rng.random() < 0.5
        ):
            self.rows.append(
                {
                    "block0": self.free_blocks.pop(
                        self.rng.randrange(len(self.free_blocks))
                    ),
                    "seq_len": 0,
                    "remaining": self.rng.randint(1, self.max_output),
                    "started": False,
                }
            )

        block0, seq_lens, query_lens = [], [], []
        for request in self.rows:
            if request["started"] and self.rng.random() < self.skip_probability:
                continue
            if request["started"]:
                query_len = 1
            else:
                query_len = self.rng.randint(1, self.max_prompt)
                request["started"] = True
            request["seq_len"] += query_len
            request["remaining"] -= 1
            block0.append(request["block0"])
            seq_lens.append(request["seq_len"])
            query_lens.append(query_len)

        return Step(block0, seq_lens, query_lens)
