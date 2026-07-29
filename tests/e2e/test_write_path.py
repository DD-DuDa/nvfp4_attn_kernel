"""What a live engine actually leaves in the NVFP4 cache and the BF16 tail.

The check is byte equality, not similarity. Every full page is compared against
what the packed quantizer produces from the same tokens, and every tail token
against the BF16 the layer handed over, so the only thing being tested is
placement: whether each token reached the page, block, slot, and offset it
belongs in. Numerical quality is the kernel suite's job, and it is the same
quantizer either way.

Prompt lengths straddle every page boundary that matters — just under, exactly
on, and just over one and eight pages — because that is where a page-splitting
bug hides. The last layer is checked alongside the first so that a mistake in
which layer owns which tail slice cannot pass.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch

from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_pages,
    quantize_value_pages,
)
from nvfp4_vllm import layout


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8
PROMPT_LENGTHS = (100, 128, 129, 900, 1024, 1025)


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass
class Capture:
    """One layer's view of one step, kept so it can be checked afterwards."""

    layer_index: int
    key: torch.Tensor
    value: torch.Tensor
    query_start_loc: list[int]
    seq_lens: list[int]
    seqused_fp4: list[int]
    row_to_slot: list[int]
    block_table: torch.Tensor
    written_blocks: torch.Tensor
    written_pages: torch.Tensor
    tail_key: torch.Tensor
    tail_value: torch.Tensor


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


@pytest.fixture(scope="module")
def captures():
    _require_sm100()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import nvfp4_vllm.impl as impl_module

    recorded: list[Capture] = []
    original = impl_module.NVFP4Impl.do_kv_cache_update

    def record(self, layer, key, value, kv_cache, slot_mapping):
        original(self, layer, key, value, kv_cache, slot_mapping)
        if self.runtime is None or self.layer_index not in (0, 31):
            return
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

        metadata, _, _, _ = get_attention_context(layer.layer_name)
        if metadata is None:
            return
        tokens = metadata.num_actual_tokens
        destinations = metadata.destination_pages
        written = destinations[destinations >= 0]
        recorded.append(
            Capture(
                layer_index=self.layer_index,
                key=key[:tokens].clone(),
                value=value[:tokens].clone(),
                query_start_loc=metadata.query_start_loc.tolist(),
                seq_lens=metadata.seq_lens.tolist(),
                seqused_fp4=metadata.seqused_fp4.tolist(),
                row_to_slot=metadata.row_to_slot.tolist(),
                block_table=metadata.block_table.clone(),
                written_blocks=written.clone(),
                # The cache is far too large to snapshot, so only the blocks
                # this step claims to have filled are kept.
                written_pages=kv_cache[written.long()].clone(),
                tail_key=self.runtime.tail_key[self.layer_index].clone(),
                tail_value=self.runtime.tail_value[self.layer_index].clone(),
            )
        )

    impl_module.NVFP4Impl.do_kv_cache_update = record
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
        recorded.clear()
        llm.generate(
            [
                TokensPrompt(prompt_token_ids=list(range(10, 10 + length)))
                for length in PROMPT_LENGTHS
            ],
            SamplingParams(max_tokens=2, ignore_eos=True, temperature=0.0),
        )
        yield [capture for capture in recorded if capture.written_blocks.numel()
               or capture.tail_key.numel()]
    finally:
        impl_module.NVFP4Impl.do_kv_cache_update = original
        del llm


def _same_bytes(written: torch.Tensor, expected: torch.Tensor) -> bool:
    """Bit-pattern equality, which is what placement means for the tail.

    Not ``torch.equal``: what these layers receive is not a healthy forward
    pass — attention contributes nothing until the read path lands — and a
    value that compares unequal to itself would make a faithful copy look like
    a misplaced one.
    """
    return torch.equal(
        written.reshape(-1).view(torch.int16),
        expected.reshape(-1).view(torch.int16),
    )


def _prefill_rows(capture: Capture):
    """Rows of this step that arrived as a fresh prompt, with their tokens."""
    starts = capture.query_start_loc
    for row, slot in enumerate(capture.row_to_slot):
        start, end = starts[row], starts[row + 1]
        if slot >= 0 and end - start > 1:
            yield row, slot, start, end


def test_several_prompts_are_written_in_one_step(captures):
    """The interesting case has to actually occur, not just be hoped for.

    One prefill per step would make every check below trivially satisfiable by
    a kernel that ignores the row index entirely. vLLM batches prompts, so the
    per-row source offsets and destination blocks are genuinely interleaved —
    and if a scheduler change ever stopped that, the checks would quietly
    weaken instead of failing.
    """
    widest = max(len(list(_prefill_rows(capture))) for capture in captures)
    assert widest > 1, (
        "no step carried more than one prefill row, so nothing here "
        "distinguishes per-row placement from single-row placement"
    )


def test_every_full_page_holds_the_quantizer_s_own_bytes(captures):
    """A page in the cache must equal the packed quantizer's output for it.

    Both sides run the same kernel, so any difference means the tokens, the
    block, or the region offset was wrong.
    """
    compared = 0
    for capture in captures:
        if not capture.written_blocks.numel():
            continue
        carved = layout.carve(
            capture.written_pages, capture.key.shape[1], capture.key.shape[2]
        )
        position = {
            int(block): index
            for index, block in enumerate(capture.written_blocks.tolist())
        }
        for row, _, start, _ in _prefill_rows(capture):
            pages = capture.seqused_fp4[row] // PAGE_SIZE
            if not pages:
                continue
            windows = torch.stack(
                [
                    capture.key[start + page * PAGE_SIZE :][:PAGE_SIZE]
                    for page in range(pages)
                ]
            ).contiguous()
            values = torch.stack(
                [
                    capture.value[start + page * PAGE_SIZE :][:PAGE_SIZE]
                    for page in range(pages)
                ]
            ).contiguous()
            expected_key, expected_key_sf = quantize_key_pages(windows)
            expected_value, expected_value_sf = quantize_value_pages(values)

            where = [
                position[int(capture.block_table[row, page])]
                for page in range(pages)
            ]
            assert torch.equal(carved[0][where], expected_key), (
                f"layer {capture.layer_index} row {row}: packed K differs"
            )
            assert torch.equal(carved[1][where], expected_key_sf), (
                f"layer {capture.layer_index} row {row}: K scales differ"
            )
            assert torch.equal(carved[2][where], expected_value), (
                f"layer {capture.layer_index} row {row}: packed V differs"
            )
            assert torch.equal(carved[3][where], expected_value_sf), (
                f"layer {capture.layer_index} row {row}: V scales differ"
            )
            compared += pages

    # (length - 1) // 128 pages per prompt, so 0, 0, 1, 7, 7 and 8, on each of
    # the two layers. Asserted so that a step going unrecorded cannot turn this
    # into a test that checks nothing.
    expected = 2 * sum((length - 1) // PAGE_SIZE for length in PROMPT_LENGTHS)
    assert compared == expected, f"checked {compared} pages, wanted {expected}"


def test_the_tail_holds_the_tokens_no_page_claimed(captures):
    """Whatever a page did not take must sit in the row's slot, verbatim.

    The tail is BF16, so this is exact equality against what the layer handed
    over, and it pins the slot and the offset within it as well as the bytes.
    """
    checked = 0
    for capture in captures:
        for row, slot, start, end in _prefill_rows(capture):
            quantized = capture.seqused_fp4[row]
            residual = capture.seq_lens[row] - quantized
            assert 1 <= residual <= PAGE_SIZE, residual
            expected_key = capture.key[end - residual : end]
            expected_value = capture.value[end - residual : end]
            assert _same_bytes(
                capture.tail_key[slot, :residual], expected_key
            ), f"layer {capture.layer_index} row {row}: tail K differs"
            assert _same_bytes(
                capture.tail_value[slot, :residual], expected_value
            ), f"layer {capture.layer_index} row {row}: tail V differs"
            checked += 1
    assert checked == 2 * len(PROMPT_LENGTHS), checked


def test_a_page_boundary_prompt_puts_a_whole_page_in_the_tail(captures):
    """A prompt of exactly N pages keeps its last page in BF16, not in FP4.

    The split is by completed page rather than by token count, so a length
    that is a multiple of 128 must quantize N-1 pages and leave 128 tokens in
    the tail. Getting this backwards would quantize tokens the next step still
    needs to append to.
    """
    seen = {}
    for capture in captures:
        for row, _, start, end in _prefill_rows(capture):
            length = capture.seq_lens[row]
            seen[length] = (
                capture.seqused_fp4[row] // PAGE_SIZE,
                length - capture.seqused_fp4[row],
            )

    assert seen == {
        100: (0, 100),
        128: (0, 128),
        129: (1, 1),
        900: (7, 4),
        1024: (7, 128),
        1025: (8, 1),
    }, seen


def test_the_decode_token_lands_after_the_prefill_tail(captures):
    """A step that adds one token extends the tail rather than rewriting it.

    Nothing in the step says where in the sequence that token belongs; the
    kernel derives it, and an off-by-one here would silently overwrite the
    prompt's last token.

    A prompt that ended exactly on a page boundary wraps its tail back to
    offset zero here, losing what promotion has not yet moved into FP4. That
    is S8's gap, not a placement bug, and the prefill assertions above are
    unaffected because they read a snapshot taken before this step.
    """
    decode_steps = [
        capture
        for capture in captures
        if capture.query_start_loc[-1] == len(capture.seq_lens)
    ]
    assert decode_steps, "the generate call produced no decode step to check"

    for capture in decode_steps:
        for row, slot in enumerate(capture.row_to_slot):
            if slot < 0:
                continue
            offset = capture.seq_lens[row] - 1 - capture.seqused_fp4[row]
            assert _same_bytes(
                capture.tail_key[slot, offset], capture.key[row]
            ), f"layer {capture.layer_index} row {row}: decode K misplaced"
            assert _same_bytes(
                capture.tail_value[slot, offset], capture.value[row]
            ), f"layer {capture.layer_index} row {row}: decode V misplaced"
