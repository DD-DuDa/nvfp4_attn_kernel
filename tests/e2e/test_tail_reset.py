"""Clearing a tail slot before a new request moves into it.

A slot outlives the request that filled it. The decode kernel reads a row's
whole tail page and masks the positions past its length by weighting them zero,
which is a multiply, so a NaN left behind past the length by a previous tenant
is not masked at all — it takes the new tenant's entire output with it. The
first test here shows that directly, so the rest have a reason to exist.

Needs a GPU, but no model and no vLLM engine, so it runs unconditionally. The
one test that runs the decode kernel needs SM100 as well.
"""

from __future__ import annotations

import pytest
import torch

from nvfp4_vllm.write import reset_new_request_tails


PAGE_SIZE = 128
LAYERS = 3
SLOTS = 4
HEADS_KV = 8
HEAD_DIM = 128


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _tail() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (LAYERS, SLOTS, PAGE_SIZE, HEADS_KV, HEAD_DIM)
    generator = torch.Generator(device="cuda").manual_seed(0x7A11)
    key = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    return key, value


def _metadata(
    rows: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn ``(slot, query_len, seq_len)`` per row into the three step arrays."""
    slots = [slot for slot, _, _ in rows]
    starts = [0]
    for _, query_len, _ in rows:
        starts.append(starts[-1] + query_len)
    return (
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(
            [seq_len for _, _, seq_len in rows], dtype=torch.int32, device="cuda"
        ),
        torch.tensor(slots, dtype=torch.int32, device="cuda"),
    )


def test_a_stale_nan_past_the_length_reaches_the_output():
    """Why the reset exists: the mask does not stop a NaN, it multiplies it.

    Only the value page matters. A key past the length has its score
    overwritten with -inf before anything is done with it, so its NaN dies
    there; a value is weighted by the zero that -inf became and survives.
    """
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required to run the decode kernel")
    pytest.importorskip("cutlass")

    from nvfp4_decode_kernel import _quantize, fp4_decode

    live = 5
    torch.manual_seed(0xDEC0DE)
    query = torch.randn(1, 32, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    pages = (
        torch.randn(4, PAGE_SIZE, HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    key_fp4, key_scales = _quantize.quantize_key_pages(pages)
    value_fp4, value_scales = _quantize.quantize_value_pages(pages)
    seqused_residual = torch.full((1,), live, dtype=torch.int32, device="cuda")

    def decode(tail_value: torch.Tensor) -> torch.Tensor:
        return fp4_decode(
            query,
            key_fp4,
            key_scales,
            value_fp4,
            value_scales,
            torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            torch.full((1,), PAGE_SIZE, dtype=torch.int32, device="cuda"),
            residual_key_pages_bf16=pages,
            residual_value_pages_bf16=tail_value,
            residual_page_ids=torch.tensor([2], dtype=torch.int32, device="cuda"),
            seqused_residual=seqused_residual,
            has_bf16=seqused_residual > 0,
        )

    assert torch.isfinite(decode(pages.clone())).all()

    poisoned = pages.clone()
    poisoned[2, live:] = float("nan")
    assert torch.isnan(decode(poisoned)).all()

    cleared = pages.clone()
    cleared[2, live:] = 0.0
    assert torch.isfinite(decode(cleared)).all()


def test_a_new_request_starts_from_a_clear_page():
    key, value = _tail()
    key[:, 2] = float("nan")
    value[:, 2] = float("nan")

    # Row 0 arrives with a prompt and nothing computed, so slot 2 is changing
    # hands this step.
    query_start_loc, seq_lens, row_to_slot = _metadata([(2, 133, 133)])
    reset_new_request_tails(
        tail_key=key,
        tail_value=value,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        row_to_slot=row_to_slot,
    )

    assert (key[:, 2] == 0).all()
    assert (value[:, 2] == 0).all()


def test_a_single_token_prompt_is_a_new_request_too():
    """The batch reordering files it among the decodes; it is still new."""
    key, value = _tail()
    key[:, 1] = float("nan")
    value[:, 1] = float("nan")

    query_start_loc, seq_lens, row_to_slot = _metadata([(1, 1, 1)])
    reset_new_request_tails(
        tail_key=key,
        tail_value=value,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        row_to_slot=row_to_slot,
    )

    assert (key[:, 1] == 0).all()
    assert (value[:, 1] == 0).all()


def test_a_continuing_request_keeps_its_tail():
    """The opposite failure: clearing a live request's history mid-flight."""
    key, value = _tail()
    before_key = key.clone()
    before_value = value.clone()

    # One decode step of a request that already has 200 tokens behind it.
    query_start_loc, seq_lens, row_to_slot = _metadata([(3, 1, 201)])
    reset_new_request_tails(
        tail_key=key,
        tail_value=value,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        row_to_slot=row_to_slot,
    )

    assert torch.equal(key, before_key)
    assert torch.equal(value, before_value)


def test_only_the_arriving_rows_slot_is_touched():
    key, value = _tail()
    before_key = key.clone()
    before_value = value.clone()

    # A new request at row 0 next to a decode at row 1 and a dead row at 2.
    query_start_loc, seq_lens, row_to_slot = _metadata(
        [(0, 64, 64), (3, 1, 300), (-1, 0, 0)]
    )
    reset_new_request_tails(
        tail_key=key,
        tail_value=value,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        row_to_slot=row_to_slot,
    )

    assert (key[:, 0] == 0).all()
    assert (value[:, 0] == 0).all()
    for slot in (1, 2, 3):
        assert torch.equal(key[:, slot], before_key[:, slot])
        assert torch.equal(value[:, slot], before_value[:, slot])


def test_every_layer_is_cleared():
    """A slot's history has to end for the whole model at the same moment."""
    key, value = _tail()
    key[:] = float("nan")
    value[:] = float("nan")

    query_start_loc, seq_lens, row_to_slot = _metadata([(1, 10, 10)])
    reset_new_request_tails(
        tail_key=key,
        tail_value=value,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        row_to_slot=row_to_slot,
    )

    for layer in range(LAYERS):
        assert (key[layer, 1] == 0).all(), f"layer {layer} kept its stale page"
        assert (value[layer, 1] == 0).all(), f"layer {layer} kept its stale page"
