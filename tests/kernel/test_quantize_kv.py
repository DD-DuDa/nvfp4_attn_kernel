"""Indexed K/V page quantization.

The paged variant reads 128 tokens starting anywhere in a flat activation
buffer and writes them to any page of a destination whose pages may be spread
apart. Every case here is checked byte for byte against the densely packed
quantizer given the same tokens, because a serving cache and its reference
decode must agree exactly, not approximately.
"""

from __future__ import annotations

import pytest
import torch

from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_pages,
    quantize_key_tokens_into,
    quantize_value_pages,
    quantize_value_tokens_into,
)


PAGE_SIZE = 128
HEAD_DIM = 128
HEADS = 8
SCALE_BYTES_PER_HEAD = 1024


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")
    torch.manual_seed(0)


def _spread_pages(shape, strides, pages, pitch, dtype=torch.uint8):
    """Allocate pages `pitch` bytes apart, wider than the page itself needs.

    Padding between pages stands in for whatever else a real cache page carries.
    Filling it with a non-zero pattern makes an out-of-page write visible.
    """
    buffer = torch.full(
        (pages, pitch), 0xA5, dtype=torch.uint8, device="cuda"
    )
    view = buffer.view(dtype).as_strided((pages,) + shape, (pitch,) + strides, 0)
    return buffer, view


def _key_destination(pages, pitch=100 * 1024):
    packed = _spread_pages(
        (PAGE_SIZE, HEADS, HEAD_DIM // 2),
        (HEADS * HEAD_DIM // 2, HEAD_DIM // 2, 1),
        pages,
        pitch,
    )
    scales = _spread_pages(
        (32, 4, 1, 4, 2, HEADS),
        (16, 4, 1024, 1, 512, 1024),
        pages,
        pitch,
    )
    return packed, scales


def _value_destination(pages, pitch=100 * 1024):
    packed = _spread_pages(
        (HEADS, HEAD_DIM, PAGE_SIZE // 2),
        (HEAD_DIM * PAGE_SIZE // 2, PAGE_SIZE // 2, 1),
        pages,
        pitch,
    )
    scales = _spread_pages(
        (32, 4, 1, 4, 2, HEADS),
        (16, 4, 1024, 1, 512, 1024),
        pages,
        pitch,
    )
    return packed, scales


def _tokens(count):
    return torch.randn(
        count, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    ) * 0.3


def _as_int32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param(
            (quantize_key_pages, quantize_key_tokens_into, _key_destination),
            id="key",
        ),
        pytest.param(
            (
                quantize_value_pages,
                quantize_value_tokens_into,
                _value_destination,
            ),
            id="value",
        ),
    ],
)
def test_unaligned_windows_land_byte_for_byte_where_the_indices_say(direction):
    """The whole contract in one case: any window, any page, exact bytes.

    Sources start off page boundaries and out of order, destinations are
    written in yet another order, and the pages sit far apart. The expected
    bytes come from the dense quantizer fed the same windows materialized by
    hand, so this pins the paged path to the packed one rather than to itself.
    """
    dense_quantize, indexed_quantize, make_destination = direction
    starts = [0, 37, 1000, 128, 511]
    destinations = [4, 1, 0, 3, 2]
    tokens = _tokens(2048)

    (_, packed), (_, scales) = make_destination(len(starts))
    indexed_quantize(
        tokens, packed, scales, _as_int32(starts), _as_int32(destinations)
    )

    windows = torch.stack(
        [tokens[start : start + PAGE_SIZE] for start in starts]
    )
    expected_packed, expected_scales = dense_quantize(windows)

    order = _as_int32(destinations).long()
    assert torch.equal(packed[order], expected_packed)
    assert torch.equal(
        scales.view(torch.float8_e4m3fn)[order], expected_scales
    )


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param(
            (quantize_key_tokens_into, _key_destination), id="key"
        ),
        pytest.param(
            (quantize_value_tokens_into, _value_destination), id="value"
        ),
    ],
)
def test_a_negative_destination_page_writes_nothing(direction):
    """Inactive grid slots must be inert, not merely harmless.

    A fixed launch shape covering a varying batch means most slots do no work,
    and the pages they would have touched belong to other sequences.
    """
    indexed_quantize, make_destination = direction
    tokens = _tokens(512)

    (_, packed), (_, scales) = make_destination(3)
    untouched_packed = packed.clone()
    untouched_scales = scales.clone()

    indexed_quantize(
        tokens, packed, scales, _as_int32([0, 128, 256]), _as_int32([-1, 1, -1])
    )

    assert torch.equal(packed[[0, 2]], untouched_packed[[0, 2]])
    assert torch.equal(scales[[0, 2]], untouched_scales[[0, 2]])
    assert not torch.equal(packed[1], untouched_packed[1])


def test_pages_stay_inside_their_pitch():
    """Nothing may be written between pages, in either direction.

    K and V share a destination page in the real cache, so a write that runs
    past its own region silently corrupts the other one.
    """
    pitch = 100 * 1024
    tokens = _tokens(512)
    starts = _as_int32([0, 128, 256, 384])
    destinations = _as_int32([0, 1, 2, 3])

    for indexed_quantize, make_destination, region_bytes in (
        (
            quantize_key_tokens_into,
            _key_destination,
            (PAGE_SIZE * HEADS * HEAD_DIM // 2, HEADS * SCALE_BYTES_PER_HEAD),
        ),
        (
            quantize_value_tokens_into,
            _value_destination,
            (
                HEADS * HEAD_DIM * PAGE_SIZE // 2,
                HEADS * SCALE_BYTES_PER_HEAD,
            ),
        ),
    ):
        (packed_buffer, packed), (scale_buffer, scales) = make_destination(
            4, pitch
        )
        indexed_quantize(tokens, packed, scales, starts, destinations)
        for buffer, used in zip(
            (packed_buffer, scale_buffer), region_bytes
        ):
            assert torch.all(buffer[:, used:] == 0xA5), (
                "quantization wrote past the end of its region"
            )
