"""Indexed K/V page quantization.

The paged variant reads 128 tokens starting anywhere in a flat activation
buffer and writes them to any page of a destination whose pages may be spread
apart. Every case here is checked byte for byte against the densely packed
quantizer given the same tokens, because a serving cache and its reference
decode must agree exactly, not approximately.

A base table repeats that over destinations that are separate allocations —
one per model layer, which is how vLLM allocates a KV cache. The layers share
the indices and the page layout and nothing else, so what the last test here
is really pinning is that each layer's bytes went to the address the table
gave and to no other.
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
def test_a_base_table_sends_each_layer_to_its_own_allocation(direction):
    """One launch, one set of indices, a separate destination per layer.

    The destinations are allocated independently and handed to layers in an
    order their addresses do not predict, so a kernel that reached layer ``l``
    by striding off a single base would land on the wrong one. Each layer also
    reads its own slice of the source, which is what makes a swap visible:
    every layer would then hold bytes belonging to another.
    """
    dense_quantize, indexed_quantize, make_destination = direction
    layers = 4
    tokens_per_layer = 512
    starts = [0, 37, 128, 256, 383]
    destinations = [2, -1, 0, 3, 1]
    live = [(start, page) for start, page in zip(starts, destinations) if page >= 0]
    tokens = _tokens(layers * tokens_per_layer)

    pools = [make_destination(len(starts)) for _ in range(layers)]
    scrambled = [pools[index] for index in (3, 0, 2, 1)]
    table = torch.tensor(
        [
            [packed.data_ptr() for (packed, _), _ in scrambled],
            [scales.data_ptr() for _, (scales, _) in scrambled],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    (_, first_packed), (_, first_scales) = scrambled[0]
    indexed_quantize(
        tokens,
        first_packed,
        first_scales,
        _as_int32(starts),
        _as_int32(destinations),
        table,
    )

    for layer, ((packed_buffer, packed), (scale_buffer, scales)) in enumerate(
        scrambled
    ):
        base = layer * tokens_per_layer
        windows = torch.stack(
            [tokens[base + start :][:PAGE_SIZE] for start, _ in live]
        ).contiguous()
        expected_packed, expected_scales = dense_quantize(windows)
        where = [page for _, page in live]

        assert torch.equal(packed[where], expected_packed), (
            f"layer {layer}: packed bytes differ"
        )
        assert torch.equal(
            scales.view(torch.float8_e4m3fn)[where], expected_scales
        ), f"layer {layer}: scale bytes differ"

        idle = [page for page in range(len(starts)) if page not in where]
        for buffer, used in (
            (packed_buffer, expected_packed[0].numel()),
            (scale_buffer, expected_scales[0].numel()),
        ):
            assert torch.all(buffer[idle] == 0xA5), (
                f"layer {layer}: a page with no work was written"
            )
            assert torch.all(buffer[:, used:] == 0xA5), (
                f"layer {layer}: quantization wrote past the end of its region"
            )


def test_a_page_more_than_two_gibibytes_in_is_still_the_right_page():
    """A block index times a block pitch has to be computed in 64 bits.

    In 32 it wraps at 14563 blocks, which is 2 GiB — a served cache is sized
    in the hundreds of those, and every block past the first 2 GiB would be
    written somewhere else. The pitch here is a real NVFP4 block, and the page
    is the first one whose byte offset does not fit, so this fails the moment
    any index in the destination narrows again.
    """
    pitch = 100 * 1024
    page = 2**31 // pitch + 1
    blocks = page + 1
    needed = blocks * pitch
    free, _ = torch.cuda.mem_get_info()
    if needed > free // 2:
        pytest.skip(f"{needed >> 30} GiB of destination does not fit")

    cache = torch.zeros(blocks, pitch, dtype=torch.uint8, device="cuda")
    packed = cache.as_strided(
        (blocks, PAGE_SIZE, HEADS, HEAD_DIM // 2),
        (pitch, HEADS * HEAD_DIM // 2, HEAD_DIM // 2, 1),
        0,
    )
    scales = cache.as_strided(
        (blocks, 32, 4, 1, 4, 2, HEADS),
        (pitch, 16, 4, 1024, 1, 512, 1024),
        PAGE_SIZE * HEADS * HEAD_DIM // 2,
    )

    tokens = _tokens(PAGE_SIZE)
    quantize_key_tokens_into(
        tokens, packed, scales, _as_int32([0]), _as_int32([page])
    )
    expected_packed, expected_scales = quantize_key_pages(tokens.unsqueeze(0))

    assert torch.equal(packed[page], expected_packed[0])
    assert torch.equal(
        scales[page].view(torch.float8_e4m3fn), expected_scales[0]
    )


def test_a_source_that_does_not_divide_into_layers_is_refused():
    """The per-layer source stride comes from the buffer, so it must divide.

    Rounding it down would shift every layer but the first by a few tokens —
    a plausible-looking cache that is quietly one request's worth of history
    out of step.
    """
    tokens = _tokens(3 * 512 + 1)
    (_, packed), (_, scales) = _key_destination(2)
    table = torch.tensor(
        [[packed.data_ptr()] * 3, [scales.data_ptr()] * 3],
        dtype=torch.int64,
        device="cuda",
    )
    with pytest.raises(ValueError, match="does not divide"):
        quantize_key_tokens_into(
            tokens, packed, scales, _as_int32([0]), _as_int32([0]), table
        )


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
