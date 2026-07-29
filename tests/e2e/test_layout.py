"""How a vLLM block is interpreted as an NVFP4 page.

No engine here: this is the seam between what the backend promises vLLM about
page size and what the quantizer and decode kernel actually read and write. If
those three disagree the failure downstream is silent corruption, so they are
checked directly and byte for byte.
"""

from __future__ import annotations

import pytest
import torch

from nvfp4_decode_kernel import fp4_decode
from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_pages,
    quantize_key_tokens_into,
    quantize_value_pages,
    quantize_value_tokens_into,
)
from nvfp4_vllm.backend import nvfp4_page_size_bytes
from nvfp4_vllm.layout import PAGE_SIZE, carve, page_bytes

HEAD_DIM = 128
HEADS_KV = 8
HEADS_Q = 32


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")
    torch.manual_seed(0)


@pytest.mark.parametrize("heads_kv", [1, 8, 16])
def test_the_probed_page_matches_the_page_promised_to_vllm(heads_kv):
    """vLLM budgets memory from the backend's formula and allocates that much.

    The quantizer's real output has to fit in exactly that, with nothing left
    over: a page too small corrupts the next block, a page too large wastes
    cache silently.
    """
    assert page_bytes(heads_kv, HEAD_DIM, "cuda") == nvfp4_page_size_bytes(
        PAGE_SIZE, heads_kv, HEAD_DIM
    )


def test_writing_through_carve_reproduces_the_packed_quantizer_bytes():
    """The four regions of a block must hold what a dense quantizer would.

    Sources are unaligned and out of order and destinations are scattered
    blocks, which is what a prefill batch looks like; the bytes still have to
    match the reference page for page.
    """
    blocks = 6
    tokens = (
        torch.randn(
            2048, HEADS_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        * 0.3
    )
    starts = torch.tensor(
        [0, 37, 1000, 511], dtype=torch.int32, device="cuda"
    )
    destinations = torch.tensor(
        [5, 0, 3, 1], dtype=torch.int32, device="cuda"
    )

    cache = torch.zeros(
        blocks,
        page_bytes(HEADS_KV, HEAD_DIM, "cuda"),
        dtype=torch.uint8,
        device="cuda",
    )
    key_fp4, key_sf, value_fp4, value_sf = carve(cache, HEADS_KV, HEAD_DIM)
    quantize_key_tokens_into(tokens, key_fp4, key_sf, starts, destinations)
    quantize_value_tokens_into(
        tokens, value_fp4, value_sf, starts, destinations
    )

    windows = torch.stack(
        [tokens[start : start + PAGE_SIZE] for start in starts.tolist()]
    )
    expected_key_fp4, expected_key_sf = quantize_key_pages(windows)
    expected_value_fp4, expected_value_sf = quantize_value_pages(windows)

    written = destinations.long()
    assert torch.equal(key_fp4[written], expected_key_fp4)
    assert torch.equal(key_sf[written], expected_key_sf)
    assert torch.equal(value_fp4[written], expected_value_fp4)
    assert torch.equal(value_sf[written], expected_value_sf)

    untouched = [block for block in range(blocks) if block not in (5, 0, 3, 1)]
    assert torch.all(cache[untouched] == 0), (
        "quantization touched a block no index pointed at"
    )


def test_decode_reads_back_what_carve_wrote():
    """Closes the loop: the same views the writer filled are what decode reads.

    Comparing against decode over a densely packed cache isolates addressing
    from arithmetic, so the two must agree bit for bit rather than closely.
    """
    blocks = 4
    pages = (
        torch.randn(
            blocks,
            PAGE_SIZE,
            HEADS_KV,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.3
    )
    query = (
        torch.randn(1, HEADS_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        * 0.3
    )
    page_table = torch.arange(blocks, dtype=torch.int32, device="cuda").reshape(
        1, blocks
    )
    seqused_fp4 = torch.tensor(
        [(blocks - 1) * PAGE_SIZE], dtype=torch.int32, device="cuda"
    )
    seqused_residual = torch.tensor(
        [PAGE_SIZE], dtype=torch.int32, device="cuda"
    )
    residual_page_ids = torch.tensor(
        [blocks - 1], dtype=torch.int32, device="cuda"
    )

    cache = torch.zeros(
        blocks,
        page_bytes(HEADS_KV, HEAD_DIM, "cuda"),
        dtype=torch.uint8,
        device="cuda",
    )
    carved = carve(cache, HEADS_KV, HEAD_DIM)
    starts = torch.arange(
        0, blocks * PAGE_SIZE, PAGE_SIZE, dtype=torch.int32, device="cuda"
    )
    destinations = torch.arange(blocks, dtype=torch.int32, device="cuda")
    flat = pages.reshape(-1, HEADS_KV, HEAD_DIM)
    quantize_key_tokens_into(flat, carved[0], carved[1], starts, destinations)
    quantize_value_tokens_into(flat, carved[2], carved[3], starts, destinations)

    dense = (*quantize_key_pages(pages), *quantize_value_pages(pages))

    def decode(key_fp4, key_sf, value_fp4, value_sf):
        return fp4_decode(
            query=query,
            key_pages_fp4=key_fp4,
            key_scales=key_sf,
            value_pages_fp4=value_fp4,
            value_scales=value_sf,
            fp4_page_table=page_table,
            seqused_fp4=seqused_fp4,
            residual_key_pages_bf16=pages,
            residual_value_pages_bf16=pages,
            residual_page_ids=residual_page_ids,
            seqused_residual=seqused_residual,
            has_bf16=torch.ones(1, dtype=torch.bool, device="cuda"),
            softmax_scale=HEAD_DIM**-0.5,
        )

    assert torch.equal(
        decode(carved[0], carved[1], carved[2], carved[3]),
        decode(dense[0], dense[1], dense[2], dense[3]),
    )
