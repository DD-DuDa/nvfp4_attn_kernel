"""Correctness tests for NVFP4 decode-query quantization."""

from __future__ import annotations

import pytest
import torch


PAGE_SIZE = 128
HEADS = 32
HEAD_DIM = 128


@pytest.fixture(scope="module")
def quantizers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")

    pytest.importorskip("flashinfer")
    pytest.importorskip("cutlass")

    from nvfp4_decode_kernel._quantize import quantize_query
    from nvfp4_decode_kernel.quantize_q_kernel import (
        quantize_decode_q_to_padded_fp4,
    )
    from nvfp4_decode_kernel._quantize_flashinfer import (
        flashinfer_quantize_query,
    )

    return (
        quantize_query,
        quantize_decode_q_to_padded_fp4,
        flashinfer_quantize_query,
    )


@pytest.mark.parametrize("rows", [1, 7, 8])
def test_quantize_query_matches_flashinfer(quantizers, rows):
    quantize_query, _, flashinfer_quantize_query = quantizers
    torch.manual_seed(0x5200 + rows)
    query = torch.randn(
        rows,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 3.0

    actual_fp4, actual_scales = quantize_query(query)
    expected_fp4, expected_scales = flashinfer_quantize_query(query)

    assert torch.equal(
        actual_fp4.view(torch.uint8),
        expected_fp4[:, :1].view(torch.uint8),
    )
    assert torch.equal(actual_scales, expected_scales)


def test_quantize_query_supports_strided_indexed_input(quantizers):
    quantize_query, _, flashinfer_quantize_query = quantizers
    torch.manual_seed(0x7500)
    qkv = torch.randn(
        10,
        HEADS,
        3,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query = qkv[:, :, 0, :]
    row_indices = torch.tensor(
        [9, 2, 8, 1],
        dtype=torch.int32,
        device="cuda",
    )

    actual_fp4, actual_scales = quantize_query(
        query,
        row_indices=row_indices,
    )
    expected_fp4, expected_scales = flashinfer_quantize_query(
        query,
        row_indices=row_indices,
    )

    assert torch.equal(
        actual_fp4.view(torch.uint8),
        expected_fp4[:, :1].view(torch.uint8),
    )
    assert torch.equal(actual_scales, expected_scales)


def test_quantize_query_writes_padded_bf16_query(quantizers):
    _, quantize_decode_q_to_padded_fp4, flashinfer_quantize_query = quantizers
    rows = 4
    torch.manual_seed(0x6300)
    query = torch.randn(
        rows,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_fp4, expected_scales = flashinfer_quantize_query(query)

    query_fp4 = torch.zeros(
        rows,
        1,
        HEADS,
        HEAD_DIM // 2,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float4_e2m1fn_x2)
    scales_storage = torch.zeros(
        rows,
        1,
        HEADS,
        HEAD_DIM // 64,
        32,
        4,
        4,
        dtype=torch.uint8,
        device="cuda",
    )
    query_scales = scales_storage.permute(
        0, 2, 1, 3, 4, 5, 6
    ).permute(
        4, 5, 2, 6, 3, 1, 0
    )
    query_padded = torch.zeros(
        rows,
        PAGE_SIZE,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    quantize_decode_q_to_padded_fp4(
        query,
        query_fp4,
        query_scales,
        query_padded,
    )

    assert torch.equal(
        query_fp4.view(torch.uint8),
        expected_fp4[:, :1].view(torch.uint8),
    )
    assert torch.equal(query_scales, expected_scales)
    assert torch.equal(query_padded[:, 0], query)
    assert torch.count_nonzero(query_padded[:, 1:]).item() == 0
