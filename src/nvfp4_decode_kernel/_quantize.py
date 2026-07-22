"""NVFP4 query quantization."""

from __future__ import annotations

import torch

from .quantize_kv_kernel import quantize_key_pages, quantize_value_pages
from .quantize_q_kernel import quantize_decode_q_to_padded_fp4


PAGE_SIZE = 128


def quantize_query(
    query: torch.Tensor,
    *,
    row_indices: torch.Tensor | None = None,
    query_padded_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 decode rows with the CuTeDSL query kernel."""
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise ValueError("query must be a BF16 CUDA tensor")
    if query.ndim != 3 or query.stride(-1) != 1:
        raise ValueError("query must have shape [rows, heads, head_dim]")
    if row_indices is not None and (
        row_indices.dtype is not torch.int32
        or not row_indices.is_cuda
        or not row_indices.is_contiguous()
        or row_indices.ndim != 1
    ):
        raise ValueError(
            "row_indices must be contiguous 1-D INT32 CUDA"
        )

    _, heads, head_dim = query.shape
    rows = query.shape[0] if row_indices is None else row_indices.shape[0]
    if head_dim % 64 != 0:
        raise ValueError("query head_dim must be divisible by 64")

    query_fp4 = torch.zeros(
        rows,
        PAGE_SIZE,
        heads,
        head_dim // 2,
        dtype=torch.uint8,
        device=query.device,
    ).view(torch.float4_e2m1fn_x2)

    rest_k = head_dim // 64
    scales = torch.zeros(
        rows,
        1,
        heads,
        rest_k,
        32,
        4,
        4,
        dtype=torch.uint8,
        device=query.device,
    )
    query_scales = scales.permute(
        0, 2, 1, 3, 4, 5, 6
    ).permute(
        4, 5, 2, 6, 3, 1, 0
    )

    quantize_decode_q_to_padded_fp4(
        query,
        query_fp4,
        query_scales,
        query_padded_out,
        row_indices=row_indices,
    )

    return query_fp4, query_scales
