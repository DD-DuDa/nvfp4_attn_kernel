"""FlashInfer NVFP4 query quantization."""

from __future__ import annotations

import torch
from flashinfer.fp4_quantization import fp4_quantize


PAGE_SIZE = 128
SF_VEC_SIZE = 16

_global_scales: dict[torch.device, torch.Tensor] = {}


def _global_scale(device: torch.device) -> torch.Tensor:
    scale = _global_scales.get(device)
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32, device=device)
        _global_scales[device] = scale
    return scale


def flashinfer_quantize_query(
    query: torch.Tensor,
    *,
    row_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 decode rows into the FA4 FP4 and scale layouts."""
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise ValueError("query must be a BF16 CUDA tensor")
    if query.ndim != 3 or query.stride(-1) != 1:
        raise ValueError("query must have shape [rows, heads, head_dim]")

    if row_indices is None:
        compact_query = query
    else:
        if (
            row_indices.dtype is not torch.int32
            or not row_indices.is_cuda
            or row_indices.ndim != 1
        ):
            raise ValueError("row_indices must be a 1-D INT32 CUDA tensor")
        compact_query = query.index_select(0, row_indices.long())

    rows, heads, head_dim = compact_query.shape
    if head_dim % 64 != 0:
        raise ValueError("query head_dim must be divisible by 64")

    padded = torch.zeros(
        rows,
        PAGE_SIZE,
        heads,
        head_dim,
        dtype=query.dtype,
        device=query.device,
    )
    padded[:, 0] = compact_query

    fp4_data, scale_data = fp4_quantize(
        padded.reshape(rows * PAGE_SIZE, heads * head_dim),
        _global_scale(query.device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
        is_sf_8x4_layout=False,
    )

    query_fp4 = (
        fp4_data.reshape(rows, PAGE_SIZE, heads, head_dim // 2)
        .view(torch.int8)
        .view(torch.float4_e2m1fn_x2)
    )

    rest_k = head_dim // 64
    scales = scale_data.reshape(
        rows, heads * rest_k, 32, 4, 4
    ).reshape(
        rows, 1, heads, rest_k, 32, 4, 4
    )
    query_scales = scales.permute(
        0, 2, 1, 3, 4, 5, 6
    ).permute(
        4, 5, 2, 6, 3, 1, 0
    )

    return query_fp4, query_scales
