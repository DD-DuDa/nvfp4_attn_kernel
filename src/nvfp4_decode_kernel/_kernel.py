"""Composition layer between the public API and private kernel operations."""

from __future__ import annotations

import torch


def fp4_decode_impl(
    query: torch.Tensor,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    fp4_page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    *,
    residual_key_pages_bf16: torch.Tensor | None = None,
    residual_value_pages_bf16: torch.Tensor | None = None,
    residual_page_ids: torch.Tensor | None = None,
    seqused_residual: torch.Tensor | None = None,
    has_bf16: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    query_row_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize the current query and run the paged NVFP4 decode kernel."""
    from ._decode import decode_fp4
    from ._quantize import quantize_query

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    rows = (
        query.shape[0]
        if query_row_indices is None
        else query_row_indices.shape[0]
    )
    query_padded_bf16 = torch.zeros(
        rows,
        128,
        query.shape[1],
        query.shape[2],
        dtype=torch.bfloat16,
        device=query.device,
    )
    query_fp4, query_scales = quantize_query(
        query,
        row_indices=query_row_indices,
        query_padded_out=query_padded_bf16,
    )

    return decode_fp4(
        query_fp4=query_fp4,
        query_scales=query_scales,
        query_padded_bf16=query_padded_bf16,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        fp4_page_table=fp4_page_table,
        seqused_fp4=seqused_fp4,
        residual_key_pages_bf16=residual_key_pages_bf16,
        residual_value_pages_bf16=residual_value_pages_bf16,
        residual_page_ids=residual_page_ids,
        seqused_residual=seqused_residual,
        has_bf16=has_bf16,
        softmax_scale=softmax_scale,
        out=out,
        out_indices=out_indices,
    )
