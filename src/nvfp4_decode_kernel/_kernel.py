"""Composition layer between the public API and private kernel operations."""

from __future__ import annotations

import torch


# Rows of the tile the residual MMA reads the BF16 query through. Only row 0 of
# each tile holds a real query, since decode has one token per row.
_RESIDUAL_ROW_TILE = 128


def _padded_query_buffer(
    query: torch.Tensor,
    rows: int,
    scratch: torch.Tensor | None,
) -> torch.Tensor:
    """The BF16 query padded to the residual MMA's row tile.

    The quantizer writes only ``[row, 0, head, :]``, so the rest of each tile
    has to be zero. A freshly allocated buffer gets that from ``zeros``; a
    caller-owned one has to have been zeroed when it was created, and stays
    valid forever because nothing ever writes those positions.
    """
    heads, head_dim = query.shape[1], query.shape[2]
    if scratch is None:
        return torch.zeros(
            rows,
            _RESIDUAL_ROW_TILE,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device=query.device,
        )
    expected = (_RESIDUAL_ROW_TILE, heads, head_dim)
    if (
        scratch.dtype is not torch.bfloat16
        or scratch.device != query.device
        or scratch.ndim != 4
        or scratch.shape[0] < rows
        or tuple(scratch.shape[1:]) != expected
        or not scratch.is_contiguous()
    ):
        raise ValueError(
            "query_padded_scratch must be a contiguous BF16 tensor on the "
            f"query's device shaped [at least {rows}, "
            f"{', '.join(str(extent) for extent in expected)}]"
        )
    return scratch[:rows]


def fp4_decode_impl(
    query: torch.Tensor | None,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    fp4_page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    *,
    query_fp4: torch.Tensor | None = None,
    query_scales: torch.Tensor | None = None,
    residual_key_pages_bf16: torch.Tensor | None = None,
    residual_value_pages_bf16: torch.Tensor | None = None,
    residual_page_ids: torch.Tensor | None = None,
    seqused_residual: torch.Tensor | None = None,
    has_bf16: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    query_row_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    trusted_metadata: bool = False,
    query_padded_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare either query contract and run the shared decode core."""
    from ._decode import decode_fp4, decode_fp4_split, split_k_heuristic
    from ._quantize import quantize_query

    has_bf16_query = query is not None
    has_fp4_query = query_fp4 is not None or query_scales is not None
    if has_bf16_query == has_fp4_query:
        raise ValueError(
            "supply either BF16 query or query_fp4 + query_scales"
        )

    if has_bf16_query:
        assert query is not None
        if query_fp4 is not None or query_scales is not None:
            raise ValueError(
                "BF16 query is mutually exclusive with query_fp4/query_scales"
            )
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5
        rows = (
            query.shape[0]
            if query_row_indices is None
            else query_row_indices.shape[0]
        )
        query_padded_bf16 = (
            _padded_query_buffer(query, rows, query_padded_scratch)
            if residual_key_pages_bf16 is not None
            else None
        )
        query_fp4, query_scales = quantize_query(
            query,
            row_indices=query_row_indices,
            query_padded_out=query_padded_bf16,
            heads_kv=key_pages_fp4.shape[2],
        )
    else:
        if query_fp4 is None or query_scales is None:
            raise ValueError(
                "query_fp4 and query_scales must be provided together"
            )
        if query_row_indices is not None:
            raise ValueError(
                "query_row_indices applies only to the BF16 query path"
            )
        if query_padded_scratch is not None:
            raise ValueError(
                "query_padded_scratch applies only to the BF16 query path"
            )
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        query_padded_bf16 = None

    # A residual reaches split-K only when every one of its arguments is
    # present. Keying off a single argument would let a partially supplied
    # residual be dropped there instead of raising the all-or-none error that
    # the non-split path enforces.
    residual_arguments = (
        residual_key_pages_bf16,
        residual_value_pages_bf16,
        residual_page_ids,
        seqused_residual,
    )
    complete_residual = all(
        argument is not None for argument in residual_arguments
    )
    splittable = (
        all(argument is None for argument in residual_arguments)
        and has_bf16 is None
    ) or (complete_residual and query_padded_bf16 is not None)
    direct_scatter = out is not None or out_indices is not None
    num_splits = 1
    if splittable and not direct_scatter:
        device = query_fp4.device
        num_splits = split_k_heuristic(
            query_fp4.shape[0],
            key_pages_fp4.shape[2],
            fp4_page_table.shape[1],
            sms=torch.cuda.get_device_properties(device).multi_processor_count,
        )
    if num_splits > 1:
        # Keep the single public entry's full contract checks when dispatching
        # to the split implementation. Trusted metadata skips only the
        # intentional device-value scans, not host shape/dtype validation.
        decode_fp4(
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
            trusted_metadata=trusted_metadata,
            validate_only=True,
        )
        return decode_fp4_split(
            query_fp4=query_fp4,
            query_scales=query_scales,
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=fp4_page_table,
            seqused_fp4=seqused_fp4,
            softmax_scale=softmax_scale,
            num_splits=num_splits,
            query_padded_bf16=query_padded_bf16,
            residual_key_pages_bf16=residual_key_pages_bf16,
            residual_value_pages_bf16=residual_value_pages_bf16,
            residual_page_ids=residual_page_ids,
            seqused_residual=seqused_residual,
            has_bf16=has_bf16,
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
        trusted_metadata=trusted_metadata,
    )
