"""Framework-independent interface for paged NVFP4 decode."""

from __future__ import annotations

import torch


class KernelNotAvailableError(RuntimeError):
    """Raised when the kernel implementation has not been installed."""


def fp4_decode(
    query: torch.Tensor | None = None,
    key_pages_fp4: torch.Tensor | None = None,
    key_scales: torch.Tensor | None = None,
    value_pages_fp4: torch.Tensor | None = None,
    value_scales: torch.Tensor | None = None,
    fp4_page_table: torch.Tensor | None = None,
    seqused_fp4: torch.Tensor | None = None,
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
) -> torch.Tensor:
    """Run paged NVFP4 decode attention.

    Args:
        query: Optional BF16 query in ``[rows, heads_q, head_dim]`` layout. It may
            contain more rows than the current decode batch when
            ``query_row_indices`` is provided. Supply either this argument or
            both ``query_fp4`` and ``query_scales``.
        query_fp4: Optional pre-quantized packed E2M1 query from
            ``quantize_query``.
        query_scales: Optional E4M3 scale-factor bytes paired with
            ``query_fp4``.
        key_pages_fp4: Packed E2M1 K pages.
        key_scales: E4M3 scale factors for ``key_pages_fp4``.
        value_pages_fp4: Packed E2M1 V pages.
        value_scales: E4M3 scale factors for ``value_pages_fp4``.
        fp4_page_table: INT32 logical-to-physical FP4 page mapping, shaped
            ``[rows, max_pages_per_row]``.
        seqused_fp4: INT32 FP4 token count for each row, shaped ``[rows]``.
            Every value must be a multiple of the page size.
        residual_key_pages_bf16: Optional physical BF16 K-page cache.
        residual_value_pages_bf16: Optional physical BF16 V-page cache.
        residual_page_ids: Optional INT32 physical BF16 page ID per row.
        seqused_residual: Optional INT32 valid BF16 token count per row.
        has_bf16: Optional BOOL indicator of whether each row owns an active
            BF16 residual page. It must agree with ``seqused_residual > 0``.
        softmax_scale: Score scale. The implementation uses
            ``head_dim**-0.5`` when omitted.
        query_row_indices: Optional INT32 mapping from compact decode rows to
            rows in a full 3-D query tensor.
        out: Optional full BF16 output tensor in ``[rows, heads_q, head_dim]``
            layout. It must be supplied together with ``out_indices``.
        out_indices: Optional INT32 mapping from compact decode rows to rows
            in ``out``. When supplied, the kernel scatters directly into
            ``out``.

    Returns:
        The supplied ``out`` tensor when direct scatter is enabled; otherwise
        a compact BF16 output corresponding to the selected query rows.
    """
    try:
        from ._kernel import fp4_decode_impl
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}._kernel":
            raise
        raise KernelNotAvailableError(
            "NVFP4 kernel implementation is unavailable."
        ) from error

    required = {
        "key_pages_fp4": key_pages_fp4,
        "key_scales": key_scales,
        "value_pages_fp4": value_pages_fp4,
        "value_scales": value_scales,
        "fp4_page_table": fp4_page_table,
        "seqused_fp4": seqused_fp4,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"missing required decode arguments: {', '.join(missing)}"
        )

    return fp4_decode_impl(
        query=query,
        query_fp4=query_fp4,
        query_scales=query_scales,
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
        query_row_indices=query_row_indices,
        out=out,
        out_indices=out_indices,
        trusted_metadata=trusted_metadata,
    )
