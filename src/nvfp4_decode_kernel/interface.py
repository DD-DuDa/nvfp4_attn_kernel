"""Framework-independent interface for paged NVFP4 decode."""

from __future__ import annotations

import torch


# Rows of the tile the residual MMA reads the BF16 query through. Only row 0 of
# each tile holds a real query, since decode has one token per row.
RESIDUAL_ROW_TILE = 128


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
    num_splits: int = 1,
    query_padded_scratch: torch.Tensor | None = None,
    query_fp4_scratch: torch.Tensor | None = None,
    query_scales_scratch: torch.Tensor | None = None,
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
        out: Optional BF16 output tensor in ``[rows, heads_q, head_dim]``
            layout, written in place instead of a fresh allocation. Row i of
            the batch lands in row i unless ``out_indices`` says otherwise.
        out_indices: Optional INT32 mapping from compact decode rows to rows
            in ``out``. Costs split-K, which cannot reorder rows on its way
            out, so pass it only when the rows really do have to move.
        num_splits: How many key partitions the decode is split across, as a
            positive power of two. One, the default, is the production
            answer: split-K measures as a consistent 13-23% overhead on this
            kernel, and under CUDA graph capture the page table is frozen at
            the model's maximum length, so any count derived from it would be
            an inflated one baked into every later replay, along with the
            zero-filled FP32 partials and LSE the split path allocates. The
            parameter exists so the split implementation stays reachable
            through this entry from tests.

            A value above one is honoured only where the split path can serve
            it: no residual argument at all, or a complete residual on the
            BF16 query path, and never together with ``out_indices``, whose
            reordering the combine cannot perform. Asking for a split
            anywhere else raises, because a request quietly downgraded to one
            tile would look exactly like a split that ran.
        query_padded_scratch: Optional BF16 buffer shaped ``[at least rows,
            RESIDUAL_ROW_TILE, heads_q, head_dim]`` that holds the query padded
            to the residual MMA's row tile. Only the first row of each tile is
            written, so a caller-owned buffer has to be zeroed once and never
            again. Supplying one avoids reallocating and rezeroing it on every
            call, which matters when a residual is present on every layer of
            every step.
        query_fp4_scratch: Optional UINT8 buffer shaped ``[at least rows, 1,
            heads_q, head_dim // 2]`` that receives the packed E2M1 query. The
            quantizer rewrites every byte of it, so whatever it arrives
            holding is irrelevant.
        query_scales_scratch: Optional UINT8 buffer shaped ``[at least rows,
            1, heads_q, head_dim // 64, 32, 4, 4]`` that receives the E4M3
            query scale factors. Unlike ``query_fp4_scratch`` this one carries
            the same obligation as ``query_padded_scratch``, and for a
            sharper reason: the layout reserves ``32 * 4`` scale slots per KV
            head and only the ``heads_q // heads_kv`` of them that carry a
            query head are ever written. Nothing clears the others, so a
            buffer zeroed once is byte-for-byte what an internal allocation
            would have produced, while one handed over uninitialized feeds the
            MMA whatever it found for as long as it is reused. No shape check
            can catch that.

            For the same reason a buffer belongs to one ``(heads_q, heads_kv,
            head_dim)`` triple for its whole life. Its shape says nothing
            about ``heads_kv``, so the shape check cannot notice the reuse,
            but ``heads_q // heads_kv`` is exactly what picks the slots that
            get written: at ``heads_kv == 1`` query head ``h`` lands in slot
            ``h`` of the single KV head's tile, and at ``heads_kv == 8`` in
            slot ``h % (heads_q // 8)`` of tile ``h // (heads_q // 8)``.
            Handing the first buffer to the second geometry leaves the slots
            only the first one wrote still holding its scales, where the
            second needs zeros.

    Returns:
        A compact BF16 output for the selected query rows. It is a view of
        ``out`` when one was supplied rather than a fresh allocation: the
        whole of ``out`` when ``out_indices`` scattered the rows across it,
        and ``out[:rows]`` otherwise, because the rows past the batch belong
        to whoever else shares the buffer.
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
        num_splits=num_splits,
        query_padded_scratch=query_padded_scratch,
        query_fp4_scratch=query_fp4_scratch,
        query_scales_scratch=query_scales_scratch,
    )
