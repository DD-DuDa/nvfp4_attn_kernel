"""Per-step attention metadata for the NVFP4 KV cache.

Carries FlashAttention's own metadata unchanged and appends what the control
plane resolved for this step. Subclassing rather than wrapping keeps every
non-NVFP4 layer working: until the decode kernel is wired in, the layers still
run FlashAttention, and they read these objects through the base class's
fields.

Most of the fields below are device tensors written by the control kernel.
Reading any of those on the host costs a synchronization, so nothing outside a
kernel should. The two plain ints are the exception, and are marked as such:
they come from CPU-side batch descriptions, never from the device.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from .control import ControlOutputs


# kw_only because the base class ends in defaulted fields, which would
# otherwise forbid adding required ones.
@dataclass(kw_only=True)
class NVFP4Metadata(FlashAttentionMetadata):
    """FlashAttention metadata plus this step's tail-slot assignment."""

    row_to_slot: torch.Tensor
    """``[num_reqs]`` int32. Tail slot owning each batch row, -1 if the row is
    padding."""

    token_to_slot: torch.Tensor
    """``[num_actual_tokens]`` int32. Tail slot each token belongs to, so the
    KV write can find its destination without a per-row loop."""

    seqused_fp4: torch.Tensor
    """``[num_reqs]`` int32. Tokens of each sequence that live in FP4 pages,
    always a whole number of pages."""

    seqused_residual: torch.Tensor
    """``[num_reqs]`` int32. Tokens of each sequence that live in the BF16
    tail. Together with ``seqused_fp4`` this sums to the sequence length."""

    promotion_mask: torch.Tensor
    """``[num_reqs]`` bool. Rows whose tail filled a whole page this step and
    must be quantized into the FP4 cache before the next one."""

    decode_prefix_rows: int
    """How many rows of this batch emit exactly one token. vLLM has already
    moved them to the front, so they are rows ``[0, decode_prefix_rows)``. A
    host int, not a device tensor: it comes from ``query_start_loc_cpu``, so
    reading it is free.

    Not the base class's ``num_decode_reqs``, which counts something else
    (rows that carry decode context under decode context parallelism) and is
    left at zero outside that feature."""

    decode_prefix_tokens: int
    """Where the prefill tokens begin. Equal to ``decode_prefix_rows`` while
    every decode row emits one token, but derived rather than assumed."""

    prefill_query_start_loc: torch.Tensor
    """``[num_prefills + 1]`` int32. ``query_start_loc`` rebased on the first
    prefill token, which is what FlashAttention needs once the decode prefix
    has been sliced off. Built here rather than per layer because every layer
    of the step would otherwise recompute the same subtraction on the
    device."""

    decode_page_columns: int
    """How many page-table columns the decode kernel has to walk. The kernel
    reads the table's width as the longest row it could see and picks a split
    count from it, so a table left at the model's capacity makes a short batch
    look long and buys splits that own no page.

    The batch's longest row, not the decodes' own: the per-row lengths live on
    the device, and vLLM's host-side copy is materialized by a transfer that
    would cost the step a synchronization. A prefill sharing the step can only
    widen this, never narrow it wrongly. A row's last page is always its
    tail's, never an FP4 one, which is what puts the boundary one token short
    of the length."""

    source_tokens: torch.Tensor
    """``[work]`` int32. First token of each full page this step must quantize,
    as an index into the step's flattened K/V."""

    destination_pages: torch.Tensor
    """``[work]`` int32. Block each of those pages belongs in, -1 where the
    grid slot has no work. Paired with ``source_tokens`` and shared by every
    layer, since which pages exist does not depend on the layer."""

    @classmethod
    def from_flash(
        cls,
        base: FlashAttentionMetadata,
        outputs: ControlOutputs,
        *,
        source_tokens: torch.Tensor,
        destination_pages: torch.Tensor,
        decode_prefix_rows: int,
        decode_prefix_tokens: int,
        prefill_query_start_loc: torch.Tensor,
        decode_page_columns: int,
    ) -> "NVFP4Metadata":
        """Carry every FlashAttention field across and append the slot fields.

        Copied field by field rather than mutated in place: the base object is
        what FlashAttention's own builder returned, and a metadata object is
        read by every layer in the step.
        """
        return cls(
            **{field.name: getattr(base, field.name) for field in fields(base)},
            row_to_slot=outputs.row_to_slot,
            token_to_slot=outputs.token_to_slot,
            seqused_fp4=outputs.seqused_fp4,
            seqused_residual=outputs.seqused_residual,
            promotion_mask=outputs.promotion_mask,
            decode_prefix_rows=decode_prefix_rows,
            decode_prefix_tokens=decode_prefix_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            decode_page_columns=decode_page_columns,
            source_tokens=source_tokens,
            destination_pages=destination_pages,
        )
