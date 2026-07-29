"""Per-step attention metadata for the NVFP4 KV cache.

Carries FlashAttention's own metadata unchanged and appends what the control
plane resolved for this step. Subclassing rather than wrapping keeps every
non-NVFP4 layer working: until the decode kernel is wired in, the layers still
run FlashAttention, and they read these objects through the base class's
fields.

The fields below are device tensors written by the control kernel. Reading any
of them on the host costs a synchronization, so nothing outside a kernel
should.
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

    decode_row_indices: torch.Tensor
    """``[num_slots]`` int32. Batch rows that emit exactly one token, packed to
    the front and padded with -1, which is what the decode kernel indexes
    by."""

    decode_count: torch.Tensor
    """``[1]`` int32. How many entries of ``decode_row_indices`` are real. On
    the device because the host does not need it and reading it would
    synchronize."""

    active_row_mask: torch.Tensor
    """``[num_slots]`` bool. The same information as ``decode_count`` in the
    shape a kernel can predicate on."""

    @classmethod
    def from_flash(
        cls, base: FlashAttentionMetadata, outputs: ControlOutputs
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
            decode_row_indices=outputs.decode_row_indices,
            decode_count=outputs.decode_count,
            active_row_mask=outputs.active_row_mask,
        )
