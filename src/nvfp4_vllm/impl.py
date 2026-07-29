"""Attention implementation for the NVFP4 KV cache."""

from __future__ import annotations

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


def _is_nvfp4_cache(kv_cache: torch.Tensor) -> bool:
    """True for the flat ``[num_blocks, page_bytes]`` NVFP4 page view.

    FlashAttention's cache is four-dimensional, so rank tells the two apart. A
    zero-sized tensor is the placeholder passed before the cache exists.
    """
    return kv_cache.numel() > 0 and kv_cache.ndim == 2


class NVFP4Impl(FlashAttentionImpl):
    """FlashAttention, except that the NVFP4 cache is not touched yet.

    Neither the read nor the write path for the packed cache exists, so a run
    with ``kv_cache_dtype="nvfp4"`` starts and allocates but cannot produce
    meaningful output.
    """

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if _is_nvfp4_cache(kv_cache):
            return
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _is_nvfp4_cache(kv_cache):
            return output
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
