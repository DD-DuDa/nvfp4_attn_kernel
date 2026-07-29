"""Attention backend that declares the NVFP4 KV cache page layout to vLLM."""

from __future__ import annotations

from typing import ClassVar

from vllm.config import get_current_vllm_config_or_none
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


PAGE_SIZE = 128
NVFP4 = "nvfp4"


def _cache_dtype() -> str | None:
    """The configured ``kv_cache_dtype``, or None outside an engine."""
    config = get_current_vllm_config_or_none()
    if config is None or config.cache_config is None:
        return None
    return config.cache_config.cache_dtype


def nvfp4_page_size_bytes(
    block_size: int, num_kv_heads: int, head_size: int
) -> int:
    """Bytes one K+V page occupies for a single layer.

    E2M1 data packs two values per byte and each group of 16 values carries one
    E4M3 scale byte. vLLM budgets the cache from the same expression in
    ``AttentionSpec.real_page_size_bytes``, so the two must agree exactly or the
    allocation will not match the shape.
    """
    packed_head_dim = head_size // 2 + head_size // 16
    return 2 * block_size * num_kv_heads * packed_head_dim


class NVFP4Backend(FlashAttentionBackend):
    """Paged NVFP4 KV cache backend.

    Inherits FlashAttention's impl and builder for now; only the KV cache
    declaration is NVFP4-specific.
    """

    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        *FlashAttentionBackend.supported_kv_cache_dtypes,
        NVFP4,
    ]

    @staticmethod
    def get_name() -> str:
        # vLLM resolves the enum via ``AttentionBackendEnum[get_name()]``.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import NVFP4Impl

        return NVFP4Impl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # V is quantized along the token axis a full page at a time, so a page
        # cannot be split or partially filled.
        return [PAGE_SIZE]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str != NVFP4:
            return FlashAttentionBackend.get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str=cache_dtype_str,
            )
        # A flat per-page byte view, committing to nothing about how the four
        # regions (K data, K scales, V data, V scales) sit inside it.
        page_bytes = nvfp4_page_size_bytes(block_size, num_kv_heads, head_size)
        return (num_blocks, page_bytes)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # The permutation must have the same rank as the shape above, and vLLM
        # does not pass the cache dtype here. Declining makes the caller fall
        # back to the identity order, which is what a flat byte page needs.
        if _cache_dtype() == NVFP4:
            raise NotImplementedError
        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )
