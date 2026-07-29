"""Attention implementation for the NVFP4 KV cache."""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from .guards import NVFP4, check_supported
from .runtime import WriteRuntime
from .write import write_kv


class NVFP4Impl(FlashAttentionImpl):
    """Writes the NVFP4 cache; attention itself is still FlashAttention's.

    The write path is complete: full pages are quantized into vLLM blocks and
    the remainder goes to the BF16 tail. The read path is not, so a run with
    ``kv_cache_dtype="nvfp4"`` fills a correct cache but returns nothing
    meaningful from attention.
    """

    def __init__(self, *args, **kwargs) -> None:
        config = get_current_vllm_config()
        # Ahead of super(), so an unsupported engine configuration is reported
        # instead of whatever FlashAttention's own setup happens to fail on.
        check_supported(config)
        super().__init__(*args, **kwargs)

        cache_config = config.cache_config
        self.nvfp4 = (
            cache_config is not None and cache_config.cache_dtype == NVFP4
        )
        self.num_slots = config.scheduler_config.max_num_seqs
        self.runtime: WriteRuntime | None = None
        self.layer_index = 0

    def _bind_runtime(self, device: torch.device) -> None:
        """Create the model's shared write-path state once and hand it around.

        Called from the profile run, where every attention layer is already
        registered but the KV cache does not exist yet. Allocating the tail
        here is what makes vLLM subtract it from the cache budget.
        """
        modules = [
            module
            for module in get_forward_context().no_compile_layers.values()
            if isinstance(getattr(module, "impl", None), NVFP4Impl)
        ]
        if not any(module.impl is self for module in modules):
            raise ValueError(
                "an NVFP4 layer is running outside the forward context that "
                "lists it, so the write path cannot be shared with its peers"
            )
        shapes = {
            (module.impl.num_kv_heads, module.impl.head_size)
            for module in modules
        }
        if len(shapes) != 1:
            raise ValueError(
                f"NVFP4 needs one K/V shape across layers, found {shapes}"
            )
        num_kv_heads, head_dim = shapes.pop()

        runtime = WriteRuntime(
            num_layers=len(modules),
            num_slots=self.num_slots,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
        )
        # The index only has to be a stable bijection; dict order follows the
        # order the layers were constructed in, which makes it readable too.
        for index, module in enumerate(modules):
            module.impl.runtime = runtime
            module.impl.layer_index = index

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not self.nvfp4:
            super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
            return

        metadata, _, _, _ = get_attention_context(layer.layer_name)
        if metadata is None or self.runtime is None:
            return

        key_pages_fp4, key_scales, value_pages_fp4, value_scales = (
            self.runtime.views(kv_cache)
        )
        write_kv(
            key=key,
            value=value,
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            tail_key=self.runtime.tail_key[self.layer_index],
            tail_value=self.runtime.tail_value[self.layer_index],
            source_tokens=metadata.source_tokens,
            destination_pages=metadata.destination_pages,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            seqused_fp4=metadata.seqused_fp4,
            row_to_slot=metadata.row_to_slot,
            num_actual_tokens=metadata.num_actual_tokens,
        )

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
        if not self.nvfp4:
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
        if self.runtime is None:
            self._bind_runtime(query.device)
        # No read path yet: the cache now holds FP4 that FlashAttention cannot
        # read. Contributing nothing is wrong but bounded, whereas returning
        # the buffer untouched feeds whatever was in that memory into the
        # residual stream, and NaN there makes every later layer's K/V NaN.
        return output.zero_()
