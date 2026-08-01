"""Attention implementation for the NVFP4 KV cache."""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from nvfp4_decode_kernel import fp4_decode

from .guards import NVFP4, check_supported, check_layer_supported
from .runtime import LayerRuntime
from .write import reset_new_request_tails, write_kv


class NVFP4Impl(FlashAttentionImpl):
    """Attention over the NVFP4 cache, split by what each row is doing.

    vLLM has sorted the one-token rows to the front of the batch (see
    ``builder.decode_split``), which splits the step cleanly in two. The decode
    prefix reads the FP4 pages and the BF16 tail through ``fp4_decode``. The
    prefill suffix reads neither: with chunked prefill and prefix caching both
    refused, a prompt arrives whole, so FlashAttention attends it against the
    K/V of this same forward pass and never touches the block table.

    Either side can be empty, and usually one is.
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
        self.num_slots = 0
        if self.nvfp4:
            check_layer_supported(self)
            # FlashAttention turns this on for any quantized cache, and the
            # attention layer then hands the impl an FP8 query. Both paths here
            # need BF16: prefill attends against this pass's own BF16 K/V, and
            # decode quantizes to FP4, not FP8.
            self.supports_quant_query_input = False
            self.num_slots = config.scheduler_config.max_num_seqs
        self.runtime: LayerRuntime | None = None
        self.layer_index = 0

    def _bind_runtime(self, device: torch.device) -> None:
        """Create the model's shared state once and hand it around.

        Called from the profile run, where every attention layer is already
        registered but the KV cache does not exist yet. Allocating here is what
        makes vLLM subtract this memory from the cache budget.
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
            (
                module.impl.num_heads,
                module.impl.num_kv_heads,
                module.impl.head_size,
            )
            for module in modules
        }
        if len(shapes) != 1:
            raise ValueError(
                f"NVFP4 needs one attention shape across layers, found {shapes}"
            )
        num_heads, num_kv_heads, head_dim = shapes.pop()

        runtime = LayerRuntime(
            num_layers=len(modules),
            num_slots=self.num_slots,
            num_heads=num_heads,
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
        # Both mean the profile run, where there is no cache to write into:
        # the builder returns no metadata, and the runtime is not bound until
        # the first forward. On a real step neither can be None.
        if metadata is None or self.runtime is None:
            return

        # The first layer speaks for the whole model here: a slot's previous
        # tenant has to be erased from every layer before any of them writes
        # this step, and layers run in the order they were indexed.
        if self.layer_index == 0:
            reset_new_request_tails(
                tail_key=self.runtime.tail_key,
                tail_value=self.runtime.tail_value,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                row_to_slot=metadata.row_to_slot,
            )

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
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported by the NVFP4 "
                "attention path"
            )
        if self.runtime is None:
            self._bind_runtime(query.device)
        if attn_metadata is None:
            # Profile run. The cache is a placeholder with no storage, so there
            # is nothing to read; zero rather than leave the buffer untouched,
            # since NaN in the residual stream propagates into every later
            # layer's K/V.
            return output.zero_()

        rows = attn_metadata.decode_prefix_rows
        tokens = attn_metadata.decode_prefix_tokens
        if rows:
            self._decode(rows, query, kv_cache, attn_metadata, output)
        if tokens < attn_metadata.num_actual_tokens:
            self._prefill(tokens, query, key, value, attn_metadata, output)
        return output

    def _decode(
        self,
        rows: int,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> None:
        """Attention for the one-token rows, over the FP4 pages and the tail.

        Every per-row argument is a prefix slice of this step's arrays, which
        is what the batch reordering buys: no compaction, no scatter, and an
        output that lands where vLLM already wants it.
        """
        key_pages_fp4, key_scales, value_pages_fp4, value_scales = (
            self.runtime.views(kv_cache)
        )
        pages = attn_metadata.decode_page_columns
        fp4_decode(
            query=query[:rows],
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=attn_metadata.block_table[:rows, :pages],
            seqused_fp4=attn_metadata.seqused_fp4[:rows],
            residual_key_pages_bf16=self.runtime.tail_key[self.layer_index],
            residual_value_pages_bf16=self.runtime.tail_value[
                self.layer_index
            ],
            residual_page_ids=attn_metadata.row_to_slot[:rows],
            seqused_residual=attn_metadata.seqused_residual[:rows],
            softmax_scale=self.scale,
            # The control plane produced these on the device this same step and
            # nothing since could have invalidated them. Checking would mean
            # reading them back, which costs a synchronization per layer.
            trusted_metadata=True,
            query_padded_scratch=self.runtime.query_padded,
            out=output[:rows],
        )

    def _prefill(
        self,
        tokens: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> None:
        """Attention for the rows that arrived with their whole prompt.

        Cache-free: ``k`` and ``v`` are this pass's own tensors and
        ``cu_seqlens_k`` is ``cu_seqlens_q``, so the FP4 pages this prompt is
        about to be written into are never read. That is what lets prefill run
        against a cache FlashAttention cannot decode.

        ``max_seqlen_q`` is the whole batch's maximum rather than the suffix's.
        The prefills are the long rows, so the two agree unless the batch is
        pure decode, in which case this does not run.
        """
        starts = attn_metadata.prefill_query_start_loc
        num_actual_tokens = attn_metadata.num_actual_tokens
        flash_attn_varlen_func(
            q=query[tokens:num_actual_tokens],
            k=key[tokens:num_actual_tokens],
            v=value[tokens:num_actual_tokens],
            out=output[tokens:num_actual_tokens],
            cu_seqlens_q=starts,
            max_seqlen_q=attn_metadata.max_query_len,
            cu_seqlens_k=starts,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            fa_version=self.vllm_flash_attn_version,
        )
