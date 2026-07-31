"""Engine configuration the NVFP4 KV cache path refuses to run under.

These checks cannot live in ``AttentionBackend.validate_configuration``:
``_cached_get_attn_backend`` memoizes on ``AttentionSelectorConfig``, which
carries none of the fields below, so a second engine in the same process would
reuse the first one's verdict and never be validated. ``NVFP4Impl.__init__``
runs once per layer per engine and is not cached.
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backend import AttentionImpl, AttentionType

from .control import MAX_SUPPORTED_SLOTS


NVFP4 = "nvfp4"

# vLLM runs at most ``max_num_seqs`` requests concurrently, and each running
# request needs one BF16 tail slot. The slot table itself is sized from
# ``max_num_seqs``; this is the width v1 is validated at.
MAX_SLOTS = MAX_SUPPORTED_SLOTS


class UnsupportedConfigError(ValueError):
    """An engine configuration the NVFP4 path cannot serve correctly."""


def check_supported(vllm_config: VllmConfig) -> None:
    """Raise unless the engine is configured for the NVFP4 KV cache path.

    A no-op unless ``kv_cache_dtype`` is ``nvfp4``. With any other cache dtype
    the backend is a plain FlashAttention pass-through: none of the machinery
    these constraints protect is in play.
    """
    cache_config = vllm_config.cache_config
    if cache_config is None or cache_config.cache_dtype != NVFP4:
        return

    scheduler_config = vllm_config.scheduler_config
    if scheduler_config is not None and scheduler_config.max_num_seqs > MAX_SLOTS:
        raise UnsupportedConfigError(
            f"max_num_seqs={scheduler_config.max_num_seqs} exceeds the "
            f"{MAX_SLOTS} BF16 tail slots the NVFP4 path preallocates. Pass "
            f"max_num_seqs<={MAX_SLOTS}, or use a BF16 KV cache."
        )

    if cache_config.enable_prefix_caching:
        raise UnsupportedConfigError(
            "prefix caching is not supported by the NVFP4 KV cache: a cache "
            "hit resumes a sequence mid-page, and prefill over an FP4 prefix "
            "is not implemented. Pass enable_prefix_caching=False."
        )

    # Everything below keeps one invariant: a step either runs a whole prompt
    # from nothing, or extends a sequence by exactly one token. The control
    # plane reports a violation through ``error_code``, but only after the
    # tokens have gone somewhere wrong, so the configuration is refused first.
    if scheduler_config is not None and scheduler_config.enable_chunked_prefill:
        raise UnsupportedConfigError(
            "chunked prefill is not supported by the NVFP4 KV cache: the "
            "second chunk of a split prompt resumes mid-page, which needs "
            "prefill over an FP4 prefix. Pass enable_chunked_prefill=False; "
            "vLLM then requires max_num_batched_tokens >= max_model_len so "
            "that no prompt has to be split."
        )

    if scheduler_config is not None and scheduler_config.long_prefill_token_threshold:
        raise UnsupportedConfigError(
            "long_prefill_token_threshold="
            f"{scheduler_config.long_prefill_token_threshold} splits long "
            "prompts even with chunked prefill disabled. Leave it at 0."
        )

    if vllm_config.speculative_config is not None:
        raise UnsupportedConfigError(
            "speculative decoding is not supported by the NVFP4 KV cache: it "
            "verifies several draft tokens per step, and the decode kernel "
            "attends one query token per sequence."
        )

    if cache_config.kv_cache_dtype_skip_layers:
        raise UnsupportedConfigError(
            "kv_cache_dtype_skip_layers="
            f"{cache_config.kv_cache_dtype_skip_layers} leaves some layers on "
            "a different cache dtype. That splits the model into KV cache "
            "groups with their own metadata builders, so the slot table would "
            "advance more than once per step, and it makes vLLM zero freshly "
            "allocated blocks. Leave it empty."
        )

    if cache_config.kv_offloading_size is not None:
        raise UnsupportedConfigError(
            "KV offloading is not supported by the NVFP4 KV cache: it copies "
            "pages by address and size, which does not describe the packed "
            "FP4 layout. Leave kv_offloading_size unset."
        )

    parallel_config = vllm_config.parallel_config
    if parallel_config is not None and parallel_config.use_ubatching:
        raise UnsupportedConfigError(
            "microbatching is not supported by the NVFP4 KV cache: each "
            "ubatch builds its own attention metadata, so the slot table "
            "would advance twice per step and the second half of the batch "
            "would extend tails the first half had already extended. Leave "
            "enable_dbo unset and ubatch_size at 0."
        )

    if parallel_config is not None and parallel_config.pipeline_parallel_size > 1:
        raise UnsupportedConfigError(
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size} "
            "is not supported by the NVFP4 KV cache: page promotion is driven "
            "from the last attention layer, which a pipeline stage does not "
            "own. Pass pipeline_parallel_size=1."
        )

    compilation_config = vllm_config.compilation_config
    if (
        compilation_config is not None
        and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        raise UnsupportedConfigError(
            f"cudagraph_mode={compilation_config.cudagraph_mode.name} is not "
            "supported by the NVFP4 KV cache: graph capture over the promotion "
            "path is unverified. Pass enforce_eager=True."
        )


def check_layer_supported(impl: AttentionImpl) -> None:
    """Raise unless this attention layer is one the decode kernel can serve.

    Separate from ``check_supported`` because these come from the model rather
    than the engine configuration. Each of them is a modifier the decode kernel
    does not implement, so serving the layer would return plausible attention
    computed against the wrong mask or the wrong scores — wrong quietly, which
    is the failure mode worth spending a check on.
    """
    if impl.attn_type != AttentionType.DECODER:
        raise UnsupportedConfigError(
            f"attention type {impl.attn_type} is not supported by the NVFP4 "
            "KV cache, which is a decoder cache."
        )
    if impl.kv_sharing_target_layer_name is not None:
        raise UnsupportedConfigError(
            "cross-layer KV sharing is not supported by the NVFP4 KV cache: "
            "the BF16 tail is indexed by layer, so a layer that reads another "
            "layer's cache would read its own empty tail."
        )
    unsupported = {
        "sliding window": impl.sliding_window != (-1, -1),
        "logit soft capping": bool(impl.logits_soft_cap),
        "ALiBi": impl.alibi_slopes is not None,
        "attention sinks": impl.sinks is not None,
    }
    named = [name for name, present in unsupported.items() if present]
    if named:
        raise UnsupportedConfigError(
            f"{', '.join(named)} is not supported by the NVFP4 decode kernel, "
            "which computes unmodified scaled dot-product attention over the "
            "whole cache."
        )
