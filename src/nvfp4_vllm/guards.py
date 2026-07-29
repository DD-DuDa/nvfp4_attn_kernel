"""Engine configuration the NVFP4 KV cache path refuses to run under.

These checks cannot live in ``AttentionBackend.validate_configuration``:
``_cached_get_attn_backend`` memoizes on ``AttentionSelectorConfig``, which
carries none of the fields below, so a second engine in the same process would
reuse the first one's verdict and never be validated. ``NVFP4Impl.__init__``
runs once per layer per engine and is not cached.
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode, VllmConfig


NVFP4 = "nvfp4"

# vLLM runs at most ``max_num_seqs`` requests concurrently, and each running
# request needs one BF16 tail slot: the partial page its sequence has not
# filled yet, for every layer. That is 16 MiB per slot on an 8B model, and the
# buffer is preallocated, so the engine cannot admit more rows than slots.
#
# Eight rather than vLLM's default of 256 is a v1 scope limit: wide enough to
# exercise the multi-row decode path, narrow enough to preallocate 128 MiB.
MAX_SLOTS = 8


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

    if cache_config.kv_offloading_size is not None:
        raise UnsupportedConfigError(
            "KV offloading is not supported by the NVFP4 KV cache: it copies "
            "pages by address and size, which does not describe the packed "
            "FP4 layout. Leave kv_offloading_size unset."
        )

    parallel_config = vllm_config.parallel_config
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
