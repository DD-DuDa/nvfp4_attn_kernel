"""Out-of-tree NVFP4 KV cache backend for vLLM V1."""


def register() -> None:
    """Entry point for the ``vllm.general_plugins`` group."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "nvfp4_vllm.backend.NVFP4Backend",
    )


__all__ = ["register"]
