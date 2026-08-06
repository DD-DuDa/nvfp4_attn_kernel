"""Construct an engine that uses this backend, with the K shift already on.

Selecting this cache takes five arguments, and only the first is a choice. The
rest are settings the write path cannot work without: chunked prefill hands it
a prompt that resumes mid-page, a prefix hit skips the write the cache is
assembled from, and the page size is the one the kernel indexes. An engine
carrying some of them and not the others still starts, and then serves from a
different cache without saying so.

The shift is the same shape of problem. It applies only when a vector exists,
and constructing an engine does not produce one -- it is either loaded from a
sidecar or measured at startup. So an engine argument alone cannot turn
centering on, and the version of that mistake that hurts is the quiet one:
``worker_extension_cls`` on its own installs the calibration entry points and
nothing else, which runs, and reports nothing, and is the baseline.
``key_shift`` is here so that the thing the caller wants is the thing they
write.
"""

from __future__ import annotations

import os
from typing import Any

from .backend import PAGE_SIZE
from .calibrate import WORKER_EXTENSION_CLS, calibrate_key_shift
from .guards import MAX_SLOTS
from .runtime import KEY_SHIFT_ENV


REQUIRED_ENGINE_KWARGS: dict[str, Any] = {
    "kv_cache_dtype": "nvfp4",
    "attention_config": {"backend": "CUSTOM"},
    "block_size": PAGE_SIZE,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
}


def build_llm(model: str, *, key_shift: Any = "auto", **kwargs: Any):
    """An ``LLM`` on the NVFP4 cache, with the K shift in force.

    ``key_shift`` is one of:

    ``"auto"``
        Measure the mean from the model itself before serving anything. About
        seven seconds and no data; see :func:`nvfp4_vllm.calibrate_key_shift`.
    a path
        Load a sidecar written earlier, by ``calibrate_key_shift(save_to=...)``
        or by ``tools/key_shift_export.py``. Free.
    ``None``
        No shift, which is what every run before this one did.

    Remaining keyword arguments reach ``LLM`` unchanged. Contradicting one the
    backend requires raises rather than being overruled in silence.
    """
    from vllm import LLM

    contradicted = {
        name: kwargs[name]
        for name, required in REQUIRED_ENGINE_KWARGS.items()
        if name in kwargs and kwargs[name] != required
    }
    if contradicted:
        wanted = {name: REQUIRED_ENGINE_KWARGS[name] for name in contradicted}
        raise ValueError(
            f"this backend requires {wanted}, and was given {contradicted}; "
            "an engine holding only some of these settings still starts, and "
            "serves from a different cache"
        )

    engine = {**REQUIRED_ENGINE_KWARGS, **kwargs, "model": model}
    # One BF16 tail slot per running request, and the slot table is a single
    # CTA that spills past this width. Defaulted rather than bound, since fewer
    # is allowed and vLLM's own default of 1024 is refused outright.
    engine.setdefault("max_num_seqs", MAX_SLOTS)
    # A prefill cannot be split, so it has to fit in one batch, and vLLM's
    # default token budget is below the windows this cache is built for.
    if engine.get("max_model_len") is not None:
        engine.setdefault(
            "max_num_batched_tokens",
            max(engine["max_model_len"], engine["max_num_seqs"]),
        )

    # Read in the worker while the layers are built, so it has to be set before
    # the engine exists. Made to agree with the argument either way: an engine
    # asked for one thing while the environment says another is the kind of
    # disagreement that shows up as an unexplained score.
    if key_shift is None or key_shift == "auto":
        os.environ.pop(KEY_SHIFT_ENV, None)
    else:
        os.environ[KEY_SHIFT_ENV] = str(key_shift)

    supplied = engine.get("worker_extension_cls")
    if supplied is None:
        # Installed even when nothing will be measured, because it also carries
        # the on/off switch, and an engine that cannot be switched has to be
        # rebuilt to answer a benchmark both ways.
        engine["worker_extension_cls"] = WORKER_EXTENSION_CLS
    elif key_shift == "auto" and supplied != WORKER_EXTENSION_CLS:
        raise ValueError(
            f"measuring the shift needs worker_extension_cls="
            f"{WORKER_EXTENSION_CLS!r}, and vLLM accepts one; either pass a "
            "sidecar path instead or mix that class into your own"
        )

    llm = LLM(**engine)
    if key_shift == "auto" and calibrate_key_shift(llm) is None:
        raise RuntimeError(
            "the engine came up without the NVFP4 cache, so there was no K "
            "mean to measure"
        )
    return llm
