"""Measure the post-RoPE K mean from a running engine, then start removing it.

Removing a per-channel constant from K before it is cached shrinks the block
scale and so the quantization step, and softmax does not see the difference.
The constant is the mean, and the mean has to come from somewhere.

Doing it here rather than in a calibration job is a judgement about what the
number costs to obtain. The mean is a far coarser quantity than the pipeline
originally built to produce it implies: one taken from an entirely different
checkpoint still removes 41.8% of the quantization error against the 51.6% a
matched 776,000-token corpus removes. What this procedure gets in seven
seconds, measured the same way on the same held-out sequences, is 48.8%.

What the mean is taken over matters in two ways, and the smaller one is the
text. The seeds below are decoded by the model being served, on the grounds
that its own output is in the distribution it will be asked about, which for a
policy being trained is literally the traffic. At a fixed token budget, using
this text rather than the workload's own cost about five points of the figures
above, so passing real task prompts is an improvement but a small one.

The larger one is position. RoPE mixes the position into K before it is
cached, and the low-frequency channels turn slowly enough that a mean over
positions 0..4k is not the mean over positions 0..16k. Measured on held-out
sequences, 4096 tokens spread across the full window recover 47.8% of the
quantization error while the same 4096 tokens taken from the start recover
40.4% -- the spread-out sample is worth as much as using every token there is.
So the generated text is tiled into one long prompt and prefilled, which also
happens to be the cheap direction: prefill covers positions about ten times
faster per token than decode reaches them.

Timing is the reason this cannot be a background task. The shift must be fixed
before any key is stored under it: a sequence whose early keys were cached
unshifted and whose later keys were not would have two different constants in
one softmax, which is the one way to get this wrong. Calibration therefore
runs to completion, and only then does the shift turn on.

The accumulator lives in the worker process and the caller does not, so the
two sides talk through ``collective_rpc``, which carries a method name rather
than a function. Reaching the runtime any other way means shipping a callable
to the engine, and the only switch that permits that also permits arbitrary
pickle over the same channel.
"""

from __future__ import annotations

from typing import Any

import torch


# vLLM mixes this into the worker, so ``collective_rpc`` can name its methods.
# An engine that wants to calibrate must be built with
# ``worker_extension_cls="nvfp4_vllm.calibrate.NVFP4WorkerExtension"``.
WORKER_EXTENSION_CLS = "nvfp4_vllm.calibrate.NVFP4WorkerExtension"


SEEDS = (
    "Explain how a hash table resolves collisions, and when each strategy "
    "is preferable.",
    "Summarise the causes of the 1929 financial crash for a reader who "
    "knows no economics.",
    "Write a short story about a lighthouse keeper who stops receiving "
    "supply ships.",
    "Derive the quadratic formula from completing the square, showing every "
    "step.",
    "Describe how photosynthesis converts light into chemical energy, at the "
    "level of a first-year undergraduate.",
    "Compare pessimistic and optimistic concurrency control in databases.",
    "Translate the idea of technical debt into terms a non-engineer manager "
    "would act on.",
    "Walk through debugging a service that is slow only under high load.",
)
TOKENS_PER_SEED = 256
# Reaching the top of the window matters more than what is up there, and the
# cost is one prefill. Clamped to what the engine will accept.
CONTEXT_TOKENS = 32_768


def _runtimes(model: torch.nn.Module) -> list[Any]:
    """The distinct LayerRuntime objects this model's attention layers share."""
    from .impl import NVFP4Impl

    seen: list[Any] = []
    for module in model.modules():
        impl = getattr(module, "impl", None)
        runtime = getattr(impl, "runtime", None)
        if isinstance(impl, NVFP4Impl) and runtime is not None:
            if not any(runtime is other for other in seen):
                seen.append(runtime)
    return seen


class NVFP4WorkerExtension:
    """The two calibration calls, reachable by name from the engine client.

    Neither returns the vector. A tensor sent back as an untyped utility
    result decodes to the three fields the encoder split it into rather than
    to a tensor, and reassembling it on the far side would mean depending on
    that private layout. The worker writes the sidecar itself instead, and
    what crosses the wire is small enough to read in a log.
    """

    def nvfp4_key_shift_is_active(self) -> bool:
        return all(
            runtime.active_key_shift is not None
            for runtime in _runtimes(self.get_model())
        )

    def nvfp4_set_key_shift_enabled(self, enabled: bool) -> int:
        runtimes = _runtimes(self.get_model())
        for runtime in runtimes:
            runtime.set_key_shift_enabled(enabled)
        return len(runtimes)

    def nvfp4_begin_key_shift_measurement(self) -> int:
        runtimes = _runtimes(self.get_model())
        for runtime in runtimes:
            runtime.begin_key_shift_measurement()
        return len(runtimes)

    def nvfp4_finish_key_shift_measurement(
        self, save_to: str | None = None
    ) -> dict[str, Any] | None:
        runtimes = _runtimes(self.get_model())
        if not runtimes:
            return None
        tokens = runtimes[0].key_shift_tokens
        shift = runtimes[0].finish_key_shift_measurement()
        for other in runtimes[1:]:
            other.finish_key_shift_measurement()

        if save_to is not None:
            from safetensors.torch import save_file

            save_file(
                {"key_shift": shift.cpu()},
                save_to,
                metadata={"schema": "1", "source": "calibrate_key_shift"},
            )
        return {
            "tokens": int(tokens),
            "shape": list(shift.shape),
            "mean_abs": float(shift.abs().mean()),
            "saved_to": save_to,
        }


def set_key_shift_enabled(llm, enabled: bool) -> bool:
    """Turn the shift on or off, for measuring one arm against the other.

    The vector survives being turned off, so a single loaded model can answer
    a benchmark both ways. Returns whether the engine had a shift to switch.

    Safe only between requests. Keys already cached were written under
    whichever setting was in force at the time, and a sequence spanning a flip
    would see two constants in one softmax. Calling this between two
    ``llm.generate`` calls satisfies that, since those are synchronous and the
    cache is dropped with the requests that filled it.
    """
    switched = llm.collective_rpc(
        "nvfp4_set_key_shift_enabled", args=(enabled,)
    )
    llm.reset_prefix_cache()
    return any(switched)


def _long_prompt(llm, texts: list[str], context_tokens: int) -> list[int]:
    """One prompt of about ``context_tokens`` tokens, tiled from ``texts``.

    Tiling repeats the sample rather than extending it, which is fine for a
    mean: the point of the long prompt is to visit high positions, and every
    position gets visited once whatever sits there.
    """
    if not any(text.strip() for text in texts):
        raise ValueError(
            "the model returned no calibration text to tile, so the mean "
            "would be measured over separators"
        )
    unit = llm.get_tokenizer().encode(
        "\n\n".join(texts) + "\n\n", add_special_tokens=False
    )
    repeats = -(-context_tokens // len(unit))
    return (unit * repeats)[:context_tokens]


def calibrate_key_shift(
    llm,
    *,
    seeds: tuple[str, ...] = SEEDS,
    tokens_per_seed: int = TOKENS_PER_SEED,
    context_tokens: int = CONTEXT_TOKENS,
    save_to: str | None = None,
) -> dict[str, Any] | None:
    """Sample the model, then subtract the K mean it measured from here on.

    Returns what was measured -- token count, shape, mean magnitude -- or
    ``None`` if the engine is not running the NVFP4 cache and there was
    nothing to calibrate.

    ``save_to`` writes a sidecar that ``NVFP4_KEY_SHIFT`` can point at, so a
    later launch of the same checkpoint can skip this. It is refused above one
    worker: each worker measures the KV heads it was given, so the vector is
    per rank and a single file would describe only one of them. Calibrating
    without saving is correct at any degree of parallelism.
    """
    from vllm import SamplingParams, TokensPrompt

    try:
        found = llm.collective_rpc("nvfp4_begin_key_shift_measurement")
    except AttributeError as error:
        raise RuntimeError(
            "the engine was built without "
            f'worker_extension_cls="{WORKER_EXTENSION_CLS}", so the K mean '
            "cannot be read out of the worker"
        ) from error
    if not any(found):
        return None
    if save_to is not None and len(found) > 1:
        raise ValueError(
            f"save_to is for a single worker; this engine has {len(found)}, "
            "each holding a different slice of the KV heads"
        )

    sampled = llm.generate(
        list(seeds),
        SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=tokens_per_seed, seed=42
        ),
        use_tqdm=False,
    )

    budget = min(
        context_tokens, llm.llm_engine.model_config.max_model_len - 1
    )
    llm.generate(
        TokensPrompt(
            prompt_token_ids=_long_prompt(
                llm,
                [
                    output.prompt + output.outputs[0].text
                    for output in sampled
                ],
                budget,
            )
        ),
        SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )

    measured = llm.collective_rpc(
        "nvfp4_finish_key_shift_measurement", args=(save_to,)
    )[0]

    # Every key cached during the measurement was stored unshifted, and the
    # next request must not read one. With prefix caching refused -- which the
    # NVFP4 engine configuration requires anyway -- a finished request leaves
    # nothing behind, but saying so costs nothing and does not depend on that.
    llm.reset_prefix_cache()
    return measured
