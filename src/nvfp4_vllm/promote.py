"""Sealing a filled BF16 tail page into the FP4 cache.

A sequence's trailing tokens live in BF16 because V is packed along the token
axis a whole page at a time, so a partial page cannot be quantized. The step
that brings a tail to exactly one page removes that obstacle, and the page
moves into the block vLLM already reserved for it. Nothing else in the design
ever writes an FP4 page after a decode step.

Two launches per step, always, whether or not any row crossed a boundary. The
alternative — asking the device which rows crossed and launching only then —
costs a synchronization every step to save work on 127 steps out of 128, and
those 128 steps have to pay for the answer anyway. What makes the fixed launch
affordable is that it covers all the layers at once: a launch costs about
49 microseconds whether or not it has work to do, so one per layer would cost
three milliseconds a step against roughly one millisecond of attention.

Timing is the whole correctness argument here. This runs after the last
layer's attention, by which point every layer has read the tail as this step's
metadata describes it and written its own share of the token that filled the
page. Running it a layer early would quantize a page that a later layer has
not finished writing, and the loss would be invisible until it showed up as
slightly worse output.
"""

from __future__ import annotations

from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_tokens_into,
    quantize_value_tokens_into,
)


def launch(metadata, runtime) -> None:
    """Quantize every filled tail page into its block, for every layer.

    ``promotion_pages`` is the whole slot table rather than this step's rows,
    so the launch shape is a constant of the engine. Rows that did not fill a
    page carry -1 and are skipped inside the kernel.

    The regions passed here are one layer's, and only their layout is used;
    every layer's own base address comes from the table alongside, because a
    KV cache is allocated a layer at a time and the layers are not a stride
    apart.
    """
    bases = runtime.destination_bases
    key_packed, key_scales, value_packed, value_scales = runtime.layer_regions(0)
    quantize_key_tokens_into(
        runtime.tail_key_tokens,
        key_packed,
        key_scales,
        metadata.promotion_source_tokens,
        metadata.promotion_pages,
        bases[:2],
    )
    quantize_value_tokens_into(
        runtime.tail_value_tokens,
        value_packed,
        value_scales,
        metadata.promotion_source_tokens,
        metadata.promotion_pages,
        bases[2:],
    )
