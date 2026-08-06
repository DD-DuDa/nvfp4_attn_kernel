"""GPU state one model's NVFP4 layers share for their whole lifetime.

Three things have to outlive a step and be shared by every layer: the BF16 tail
holding each sequence's partial page, the views that reinterpret a layer's
block array as NVFP4 regions, and the buffers a layer quantizes its query
into. The metadata builder cannot own them, because it has no reference to the
attention layers, so they are attached to the layers and created the first
time one of them runs.

That first time is deliberately the profile run, before the KV cache exists.
vLLM sizes the cache from whatever memory is still free once profiling ends, so
anything allocated later would come out of memory already promised to the
cache. Allocating during profiling makes the cache exactly that much smaller,
which is the honest accounting.

Carving the cache views cannot happen that early, since during profiling the
cache is a placeholder with no storage, so the views appear on the first real
write instead.
"""

from __future__ import annotations

import os

import torch
from nvfp4_decode_kernel import RESIDUAL_ROW_TILE

from . import layout


PAGE_SIZE = 128

# Points at a safetensors file holding one ``key_shift`` tensor of shape
# ``[num_layers, num_kv_heads, head_dim]``. Unset means no shift, which is the
# behaviour every existing run has.
KEY_SHIFT_ENV = "NVFP4_KEY_SHIFT"


def load_key_shift(
    num_layers: int, num_kv_heads: int, head_dim: int, device: torch.device
) -> torch.Tensor | None:
    """A constant subtracted from post-RoPE K on its way into the cache.

    Post-RoPE K carries large per-channel offsets, and those offsets are what
    sets the group maximum that the block scale is derived from. Removing them
    before quantizing shrinks the scale, so every element in the group gets a
    finer step.

    Nothing adds the constant back. Attention scores become ``q·k - q·mu``,
    and ``q·mu`` is the same for every key a given query sees, so softmax is
    unchanged. That argument only holds if *every* stored key is shifted, which
    is why the subtraction sits at the single point all three write paths pass
    through rather than inside the quantizer.

    Kept in float32: the offsets are an order of magnitude larger than what is
    left after removing them, so rounding ``mu`` to bfloat16 first would spend
    a few percent of the post-shift range on the rounding error.
    """
    path = os.environ.get(KEY_SHIFT_ENV)
    if not path:
        return None

    from safetensors.torch import load_file

    tensors = load_file(path)
    if "key_shift" not in tensors:
        raise ValueError(
            f"{path} has no 'key_shift' tensor, found {sorted(tensors)}"
        )
    shift = tensors["key_shift"]
    expected = (num_layers, num_kv_heads, head_dim)
    if tuple(shift.shape) != expected:
        raise ValueError(
            f"{path} holds a key_shift of {tuple(shift.shape)}, this model "
            f"needs {expected}"
        )
    return shift.to(device=device, dtype=torch.float32)


class LayerRuntime:
    """The tail buffers, cache views, and query scratch one model shares."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.key_shift = load_key_shift(
            num_layers, num_kv_heads, head_dim, device
        )
        # Held separately from the vector so that turning the shift off keeps
        # what it cost to measure. A/B-ing the two arms in one process is the
        # reason to want that.
        self.key_shift_enabled = True
        # Set only while measuring a shift, which cannot happen at the same
        # time as applying one: the sum has to be taken over the K the model
        # actually produces. Each layer accumulates its own row; the token
        # count is shared, since every layer sees the same tokens.
        self.key_shift_sum: torch.Tensor | None = None
        self.key_shift_tokens = 0

        # One allocation each, indexed by layer, so promotion can take the
        # layer as a grid dimension instead of looping in Python. Zeroed
        # rather than empty because attention reads a whole tail page back and
        # the positions past the sequence must not be denormals or NaNs.
        shape = (num_layers, num_slots, PAGE_SIZE, num_kv_heads, head_dim)
        self.tail_key = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.tail_value = torch.zeros(
            shape, dtype=torch.bfloat16, device=device
        )
        # The same storage seen as one run of tokens, which is what promotion
        # quantizes out of: layer ``l`` slot ``s`` starts at token
        # ``l * num_slots * PAGE_SIZE + s * PAGE_SIZE``.
        self.tail_key_tokens = self.tail_key.view(-1, num_kv_heads, head_dim)
        self.tail_value_tokens = self.tail_value.view(
            -1, num_kv_heads, head_dim
        )

        # Zeroed once here and never again, which is the contract fp4_decode
        # asks of a caller-owned buffer: the quantizer writes row 0 of each
        # tile and nothing writes the rest. Shared by every layer, since a
        # layer is done with it before the next one runs.
        self.query_padded = torch.zeros(
            num_slots,
            RESIDUAL_ROW_TILE,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        # The quantizer's two outputs, sized and shared the same way, so that
        # quantizing a query allocates and fills nothing. The packed query is
        # rewritten whole every call and what it holds now does not matter.
        # The scales need the zeroing more than query_padded does: the layout
        # gives each KV head 32 * 4 row slots and each 64-element k-atom a
        # four-byte E4M3 word, and only the num_heads // num_kv_heads slots
        # that carry a query head are ever written.
        self.query_fp4 = torch.zeros(
            num_slots,
            1,
            num_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device=device,
        )
        self.query_scales = torch.zeros(
            num_slots,
            1,
            num_heads,
            head_dim // 64,
            32,
            4,
            4,
            dtype=torch.uint8,
            device=device,
        )

        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        self._bases: torch.Tensor | None = None
        self._expected_layer = 0

    @property
    def tail_bytes(self) -> int:
        return 2 * self.tail_key.numel() * self.tail_key.element_size()

    @property
    def active_key_shift(self) -> torch.Tensor | None:
        """What the write path should subtract, or None to store K as it is."""
        return self.key_shift if self.key_shift_enabled else None

    def set_key_shift_enabled(self, enabled: bool) -> None:
        """Turn the shift on or off between requests.

        Only between them. Keys already in the cache were stored under
        whichever setting was in force when they were written, so a sequence
        that spans a flip has two different constants in one softmax. The
        caller has to know the cache is drained; nothing here can check it.
        """
        if enabled and self.key_shift is None:
            raise ValueError(
                "no shift has been loaded or measured, so there is nothing "
                "to enable"
            )
        self.key_shift_enabled = enabled

    def begin_key_shift_measurement(self) -> None:
        """Measure the K mean instead of removing it, until told to stop.

        Any shift in force is dropped first. Measuring against a shifted K
        would give the residual mean rather than the mean, and the two are not
        interchangeable: the second measurement would be near zero and would
        look like there was nothing to remove.
        """
        self.key_shift = None
        self.key_shift_sum = torch.zeros(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float64,
            device=self.tail_key.device,
        )
        self.key_shift_tokens = 0

    def finish_key_shift_measurement(self) -> torch.Tensor:
        """Freeze what was measured and start subtracting it."""
        if self.key_shift_sum is None or self.key_shift_tokens == 0:
            raise ValueError(
                "no tokens were measured, so there is no mean to subtract; "
                "begin_key_shift_measurement must precede a forward pass"
            )
        self.key_shift = (self.key_shift_sum / self.key_shift_tokens).float()
        self.key_shift_enabled = True
        self.key_shift_sum = None
        self.key_shift_tokens = 0
        return self.key_shift

    def views(
        self, layer_index: int, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """The four NVFP4 regions of one layer's block array.

        Carving is only address arithmetic, but it builds Python objects and
        this runs once per layer per step, so the result is kept. A layer's
        cache tensor is allocated once and never moves, which is what makes its
        address usable as the key.

        The layer index is recorded alongside, because promotion needs every
        layer's address in one table and this is the only place that learns
        which cache a layer actually uses. Taking it from here rather than
        from vLLM's module registry means promotion writes into the tensor the
        write path wrote into, under the index the write path used, without
        either of those having to be what vLLM believes.
        """
        key = kv_cache.data_ptr()
        carved = self._views.get(key)
        if carved is None:
            carved = layout.carve(kv_cache, self.num_kv_heads, self.head_dim)
            self._views[key] = carved
        if self._by_layer.get(layer_index) is not carved:
            self._by_layer[layer_index] = carved
            self._bases = None
        return carved

    @property
    def destination_bases(self) -> torch.Tensor:
        """``[4, num_layers]`` int64. Where each layer's four regions start.

        Built on first use and kept, because it is a host-to-device copy and
        promotion runs every step. Every layer must have written its cache by
        then, which the execution order below guarantees: promotion fires
        after the last layer, and each one carves its cache before it writes.
        """
        if self._bases is None:
            missing = [
                index
                for index in range(self.num_layers)
                if index not in self._by_layer
            ]
            if missing:
                raise ValueError(
                    f"layers {missing} have not written their KV cache, so "
                    "promotion has no address to send their pages to"
                )
            self._bases = torch.tensor(
                [
                    [
                        self._by_layer[layer][region].data_ptr()
                        for layer in range(self.num_layers)
                    ]
                    for region in range(4)
                ],
                dtype=torch.int64,
                device=self.tail_key.device,
            )
        return self._bases

    def layer_regions(self, layer_index: int) -> tuple[torch.Tensor, ...]:
        """One layer's carved regions, for their layout rather than their
        address. Promotion needs a tensor shaped like a destination; which
        layer it came from does not matter, since they are all carved alike."""
        return self._by_layer[layer_index]

    def expect_layer(self, layer_index: int) -> None:
        """Refuse a step whose layers do not run in the order they are indexed.

        Promotion fires after the highest index, which is the last layer only
        if construction order is execution order. That holds for a decoder
        stack, and nothing in vLLM promises it. If it stopped holding, a page
        would be sealed into FP4 before some layer had written its share of
        the boundary token, and what reached the user would be a model that is
        slightly and inexplicably worse. Two integers a layer is cheaper than
        ever having to debug that.

        The count carries across steps rather than resetting at layer 0, so
        that a step which ran only some of its layers is caught by the next
        step's first layer arriving early. It resynchronizes when it complains,
        because the other way to leave it mid-stack is for something else to
        raise partway through a step, and a stale count would then answer every
        later step with a complaint about layer order instead of about whatever
        actually broke.
        """
        if layer_index != self._expected_layer:
            expected = self._expected_layer
            self._expected_layer = (layer_index + 1) % self.num_layers
            raise ValueError(
                f"layer {layer_index} ran where layer {expected} was expected; "
                "promotion assumes layers run in index order"
            )
        self._expected_layer = (layer_index + 1) % self.num_layers
