"""GPU state one model's NVFP4 layers share for their whole lifetime.

Three things have to outlive a step and be shared by every layer: the BF16 tail
holding each sequence's partial page, the views that reinterpret a layer's
block array as NVFP4 regions, and the buffer the decode kernel reads its BF16
query through. The metadata builder cannot own them, because it has no
reference to the attention layers, so they are attached to the layers and
created the first time one of them runs.

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

import torch
from nvfp4_decode_kernel import RESIDUAL_ROW_TILE

from . import layout


PAGE_SIZE = 128


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
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # One allocation each, indexed by layer, so promotion can take the
        # layer as a grid dimension instead of looping in Python. Zeroed
        # rather than empty because attention reads a whole tail page back and
        # the positions past the sequence must not be denormals or NaNs.
        shape = (num_layers, num_slots, PAGE_SIZE, num_kv_heads, head_dim)
        self.tail_key = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.tail_value = torch.zeros(
            shape, dtype=torch.bfloat16, device=device
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

        self._views: dict[int, tuple[torch.Tensor, ...]] = {}

    @property
    def tail_bytes(self) -> int:
        return 2 * self.tail_key.numel() * self.tail_key.element_size()

    def views(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The four NVFP4 regions of one layer's block array.

        Carving is only address arithmetic, but it builds Python objects and
        this runs once per layer per step, so the result is kept. A layer's
        cache tensor is allocated once and never moves, which is what makes its
        address usable as the key.
        """
        key = kv_cache.data_ptr()
        carved = self._views.get(key)
        if carved is None:
            carved = layout.carve(kv_cache, self.num_kv_heads, self.head_dim)
            self._views[key] = carved
        return carved
