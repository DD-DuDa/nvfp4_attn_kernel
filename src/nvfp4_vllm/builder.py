"""Metadata builder that advances the tail-slot control plane once per step.

vLLM calls ``build()`` once per scheduler step for each attention group, before
any layer runs. That is the only place in a step where the batch is described
in full and nothing has executed yet, which is what the control plane needs:
it decides which BF16 tail slot each request owns, and every layer downstream
reads that decision rather than recomputing it.

The step counter inside the control plane therefore has to track scheduler
steps exactly. Two configurations would break that and are refused in
``guards``: microbatching builds metadata per ubatch, and full CUDA graphs
build metadata during capture, neither of which is a step.

Startup is not covered by those guards and does not need to be. Sizing the KV
cache and warming the kernels both run batches through this path before any
request exists, so the control plane sees a few steps before serving begins.
The dummy batch among them keys on the null block and is ignored; the warmup
batches look exactly like real requests, take slots, and are then abandoned,
which is the case lazy reclamation already handles.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

from .control import ControlPlane
from .guards import NVFP4, check_supported
from .metadata import NVFP4Metadata
from .write import PageWorkTable


def decode_split(
    common_attn_metadata: CommonAttentionMetadata,
) -> tuple[int, int]:
    """Where this step's decode prefix ends, in rows and in tokens.

    The one host assertion in this module, and the exception §6.3 allows: it
    reads ``query_start_loc_cpu``, which the model runner already materialized,
    so it costs no synchronization. What it guards is the premise the whole
    read path rests on — that the reordering actually happened. Without it, a
    vLLM that stopped reordering would have the decode kernel attend to prefill
    rows' caches with decode rows' queries, which is wrong everywhere and loud
    nowhere.
    """
    num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
        common_attn_metadata, decode_threshold=1
    )
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    query_lens = query_start_loc[1 : common_attn_metadata.num_reqs + 1] - (
        query_start_loc[: common_attn_metadata.num_reqs]
    )
    if not bool((query_lens[:num_decodes] == 1).all()) or not bool(
        (query_lens[num_decodes:] > 1).all()
    ):
        raise ValueError(
            "the NVFP4 read path needs one-token requests at the front of the "
            f"batch, but query lengths are {query_lens.tolist()} with a decode "
            f"prefix of {num_decodes}"
        )
    return num_decodes, num_decode_tokens


class NVFP4MetadataBuilder(FlashAttentionMetadataBuilder):
    """FlashAttention's builder, extended with this step's slot assignment.

    A pass-through under any other cache dtype: the backend is selected per
    engine, not per cache dtype, so this class is also what a BF16 run gets.

    ``supports_update_block_table`` stays inherited. It is what keeps a model
    with several KV cache groups to one ``build()`` per step, and the base
    implementation shallow-copies the metadata object, so the slot fields
    below survive the copy.

    Under NVFP4 it also asks vLLM to sort decode requests to the front of the
    batch; see ``decode_split``.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        cache_config = vllm_config.cache_config
        self.nvfp4 = cache_config is not None and cache_config.cache_dtype == NVFP4
        if not self.nvfp4:
            self.plane = None
            return

        # Ask the model runner to move one-token requests to the front of the
        # batch. That makes the decode rows a contiguous prefix, so the decode
        # kernel takes slices of this step's arrays instead of a compacted copy
        # of them, and its output is a slice of vLLM's. Left unset above, so a
        # BF16 run through this backend keeps FlashAttention's batch order.
        self._init_reorder_batch_threshold(1)

        # Already enforced when the layers were constructed. Repeated here so a
        # builder is safe to construct on its own, and so the control plane is
        # never sized from a configuration it cannot serve.
        check_supported(vllm_config)

        scheduler_config = vllm_config.scheduler_config
        self.plane = ControlPlane(
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            device=device,
        )
        # Which full pages this step writes depends only on the batch, so it is
        # resolved here alongside the slot assignment rather than once per
        # layer.
        self.work_table = PageWorkTable(
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.nvfp4:
            return base

        # seq_lens is read as num_computed_tokens + num_scheduled_tokens, and
        # the FP4/BF16 page split is derived from it, so the other reading would
        # move every page boundary by one step. Nothing asserts it here: the
        # host copies the model runner computes it from are not on
        # CommonAttentionMetadata, and the reading is self-reporting anyway. If
        # it ever flips, a fresh prefill arrives with a zero length, is taken
        # for padding, gets no slot, and returns one step later with history it
        # cannot place, which is ERR_SLOT_LOST.
        outputs = self.plane.prepare(
            block_table=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )

        work_table = self.work_table.build(
            query_start_loc=common_attn_metadata.query_start_loc,
            seqused_fp4=outputs.seqused_fp4,
            row_to_slot=outputs.row_to_slot,
            block_table=common_attn_metadata.block_table_tensor,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
        )

        return NVFP4Metadata.from_flash(
            base, outputs, work_table, decode_split(common_attn_metadata)
        )
