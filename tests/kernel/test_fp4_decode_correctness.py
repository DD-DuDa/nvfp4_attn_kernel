"""Numerical and execution contracts for the public FP4 decode API.

The kernel accepts a BF16 query, prequantized full K/V pages, and an optional
BF16 residual page per row. It quantizes Q/P internally and returns the
logical decode output.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import flash_attn_func
from nvfp4_decode_kernel import _quantize
from nvfp4_decode_kernel import fp4_decode


FP4_MIN_COSINE = 0.99
FP4_MAX_ABS_ERROR = 5e-2
QUALITY_COSINE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class AttentionCase:
    batch: int
    pages_per_row: int
    seqused_k: tuple[int, ...]
    query_heads: int
    kv_heads: int


CASES = [
    pytest.param(AttentionCase(1, 2, (256,), 32, 8), id="gqa-two-pages"),
    pytest.param(AttentionCase(1, 1, (128,), 32, 8), id="gqa-one-page"),
    pytest.param(AttentionCase(1, 2, (200,), 32, 8), id="gqa-partial-page"),
    pytest.param(AttentionCase(1, 64, (8192,), 32, 8), id="gqa-long-context"),
    pytest.param(
        AttentionCase(4, 2, (256, 200, 128, 255), 32, 8),
        id="gqa-multi-batch",
    ),
    pytest.param(
        AttentionCase(7, 3, (1, 127, 128, 129, 255, 256, 257), 32, 8),
        id="gqa-vllm-boundaries",
    ),
    pytest.param(AttentionCase(1, 2, (256,), 16, 16), id="mha"),
    pytest.param(AttentionCase(1, 2, (256,), 16, 1), id="mqa"),
]


def _output(result):
    return result[0] if isinstance(result, tuple) else result


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().flatten(),
        b.float().flatten(),
        dim=0,
    ).item()


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to the representable E2M1 values."""

    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=x.device,
    )
    absolute = x.abs()
    indices = (absolute.unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    return torch.sign(x) * magnitudes[indices]


def _nvfp4_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize groups of 16 values with NVFP4."""
    assert x.shape[-1] % 16 == 0
    groups = x.float().reshape(*x.shape[:-1], -1, 16)
    scale = groups.abs().amax(dim=-1, keepdim=True) / 6.0
    scale = scale.to(torch.float8_e4m3fn).float()
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = _round_e2m1(groups / safe_scale) * scale
    return quantized.reshape_as(x)


def _fp4_qkv_pages_round_trip(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantized NVFP4 Q/K/V in their original logical layouts."""
    q_fp4 = _nvfp4_round_trip(q).to(q.dtype)
    k_fp4 = _nvfp4_round_trip(k_pages).to(k_pages.dtype)
    v_fp4 = _nvfp4_round_trip(
        v_pages.permute(0, 2, 3, 1).contiguous()
    ).permute(0, 3, 1, 2).to(v_pages.dtype)
    return q_fp4, k_fp4, v_fp4


def _hybrid_qkv_pages_round_trip(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    residual_page_ids: torch.Tensor,
    seqused_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantized FP4 full pages with their BF16 residual pages restored."""
    q_fp4, k_hybrid, v_hybrid = _fp4_qkv_pages_round_trip(
        q, k_pages, v_pages
    )
    has_residual = seqused_residual > 0
    residual_ids = residual_page_ids[has_residual].long()
    if residual_ids.numel() > 0:
        k_hybrid[residual_ids] = k_pages[residual_ids]
        v_hybrid[residual_ids] = v_pages[residual_ids]
    return q_fp4, k_hybrid, v_hybrid


def _gather_pages(
    pages: torch.Tensor,
    page_table: torch.Tensor,
) -> torch.Tensor:
    """Gather logical pages into dense per-batch sequences for references."""
    batch, pages_per_row = page_table.shape
    page_size, heads, head_dim = pages.shape[1:]
    gathered = pages.index_select(0, page_table.reshape(-1).long())
    return gathered.reshape(
        batch,
        pages_per_row * page_size,
        heads,
        head_dim,
    )


def _flash_attention_reference(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run dense FlashAttention row-by-row for variable paged lengths."""
    outputs = []
    for row, length in enumerate(seqused_k.tolist()):
        row_table = page_table[row : row + 1]
        k = _gather_pages(k_pages, row_table)[:, :length]
        v = _gather_pages(v_pages, row_table)[:, :length]
        outputs.append(
            _output(
                flash_attn_func(
                    q[row : row + 1],
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            )
        )
    return torch.cat(outputs, dim=0)


def _torch_fp4_decode(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    residual_page_ids: torch.Tensor,
    seqused_residual: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Independent PyTorch simulation of hybrid FP4/BF16 decode."""
    q_fp4, k_fp4_pages, v_fp4_pages = _hybrid_qkv_pages_round_trip(
        q,
        k_pages,
        v_pages,
        residual_page_ids,
        seqused_residual,
    )
    q_fp4 = q_fp4.float()
    k_fp4 = _gather_pages(k_fp4_pages, page_table).float()
    v_fp4 = _gather_pages(v_fp4_pages, page_table).float()

    repeats = q.shape[2] // k_fp4.shape[2]
    k_fp4 = k_fp4.repeat_interleave(repeats, dim=2)
    v_fp4 = v_fp4.repeat_interleave(repeats, dim=2)

    scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        q_fp4,
        k_fp4,
    ) * softmax_scale

    positions = torch.arange(scores.shape[-1], device=scores.device)
    valid = positions.unsqueeze(0) < seqused_k.unsqueeze(1)
    scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    scores = scores - scores.amax(dim=-1, keepdim=True)
    p_exp = torch.exp(scores)
    denominator = p_exp.sum(dim=-1, keepdim=True)

    p_fp4 = _nvfp4_round_trip(p_exp)

    output = torch.einsum(
        "bhqk,bkhd->bqhd",
        p_fp4 / denominator,
        v_fp4,
    )
    return output


def _interleaved_page_cache(*regions: torch.Tensor) -> list[torch.Tensor]:
    """Repack dense page arrays into one page-major cache, the way vLLM stores.

    A vLLM block holds all four regions of a page inside a single window, so
    each region's page stride becomes the whole window instead of its own size.
    The bytes are unchanged; only the addressing is. Every region must already
    be dense below its page axis, which is what the quantizers return.
    """
    pages = regions[0].shape[0]
    pitches = [region.stride()[0] * region.element_size() for region in regions]
    page_bytes = sum(pitches)
    cache = torch.zeros(
        pages, page_bytes, dtype=torch.uint8, device=regions[0].device
    )

    views = []
    offset = 0
    for region, pitch in zip(regions, pitches):
        flat = region.as_strided((pages, pitch), (pitch, 1)).view(torch.uint8)
        cache[:, offset : offset + pitch] = flat
        typed = cache if region.dtype is torch.uint8 else cache.view(region.dtype)
        views.append(
            typed.as_strided(
                region.shape, (page_bytes,) + region.stride()[1:], offset
            )
        )
        offset += pitch
    return views


@pytest.fixture(scope="module", params=CASES)
def outputs(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")

    case: AttentionCase = request.param
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    query_length = 1
    page_size = 128
    head_dim = 128
    softmax_scale = head_dim**-0.5
    num_pages = case.batch * case.pages_per_row

    q = torch.randn(
        case.batch,
        query_length,
        case.query_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    k_pages = torch.randn(
        num_pages,
        page_size,
        case.kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    page_table = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    ).reshape(case.batch, case.pages_per_row)
    seqused_k = torch.tensor(
        case.seqused_k,
        dtype=torch.int32,
        device="cuda",
    )
    seqused_fp4 = ((seqused_k - 1) // page_size) * page_size
    seqused_residual = ((seqused_k - 1) % page_size) + 1
    residual_columns = (seqused_fp4 // page_size).clamp(
        max=case.pages_per_row - 1
    )
    residual_page_ids = page_table.gather(
        1, residual_columns.long().unsqueeze(1)
    ).squeeze(1)
    residual_page_ids = torch.where(
        seqused_residual > 0,
        residual_page_ids,
        torch.zeros_like(residual_page_ids),
    )

    with torch.no_grad():
        q_fp4, k_hybrid_pages, v_hybrid_pages = (
            _hybrid_qkv_pages_round_trip(
                q,
                k_pages,
                v_pages,
                residual_page_ids,
                seqused_residual,
            )
        )
        key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
        value_pages_fp4, value_scales = _quantize.quantize_value_pages(
            v_pages
        )

        def decode(k_fp4, k_sf, v_fp4, v_sf):
            return _output(
                fp4_decode(
                    query=q[:, 0],
                    key_pages_fp4=k_fp4,
                    key_scales=k_sf,
                    value_pages_fp4=v_fp4,
                    value_scales=v_sf,
                    fp4_page_table=page_table,
                    seqused_fp4=seqused_fp4,
                    residual_key_pages_bf16=k_pages,
                    residual_value_pages_bf16=v_pages,
                    residual_page_ids=residual_page_ids,
                    seqused_residual=seqused_residual,
                    has_bf16=torch.ones_like(
                        seqused_residual, dtype=torch.bool
                    ),
                    softmax_scale=softmax_scale,
                )
            ).unsqueeze(1)

        fp4_output = decode(
            key_pages_fp4, key_scales, value_pages_fp4, value_scales
        )
        paged_output = decode(
            *_interleaved_page_cache(
                key_pages_fp4, key_scales, value_pages_fp4, value_scales
            )
        )

        bf16_output = _flash_attention_reference(
            q,
            k_pages,
            v_pages,
            page_table,
            seqused_k,
            softmax_scale,
            False,
        )

        flash_fp4_output = _flash_attention_reference(
            q_fp4,
            k_hybrid_pages,
            v_hybrid_pages,
            page_table,
            seqused_k,
            softmax_scale,
            False,
        )

        torch_fp4_output = _torch_fp4_decode(
            q,
            k_pages,
            v_pages,
            page_table,
            seqused_k,
            residual_page_ids,
            seqused_residual,
            softmax_scale,
        )

    torch.cuda.synchronize()

    return (
        fp4_output,
        bf16_output,
        flash_fp4_output,
        torch_fp4_output,
        paged_output,
    )


def test_fp4_decode_reads_pages_carved_out_of_a_vllm_block(outputs):
    """Page stride must not change a single bit of the result.

    Same bytes, same kernel, only the distance between pages differs, so any
    difference is an addressing bug rather than a numerical one. Bit equality
    also carries every oracle below over to the interleaved layout without
    running them twice.
    """
    fp4_output, _, _, _, paged_output = outputs

    assert torch.equal(fp4_output, paged_output), (
        "decoding from pages interleaved at vLLM's block pitch disagreed with "
        "decoding from densely packed pages"
    )


def test_fp4_decode_quality_against_bf16_fa4(outputs):
    fp4_output, bf16_output, _, torch_fp4_output, _ = outputs

    assert torch.isfinite(fp4_output).all()

    # BF16 attention is a quality baseline, not an exact oracle: the kernel
    # quantizes Q/K/V and the unnormalized softmax probabilities P. Match the
    # independent FP4 simulation's quality, rather than requiring FP4 output
    # to equal BF16 output.
    kernel_cosine = _cosine(fp4_output, bf16_output)
    oracle_cosine = _cosine(torch_fp4_output, bf16_output)

    assert kernel_cosine >= oracle_cosine - QUALITY_COSINE_TOLERANCE, (
        f"kernel vs BF16 FlashAttention cosine={kernel_cosine:.6f}, "
        f"PyTorch FP4 oracle cosine={oracle_cosine:.6f}"
    )


def test_fp4_decode_matches_flash_attention_on_hybrid_qkv(outputs):
    fp4_output, _, flash_fp4_output, torch_fp4_output, _ = outputs
    kernel_cosine = _cosine(fp4_output, flash_fp4_output)
    oracle_cosine = _cosine(torch_fp4_output, flash_fp4_output)

    assert kernel_cosine >= oracle_cosine - QUALITY_COSINE_TOLERANCE, (
        "kernel adds more quality loss than its FP4 arithmetic permits: "
        f"kernel vs FlashAttention cosine={kernel_cosine:.6f}, "
        f"PyTorch FP4 oracle cosine={oracle_cosine:.6f}"
    )


def test_fp4_decode_matches_torch_hybrid_semantics(outputs):
    fp4_output, _, _, torch_fp4_output, _ = outputs

    cosine = _cosine(fp4_output, torch_fp4_output)
    row_cosines = [
        _cosine(actual, expected)
        for actual, expected in zip(fp4_output, torch_fp4_output)
    ]
    max_abs = (
        fp4_output.float() - torch_fp4_output.float()
    ).abs().max().item()

    assert cosine >= FP4_MIN_COSINE, (
        f"kernel vs PyTorch hybrid FP4/BF16 cosine={cosine:.6f}, "
        f"row_cosines={row_cosines}"
    )
    assert min(row_cosines) >= FP4_MIN_COSINE, (
        "kernel vs PyTorch hybrid FP4/BF16 per-row cosine below threshold: "
        f"{row_cosines}"
    )
    assert max_abs <= FP4_MAX_ABS_ERROR, (
        f"kernel vs PyTorch hybrid FP4/BF16 max_abs={max_abs:.6f}"
    )


@dataclass(frozen=True)
class ContractDecodeInputs:
    query: torch.Tensor
    key_pages_bf16: torch.Tensor
    value_pages_bf16: torch.Tensor
    key_pages_fp4: torch.Tensor
    key_scales: torch.Tensor
    value_pages_fp4: torch.Tensor
    value_scales: torch.Tensor
    page_table: torch.Tensor
    seqused_fp4: torch.Tensor
    residual_page_ids: torch.Tensor
    seqused_residual: torch.Tensor
    query_row_indices: torch.Tensor


@pytest.fixture(scope="module")
def contract_decode_inputs() -> ContractDecodeInputs:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        pytest.skip(f"SM100 is required, found compute capability {capability}")

    pytest.importorskip("cutlass")

    torch.manual_seed(0xDEC0DE)
    query = torch.randn(
        5,
        32,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.3
    key_pages_bf16 = torch.randn(
        6,
        128,
        8,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.3
    value_pages_bf16 = torch.randn_like(key_pages_bf16) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(key_pages_bf16)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(
        value_pages_bf16
    )

    return ContractDecodeInputs(
        query=query,
        key_pages_bf16=key_pages_bf16,
        value_pages_bf16=value_pages_bf16,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        page_table=torch.tensor(
            [[0, 1], [2, 3], [4, 5]],
            dtype=torch.int32,
            device="cuda",
        ),
        seqused_fp4=torch.full(
            (3,),
            128,
            dtype=torch.int32,
            device="cuda",
        ),
        residual_page_ids=torch.tensor(
            [0, 3, 5],
            dtype=torch.int32,
            device="cuda",
        ),
        seqused_residual=torch.tensor(
            [0, 72, 127],
            dtype=torch.int32,
            device="cuda",
        ),
        query_row_indices=torch.tensor(
            [4, 1, 3],
            dtype=torch.int32,
            device="cuda",
        ),
    )


def _run_contract_decode(
    inputs: ContractDecodeInputs,
    query: torch.Tensor,
    page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    *,
    query_row_indices: torch.Tensor | None = None,
    with_residual: bool,
) -> torch.Tensor:
    residual = {}
    if with_residual:
        rows = page_table.shape[0]
        residual = {
            "residual_key_pages_bf16": inputs.key_pages_bf16,
            "residual_value_pages_bf16": inputs.value_pages_bf16,
            "residual_page_ids": inputs.residual_page_ids[:rows],
            "seqused_residual": inputs.seqused_residual[:rows],
            "has_bf16": inputs.seqused_residual[:rows] > 0,
        }
    return fp4_decode(
        query,
        inputs.key_pages_fp4,
        inputs.key_scales,
        inputs.value_pages_fp4,
        inputs.value_scales,
        page_table,
        seqused_fp4,
        query_row_indices=query_row_indices,
        **residual,
    )


def test_mixed_residual_batch_keeps_zero_length_row_exact(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    with torch.no_grad():
        hybrid = _run_contract_decode(
            inputs,
            inputs.query,
            inputs.page_table,
            inputs.seqused_fp4,
            query_row_indices=inputs.query_row_indices,
            with_residual=True,
        )
        pure_first_row = _run_contract_decode(
            inputs,
            inputs.query[4:5],
            inputs.page_table[:1],
            inputs.seqused_fp4[:1],
            with_residual=False,
        )
    torch.cuda.synchronize()

    assert hybrid.shape == (3, 32, 128)
    assert torch.isfinite(hybrid).all()
    assert torch.equal(hybrid[:1], pure_first_row)


def test_query_row_indices_match_compact_query(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    compact_query = inputs.query.index_select(
        0, inputs.query_row_indices.long()
    )
    with torch.no_grad():
        indexed = _run_contract_decode(
            inputs,
            inputs.query,
            inputs.page_table,
            inputs.seqused_fp4,
            query_row_indices=inputs.query_row_indices,
            with_residual=False,
        )
        compact = _run_contract_decode(
            inputs,
            compact_query,
            inputs.page_table,
            inputs.seqused_fp4,
            with_residual=False,
        )
    torch.cuda.synchronize()

    assert torch.equal(indexed, compact)


def test_unused_fp4_page_table_tail_may_be_poisoned(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    valid_table = inputs.page_table[:1].clone()
    poisoned_table = valid_table.clone()
    poisoned_table[:, 1:] = -1
    seqused_fp4 = inputs.seqused_fp4[:1]

    with torch.no_grad():
        expected = _run_contract_decode(
            inputs,
            inputs.query[:1],
            valid_table,
            seqused_fp4,
            with_residual=False,
        )
        actual = _run_contract_decode(
            inputs,
            inputs.query[:1],
            poisoned_table,
            seqused_fp4,
            with_residual=False,
        )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_out_indices_scatter_into_vllm_output(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    out_indices = torch.tensor(
        [3, 0, 4], dtype=torch.int32, device=inputs.query.device
    )
    sentinel = torch.tensor(
        17.0, dtype=torch.bfloat16, device=inputs.query.device
    )
    output = torch.full_like(inputs.query, sentinel)

    with torch.no_grad():
        compact = _run_contract_decode(
            inputs,
            inputs.query,
            inputs.page_table,
            inputs.seqused_fp4,
            query_row_indices=inputs.query_row_indices,
            with_residual=True,
        )
        returned = fp4_decode(
            inputs.query,
            inputs.key_pages_fp4,
            inputs.key_scales,
            inputs.value_pages_fp4,
            inputs.value_scales,
            inputs.page_table,
            inputs.seqused_fp4,
            residual_key_pages_bf16=inputs.key_pages_bf16,
            residual_value_pages_bf16=inputs.value_pages_bf16,
            residual_page_ids=inputs.residual_page_ids,
            seqused_residual=inputs.seqused_residual,
            has_bf16=inputs.seqused_residual > 0,
            query_row_indices=inputs.query_row_indices,
            out=output,
            out_indices=out_indices,
        )
    torch.cuda.synchronize()

    assert returned is output
    assert torch.equal(output.index_select(0, out_indices.long()), compact)
    assert torch.equal(output[1], torch.full_like(output[1], sentinel))
    assert torch.equal(output[2], torch.full_like(output[2], sentinel))


@pytest.mark.parametrize(
    ("heads_q", "heads_kv"),
    [(8, 8), (32, 8), (32, 1)],
)
def test_prequantized_query_matches_bf16_query_exactly(
    heads_q: int,
    heads_kv: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    torch.manual_seed(0xF04 + heads_kv)
    query = torch.randn(
        2, heads_q, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        4, 128, heads_kv, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    query_fp4, query_scales = _quantize.quantize_query(
        query, heads_kv=heads_kv
    )
    page_table = torch.tensor(
        [[0, 1], [2, 3]], device="cuda", dtype=torch.int32
    )
    seqused_fp4 = torch.full(
        (2,), 256, device="cuda", dtype=torch.int32
    )

    with torch.no_grad():
        bf16_query_output = fp4_decode(
            query,
            key_pages_fp4,
            key_scales,
            value_pages_fp4,
            value_scales,
            page_table,
            seqused_fp4,
        )
        fp4_query_output = fp4_decode(
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=page_table,
            seqused_fp4=seqused_fp4,
            query_fp4=query_fp4,
            query_scales=query_scales,
        )
    torch.cuda.synchronize()
    assert torch.equal(fp4_query_output, bf16_query_output)


@pytest.mark.parametrize(
    ("heads_q", "heads_kv", "num_splits"),
    [
        (16, 16, 2),
        (32, 8, 4),
        (16, 1, 8),
    ],
)
def test_split_k_matches_unsplit_pure_fp4(
    heads_q: int,
    heads_kv: int,
    num_splits: int,
) -> None:
    """Pure FP4 split-K keeps the existing numerical gate for MHA/GQA/MQA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    from nvfp4_decode_kernel._decode import decode_fp4, decode_fp4_split

    torch.manual_seed(0x5A17 + heads_kv)
    pages = 8
    head_dim = 128
    query = torch.randn(
        1, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        pages,
        128,
        heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    query_fp4, query_scales = _quantize.quantize_query(
        query, heads_kv=heads_kv
    )
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    page_table = torch.arange(
        pages, device="cuda", dtype=torch.int32
    ).reshape(1, pages)
    seqused_fp4 = torch.full(
        (1,), pages * 128, device="cuda", dtype=torch.int32
    )
    common = dict(
        query_fp4=query_fp4,
        query_scales=query_scales,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        fp4_page_table=page_table,
        seqused_fp4=seqused_fp4,
        softmax_scale=head_dim**-0.5,
    )

    with torch.no_grad():
        unsplit = decode_fp4(query_padded_bf16=None, **common)
        split = decode_fp4_split(num_splits=num_splits, **common)
    torch.cuda.synchronize()

    cosine = _cosine(split, unsplit)
    max_abs = (split.float() - unsplit.float()).abs().max().item()
    assert cosine >= FP4_MIN_COSINE
    assert max_abs <= FP4_MAX_ABS_ERROR


def test_split_k_empty_partitions_are_ignored() -> None:
    """Unused split CTAs must contribute LSE=-inf and no output weight."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    from nvfp4_decode_kernel._decode import decode_fp4, decode_fp4_split

    torch.manual_seed(0xE117)
    heads_q, heads_kv, head_dim = 16, 1, 128
    query = torch.randn(
        1, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        1, 128, heads_kv, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    query_fp4, query_scales = _quantize.quantize_query(
        query, heads_kv=heads_kv
    )
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    common = dict(
        query_fp4=query_fp4,
        query_scales=query_scales,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        fp4_page_table=torch.zeros(
            1, 1, device="cuda", dtype=torch.int32
        ),
        seqused_fp4=torch.full(
            (1,), 128, device="cuda", dtype=torch.int32
        ),
        softmax_scale=head_dim**-0.5,
    )
    with torch.no_grad():
        unsplit = decode_fp4(query_padded_bf16=None, **common)
        split = decode_fp4_split(num_splits=8, **common)
    torch.cuda.synchronize()
    assert torch.equal(split, unsplit)


def test_prequantized_query_contract_rejects_partial_or_ambiguous_inputs(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    query_fp4, query_scales = _quantize.quantize_query(
        inputs.query[:3], heads_kv=8
    )
    common = (
        inputs.key_pages_fp4,
        inputs.key_scales,
        inputs.value_pages_fp4,
        inputs.value_scales,
        inputs.page_table,
        inputs.seqused_fp4,
    )

    with pytest.raises(ValueError, match="either BF16 query"):
        fp4_decode(None, *common)
    with pytest.raises(ValueError, match="must be provided together"):
        fp4_decode(None, *common, query_fp4=query_fp4)
    with pytest.raises(ValueError, match="either BF16 query"):
        fp4_decode(
            inputs.query[:3],
            *common,
            query_fp4=query_fp4,
            query_scales=query_scales,
        )
    with pytest.raises(ValueError, match="applies only"):
        fp4_decode(
            None,
            *common,
            query_fp4=query_fp4,
            query_scales=query_scales,
            query_row_indices=inputs.query_row_indices,
        )
    with pytest.raises(ValueError, match="layout returned"):
        fp4_decode(
            None,
            *common,
            query_fp4=query_fp4,
            query_scales=query_scales.contiguous(),
        )


def test_prequantized_query_contract_rejects_bad_tensor_metadata(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    query_fp4, query_scales = _quantize.quantize_query(
        inputs.query[:3], heads_kv=8
    )
    common = {
        "key_pages_fp4": inputs.key_pages_fp4,
        "key_scales": inputs.key_scales,
        "value_pages_fp4": inputs.value_pages_fp4,
        "value_scales": inputs.value_scales,
        "fp4_page_table": inputs.page_table,
        "seqused_fp4": inputs.seqused_fp4,
    }
    with pytest.raises(ValueError, match="packed E2M1"):
        fp4_decode(
            **common,
            query_fp4=query_fp4.view(torch.uint8),
            query_scales=query_scales,
        )
    with pytest.raises(ValueError, match="shape"):
        fp4_decode(
            **common,
            query_fp4=query_fp4[:2],
            query_scales=query_scales,
        )
    with pytest.raises(ValueError, match="shape"):
        fp4_decode(
            **common,
            query_fp4=query_fp4,
            query_scales=query_scales[..., :2],
        )
    with pytest.raises(ValueError, match="layout returned|E4M3 scale-factor bytes"):
        fp4_decode(
            **common,
            query_fp4=query_fp4,
            query_scales=query_scales.to(torch.int16),
        )
    cpu_scales = torch.empty_strided(
        query_scales.shape,
        query_scales.stride(),
        dtype=query_scales.dtype,
        device="cpu",
    )
    with pytest.raises(ValueError, match="query_scales must be a CUDA tensor"):
        fp4_decode(
            **common,
            query_fp4=query_fp4,
            query_scales=cpu_scales,
        )
    wrong_rows = query_scales.as_strided(
        (*query_scales.shape[:-1], 2),
        query_scales.stride(),
    )
    with pytest.raises(ValueError, match="shape"):
        fp4_decode(
            **common,
            query_fp4=query_fp4,
            query_scales=wrong_rows,
        )
    wrong_heads = query_scales.as_strided(
        (*query_scales.shape[:-2], 7, query_scales.shape[-1]),
        query_scales.stride(),
    )
    with pytest.raises(ValueError, match="shape"):
        fp4_decode(
            **common,
            query_fp4=query_fp4,
            query_scales=wrong_heads,
        )
    with pytest.raises(ValueError, match="contiguous packed|heads_q"):
        fp4_decode(
            **common,
            query_fp4=query_fp4[:, :, :31],
            query_scales=query_scales,
        )


def test_trusted_metadata_matches_checked_path_exactly(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    compact_query = inputs.query.index_select(
        0, inputs.query_row_indices.long()
    )
    query_fp4, query_scales = _quantize.quantize_query(
        compact_query, heads_kv=8
    )
    kwargs = {
        "key_pages_fp4": inputs.key_pages_fp4,
        "key_scales": inputs.key_scales,
        "value_pages_fp4": inputs.value_pages_fp4,
        "value_scales": inputs.value_scales,
        "fp4_page_table": inputs.page_table,
        "seqused_fp4": inputs.seqused_fp4,
        "query_fp4": query_fp4,
        "query_scales": query_scales,
    }
    with torch.no_grad():
        checked = fp4_decode(**kwargs)
        trusted = fp4_decode(**kwargs, trusted_metadata=True)
    torch.cuda.synchronize()
    assert torch.equal(trusted, checked)


def test_repeated_identical_calls_are_bitwise_identical() -> None:
    """The decode kernel must not depend on how its warps happen to interleave.

    The shape matters. Every other case in this file launches a handful of
    CTAs over one or two page blocks, which keeps the producer and consumer
    warps close enough together that an unsynchronised buffer is very unlikely
    to be reused under a live reader. Reproducing a pipeline race needs a main
    loop deep enough to reach steady state (four page blocks) and enough
    resident CTAs to spread out memory latency (18 rows x 8 KV heads = 144
    CTAs, against 148 SMs), plus per-row residual lengths that differ so the
    CTAs do not run in lockstep.
    """
    torch.manual_seed(0x0FF1CE)
    device = torch.device("cuda")
    rows, pages_per_row, heads_q, heads_kv = 18, 4, 32, 8
    total_pages = rows * pages_per_row + 1

    query = (
        torch.randn(rows, heads_q, 128, dtype=torch.bfloat16, device=device) * 0.3
    )
    key_pages_bf16 = (
        torch.randn(
            total_pages, 128, heads_kv, 128, dtype=torch.bfloat16, device=device
        )
        * 0.3
    )
    value_pages_bf16 = torch.randn_like(key_pages_bf16) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(key_pages_bf16)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(value_pages_bf16)

    call = dict(
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        fp4_page_table=torch.arange(
            rows * pages_per_row, dtype=torch.int32, device=device
        ).view(rows, pages_per_row),
        seqused_fp4=torch.full(
            (rows,), pages_per_row * 128, dtype=torch.int32, device=device
        ),
        residual_key_pages_bf16=key_pages_bf16,
        residual_value_pages_bf16=value_pages_bf16,
        residual_page_ids=torch.full(
            (rows,), total_pages - 1, dtype=torch.int32, device=device
        ),
        seqused_residual=torch.tensor(
            [0 if i == 0 else 1 + (i * 37) % 127 for i in range(rows)],
            dtype=torch.int32,
            device=device,
        ),
    )

    with torch.no_grad():
        reference = fp4_decode(query, **call)
        for _ in range(7):
            assert torch.equal(fp4_decode(query, **call), reference)
    torch.cuda.synchronize()


def test_trusted_metadata_keeps_host_shape_checks(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    query_fp4, query_scales = _quantize.quantize_query(
        inputs.query[:3], heads_kv=8
    )
    with pytest.raises(ValueError, match="shape"):
        fp4_decode(
            key_pages_fp4=inputs.key_pages_fp4,
            key_scales=inputs.key_scales,
            value_pages_fp4=inputs.value_pages_fp4,
            value_scales=inputs.value_scales,
            fp4_page_table=inputs.page_table,
            seqused_fp4=inputs.seqused_fp4[:2],
            query_fp4=query_fp4,
            query_scales=query_scales,
            trusted_metadata=True,
        )
