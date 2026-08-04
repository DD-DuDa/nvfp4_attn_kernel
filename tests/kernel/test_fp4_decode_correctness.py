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
from nvfp4_decode_kernel import RESIDUAL_ROW_TILE, fp4_decode


FP4_MIN_COSINE = 0.99
FP4_MAX_ABS_ERROR = 5e-2
QUALITY_COSINE_TOLERANCE = 1e-2
_PAGE_SIZE = 128


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

    # The residual block is a pair of BF16 MMAs in the kernel, so nothing about
    # it is FP4: not its K and V, and not the query or the probabilities that
    # meet them. Only the FP4 pages see the quantized query. Random keys hide
    # the difference because the tail is then a small, unremarkable slice of the
    # sequence, but in a real decode it holds the most recent tokens and carries
    # much of the attention mass.
    positions = torch.arange(
        k_fp4.shape[1], device=q.device
    ).unsqueeze(0)
    valid = positions < seqused_k.unsqueeze(1)
    residual = positions >= (seqused_k - seqused_residual).unsqueeze(1)
    residual = (residual & valid)[:, None, None, :]

    scores = torch.where(
        residual,
        torch.einsum("bqhd,bkhd->bhqk", q.float(), k_fp4),
        torch.einsum("bqhd,bkhd->bhqk", q_fp4, k_fp4),
    ) * softmax_scale

    scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    return _online_softmax_pv(scores, v_fp4, residual)


def _online_softmax_pv(
    scores: torch.Tensor,
    v_fp4: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Accumulate PV the way the kernel does: one page at a time, backwards.

    What the reference point for a probability's FP4 scale is decides how much
    of it survives, and the kernel's is not the row's maximum. It walks pages
    from the end of the sequence towards the start, so a page is quantized
    against the largest score seen from that page onwards, and rescales the
    accumulator in fp32 whenever a later page raises it. A dominant key near
    the front therefore costs nothing to the pages behind it, while quantizing
    the whole row against one global maximum flushes them to zero. Attention
    over real activations is peaked enough for the difference to be visible;
    over random keys it is not.
    """
    page = _PAGE_SIZE
    batch, heads, rows, length = scores.shape
    output = torch.zeros(
        batch, heads, rows, v_fp4.shape[-1], device=scores.device
    )
    running_max = torch.full(
        (batch, heads, rows), -torch.inf, device=scores.device
    )
    running_sum = torch.zeros(batch, heads, rows, device=scores.device)

    for start in range(length - page, -1, -page):
        block = scores[..., start : start + page]
        block_max = torch.maximum(running_max, block.amax(dim=-1))
        # A block of nothing but masked positions leaves both maxima at -inf,
        # and their difference is a NaN rather than the zero it should be.
        rescale = torch.exp(running_max - block_max)
        rescale = torch.where(
            torch.isfinite(rescale), rescale, torch.zeros_like(rescale)
        )
        finite_max = torch.where(
            block_max > -torch.inf, block_max, torch.zeros_like(block_max)
        )
        p_exp = torch.exp(block - finite_max.unsqueeze(-1))

        running_sum = running_sum * rescale + p_exp.sum(dim=-1)
        output = output * rescale.unsqueeze(-1)
        p_block = torch.where(
            residual[..., start : start + page],
            p_exp.to(torch.bfloat16).float(),
            _nvfp4_round_trip(p_exp),
        )
        output = output + torch.einsum(
            "bhqk,bkhd->bhqd", p_block, v_fp4[:, start : start + page]
        )
        running_max = block_max

    return (output / running_sum.unsqueeze(-1)).permute(0, 2, 1, 3)


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


@pytest.mark.parametrize(
    "with_residual",
    [False, True],
    ids=["pure-fp4", "residual"],
)
def test_out_alone_keeps_split_k_and_writes_in_place(
    monkeypatch: pytest.MonkeyPatch,
    with_residual: bool,
) -> None:
    """An ``out`` without ``out_indices`` reaches split-K and writes in place.

    This is the path every vLLM decode step takes, bar the split count:
    ``impl.py`` hands over ``out=output[:rows]`` and no indices. The only
    other test that passes ``out`` also passes ``out_indices``, and that
    forces the single-tile path, so the split branch's write into a caller's
    buffer is otherwise never executed.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    from nvfp4_decode_kernel import _decode

    torch.manual_seed(0x0117)
    rows, heads_q, heads_kv, pages, head_dim = 1, 16, 1, 64, 128
    query = torch.randn(
        rows, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        pages,
        _PAGE_SIZE,
        heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    page_table = torch.arange(
        pages, device="cuda", dtype=torch.int32
    ).reshape(rows, pages)
    seqused_fp4 = torch.full(
        (rows,), pages * _PAGE_SIZE, device="cuda", dtype=torch.int32
    )

    # Eight splits over these 64 pages leaves every split a real main loop.
    num_splits = 8

    residual = {}
    if with_residual:
        residual = {
            "residual_key_pages_bf16": k_pages,
            "residual_value_pages_bf16": v_pages,
            "residual_page_ids": torch.zeros(
                rows, device="cuda", dtype=torch.int32
            ),
            "seqused_residual": torch.full(
                (rows,), 72, device="cuda", dtype=torch.int32
            ),
        }

    # Asking for a split is not evidence that the call took that branch, so
    # record what the dispatcher actually handed the split implementation.
    calls = []
    run_split = _decode.decode_fp4_split

    def recording_split(**kwargs):
        calls.append((kwargs["num_splits"], kwargs["out"]))
        return run_split(**kwargs)

    monkeypatch.setattr(_decode, "decode_fp4_split", recording_split)

    # vLLM's buffer is sized for the whole step, not for this batch alone.
    spare_rows = 3
    sentinel = torch.tensor(17.0, dtype=torch.bfloat16, device=query.device)
    out = torch.full(
        (rows + spare_rows, heads_q, head_dim),
        sentinel,
        dtype=torch.bfloat16,
        device=query.device,
    )

    decode_arguments = (
        key_pages_fp4,
        key_scales,
        value_pages_fp4,
        value_scales,
        page_table,
        seqused_fp4,
    )
    with torch.no_grad():
        allocated = fp4_decode(
            query, *decode_arguments, num_splits=num_splits, **residual
        )
        returned = fp4_decode(
            query,
            *decode_arguments,
            out=out,
            num_splits=num_splits,
            **residual,
        )
    torch.cuda.synchronize()

    assert [splits for splits, _ in calls] == [num_splits] * 2
    assert calls[0][1] is None
    assert calls[1][1] is out

    assert torch.equal(out[:rows], allocated)
    # The rows this batch does not own belong to other requests.
    assert torch.equal(out[rows:], torch.full_like(out[rows:], sentinel))
    # Both entries hand back the batch's own prefix of the caller's buffer,
    # not the spare rows they never wrote.
    assert returned.data_ptr() == out.data_ptr()
    assert tuple(returned.shape) == (rows, heads_q, head_dim)
    assert torch.equal(returned, allocated)


def test_num_splits_refuses_what_the_split_path_cannot_serve(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """A split that cannot be honoured raises instead of decoding with one.

    Every test that reaches split-K through this entry does so by asking for
    it, so a request quietly downgraded to a single tile would leave those
    tests passing while covering nothing.
    """
    inputs = contract_decode_inputs
    common = (
        inputs.key_pages_fp4,
        inputs.key_scales,
        inputs.value_pages_fp4,
        inputs.value_scales,
        inputs.page_table,
        inputs.seqused_fp4,
    )
    query = inputs.query[:3]

    for splits in (0, -2, 3, 6):
        with pytest.raises(ValueError, match="positive power of two"):
            fp4_decode(query, *common, num_splits=splits)

    with pytest.raises(ValueError, match="cannot be combined with out_indices"):
        fp4_decode(
            query,
            *common,
            out=torch.zeros_like(query),
            out_indices=torch.arange(3, device="cuda", dtype=torch.int32),
            num_splits=2,
        )

    # A residual reaches the split path only through the BF16 query, which is
    # what pads it to the residual MMA's row tile.
    query_fp4, query_scales = _quantize.quantize_query(query, heads_kv=8)
    with pytest.raises(ValueError, match="complete residual on the BF16 query"):
        fp4_decode(
            None,
            *common,
            query_fp4=query_fp4,
            query_scales=query_scales,
            residual_key_pages_bf16=inputs.key_pages_bf16,
            residual_value_pages_bf16=inputs.value_pages_bf16,
            residual_page_ids=inputs.residual_page_ids[:3],
            seqused_residual=inputs.seqused_residual[:3],
            num_splits=2,
        )


@pytest.mark.parametrize(
    "pages, num_splits",
    [(1, 1), (64, 8)],
    ids=["single-tile", "split-k"],
)
@pytest.mark.parametrize(
    "damage, complaint",
    [
        ("dtype", "out must be contiguous BF16"),
        ("short", "out needs at least 2 rows"),
        ("heads", "out must be contiguous BF16"),
    ],
)
def test_a_bad_out_is_refused_whether_or_not_split_k_runs(
    pages: int,
    num_splits: int,
    damage: str,
    complaint: str,
) -> None:
    """The ``out`` contract holds on the split path too.

    ``_kernel.py`` checks a split dispatch by calling ``decode_fp4`` with
    ``validate_only``, so those checks are all that stands between a caller's
    mistake and a silent write into the wrong buffer: split-K hands ``out``
    to the combine, which will happily truncate a short one or accept a
    wrongly shaped one.

    Naming a case "split-k" does not make it one; matching the complaint
    does. A ``num_splits`` the dispatcher will not serve raises about
    ``num_splits``, so a case that raises about ``out`` reached the branch it
    is named for.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    torch.manual_seed(0x0BAD)
    rows, heads_q, heads_kv, head_dim = 2, 16, 1, 128
    query = torch.randn(
        rows, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        pages,
        _PAGE_SIZE,
        heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    page_table = torch.arange(
        pages, device="cuda", dtype=torch.int32
    ).repeat(rows, 1)
    seqused_fp4 = torch.full(
        (rows,), pages * _PAGE_SIZE, device="cuda", dtype=torch.int32
    )

    shape = {
        "dtype": (rows, heads_q, head_dim),
        "short": (rows - 1, heads_q, head_dim),
        "heads": (rows, heads_q + 1, head_dim),
    }[damage]
    out = torch.zeros(
        shape,
        device="cuda",
        dtype=torch.float16 if damage == "dtype" else torch.bfloat16,
    )

    with pytest.raises(ValueError, match=complaint):
        fp4_decode(
            query,
            key_pages_fp4,
            key_scales,
            value_pages_fp4,
            value_scales,
            page_table,
            seqused_fp4,
            out=out,
            num_splits=num_splits,
        )


def test_a_single_tile_out_returns_only_the_rows_it_wrote() -> None:
    """The single-tile path returns the same prefix the split path does.

    The two disagreed: split-K returned ``out[:rows]`` while this one returned
    the caller's whole buffer, rows it never wrote included. ``out_indices``
    stays the exception, since a scatter can land anywhere in the buffer and
    only the whole of it describes where the results went.

    Passing no ``num_splits`` is what puts the call on the single-tile path.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    torch.manual_seed(0x1717)
    rows, heads_q, heads_kv, pages, head_dim = 2, 16, 1, 1, 128
    query = torch.randn(
        rows, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        pages,
        _PAGE_SIZE,
        heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    decode_arguments = (
        key_pages_fp4,
        key_scales,
        value_pages_fp4,
        value_scales,
        torch.zeros(rows, pages, device="cuda", dtype=torch.int32),
        torch.full(
            (rows,), pages * _PAGE_SIZE, device="cuda", dtype=torch.int32
        ),
    )

    spare_rows = 3
    sentinel = torch.tensor(17.0, dtype=torch.bfloat16, device=query.device)
    out = torch.full(
        (rows + spare_rows, heads_q, head_dim),
        sentinel,
        dtype=torch.bfloat16,
        device=query.device,
    )
    with torch.no_grad():
        allocated = fp4_decode(query, *decode_arguments)
        returned = fp4_decode(query, *decode_arguments, out=out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    assert tuple(returned.shape) == (rows, heads_q, head_dim)
    assert torch.equal(returned, allocated)
    assert torch.equal(out[rows:], torch.full_like(out[rows:], sentinel))


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


def _residual_call_kwargs(inputs: ContractDecodeInputs) -> dict:
    """The hybrid contract call, minus the query and the padded scratch."""
    return {
        "key_pages_fp4": inputs.key_pages_fp4,
        "key_scales": inputs.key_scales,
        "value_pages_fp4": inputs.value_pages_fp4,
        "value_scales": inputs.value_scales,
        "fp4_page_table": inputs.page_table,
        "seqused_fp4": inputs.seqused_fp4,
        "residual_key_pages_bf16": inputs.key_pages_bf16,
        "residual_value_pages_bf16": inputs.value_pages_bf16,
        "residual_page_ids": inputs.residual_page_ids,
        "seqused_residual": inputs.seqused_residual,
        "query_row_indices": inputs.query_row_indices,
    }


def test_query_padded_scratch_matches_internal_allocation(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """A caller-owned padded-query buffer must survive being reused.

    The quantizer writes only row 0 of each tile, so the rest has to stay
    zero for the residual MMA. Running twice through one buffer is what
    proves that: a call that dirtied the padding would move the second
    answer away from the first.
    """
    inputs = contract_decode_inputs
    rows = inputs.page_table.shape[0]
    call = _residual_call_kwargs(inputs)
    # Wider than this batch on purpose. Production sizes one buffer from
    # max_num_seqs and reuses it for whatever the step happens to bring.
    scratch = torch.zeros(
        rows + 5,
        RESIDUAL_ROW_TILE,
        inputs.query.shape[1],
        inputs.query.shape[2],
        dtype=torch.bfloat16,
        device=inputs.query.device,
    )
    allocated = fp4_decode(inputs.query, **call)
    first = fp4_decode(inputs.query, **call, query_padded_scratch=scratch)
    second = fp4_decode(inputs.query, **call, query_padded_scratch=scratch)
    assert torch.equal(first, allocated)
    assert torch.equal(second, allocated)
    assert not scratch[:, 1:].any()


def test_query_padded_scratch_is_validated(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    rows = inputs.page_table.shape[0]
    call = _residual_call_kwargs(inputs)
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]

    def buffer(*shape: int) -> torch.Tensor:
        return torch.zeros(
            *shape, dtype=torch.bfloat16, device=inputs.query.device
        )

    for wrong in (
        buffer(rows - 1, RESIDUAL_ROW_TILE, heads, head_dim),
        buffer(rows, RESIDUAL_ROW_TILE // 2, heads, head_dim),
        buffer(rows, RESIDUAL_ROW_TILE, heads - 1, head_dim),
    ):
        with pytest.raises(ValueError, match="query_padded_scratch"):
            fp4_decode(inputs.query, **call, query_padded_scratch=wrong)

    # The FP4-query path rejects a scratch before it looks at the query, so
    # placeholders are enough and a real quantize would only cost time.
    del call["query_row_indices"]
    placeholder = torch.empty(0, device=inputs.query.device)
    with pytest.raises(ValueError, match="query_padded_scratch applies only"):
        fp4_decode(
            **call,
            query_fp4=placeholder,
            query_scales=placeholder,
            query_padded_scratch=buffer(
                rows, RESIDUAL_ROW_TILE, heads, head_dim
            ),
        )


def _quantizer_scratch_shapes(
    rows: int, heads_q: int, head_dim: int
) -> dict[str, tuple[int, ...]]:
    """The storage shape of each quantizer output, keyed by argument name."""
    return {
        "query_fp4_scratch": (rows, 1, heads_q, head_dim // 2),
        "query_scales_scratch": (
            rows,
            1,
            heads_q,
            head_dim // 64,
            32,
            4,
            4,
        ),
    }


def _quantizer_scratch(
    rows: int,
    heads_q: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """A zeroed pair of caller-owned quantizer buffers, as decode kwargs.

    Zeroed and never refilled, which is the contract: the scale layout
    reserves slots the quantizer never writes, so their contents are whatever
    the buffer was born holding.
    """
    return {
        name: torch.zeros(shape, dtype=torch.uint8, device=device)
        for name, shape in _quantizer_scratch_shapes(
            rows, heads_q, head_dim
        ).items()
    }


def _assert_quantizes_identically(
    query: torch.Tensor,
    heads_kv: int,
    scratch: dict[str, torch.Tensor],
    row_indices: torch.Tensor | None = None,
    dirty_with: torch.Tensor | None = None,
) -> None:
    """Caller-owned buffers must hand back a fresh allocation's exact bytes.

    This is the level the "zeroed once" contract is observable at, and the
    only one. A decode cannot see a violation: the scale slots nothing writes
    belong to rows of the 128-row MMA tile that carry no query head, so the
    tensor core is fed them but the kernel discards their accumulator rows.
    The bytes still have to be right, because ``quantize_query`` hands them
    to whoever asked and the prequantized-query path takes them back.

    ``dirty_with`` is quantized through the buffers first, so the comparison
    starts from an earlier query's bytes. Without it a buffer left holding
    this very query would agree with a fresh allocation however the reuse
    contract were broken.
    """
    if dirty_with is not None:
        _quantize.quantize_query(
            dirty_with, row_indices=row_indices, heads_kv=heads_kv, **scratch
        )
    expected_fp4, expected_scales = _quantize.quantize_query(
        query, row_indices=row_indices, heads_kv=heads_kv
    )
    actual_fp4, actual_scales = _quantize.quantize_query(
        query, row_indices=row_indices, heads_kv=heads_kv, **scratch
    )
    assert torch.equal(
        actual_fp4.view(torch.uint8), expected_fp4.view(torch.uint8)
    )
    assert torch.equal(actual_scales, expected_scales)


def _shuffled(query: torch.Tensor) -> torch.Tensor:
    """A query unlike the given one in both row order and magnitude.

    Both matter: reordering changes which row's scales land where, and
    rescaling changes the E4M3 scales themselves, so nothing a call leaves
    behind can coincide with what the next one wants.
    """
    return (query.flip(0) * 2.5).contiguous()


def _residual_call_kwargs_for_rows(
    inputs: ContractDecodeInputs, rows: int
) -> dict:
    """The hybrid contract call narrowed to the first ``rows`` of the batch."""
    call = _residual_call_kwargs(inputs)
    call.update(
        fp4_page_table=inputs.page_table[:rows],
        seqused_fp4=inputs.seqused_fp4[:rows],
        residual_page_ids=inputs.residual_page_ids[:rows],
        seqused_residual=inputs.seqused_residual[:rows],
        query_row_indices=inputs.query_row_indices[:rows],
    )
    return call


def test_quantizer_scratch_matches_internal_allocation_after_reuse(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """Caller-owned quantizer buffers must survive having carried a query.

    A pristine buffer proves nothing here: it is zero everywhere the
    quantizer does not write, which is exactly what an internal allocation
    would have been. The regression only appears on a later call, once the
    slots the scale layout reserves but never writes are holding an earlier
    query's scales, so the buffer has to be dirty before the compared call.
    """
    inputs = contract_decode_inputs
    rows = inputs.page_table.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    call = _residual_call_kwargs(inputs)
    # Wider than this batch on purpose. Production sizes one buffer from
    # max_num_seqs and reuses it for whatever the step happens to bring.
    scratch = _quantizer_scratch(
        rows + 5, heads, head_dim, inputs.query.device
    )
    other_query = _shuffled(inputs.query)

    allocated = fp4_decode(inputs.query, **call)
    dirtying = fp4_decode(other_query, **call, **scratch)
    dirty_scales = scratch["query_scales_scratch"].clone()
    reused = fp4_decode(inputs.query, **call, **scratch)

    assert not torch.equal(dirtying, allocated)
    assert dirty_scales.any()
    assert torch.equal(reused, allocated)
    _assert_quantizes_identically(
        inputs.query,
        heads_kv,
        scratch,
        inputs.query_row_indices,
        dirty_with=other_query,
    )


def test_quantizer_scratch_survives_a_shrinking_batch(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """A buffer sized for the widest step has to be right for a narrow one.

    Rows are dim 0 of both buffers, so a smaller batch takes a prefix and
    leaves the rows above it holding the wider batch's quantized query.
    """
    inputs = contract_decode_inputs
    wide = inputs.page_table.shape[0]
    narrow = 1
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    scratch = _quantizer_scratch(wide, heads, head_dim, inputs.query.device)
    narrow_call = _residual_call_kwargs_for_rows(inputs, narrow)
    other_query = _shuffled(inputs.query)

    allocated = fp4_decode(inputs.query, **narrow_call)
    fp4_decode(
        other_query, **_residual_call_kwargs_for_rows(inputs, wide), **scratch
    )
    shrunk = fp4_decode(inputs.query, **narrow_call, **scratch)

    assert torch.equal(shrunk, allocated)
    _assert_quantizes_identically(
        inputs.query,
        heads_kv,
        scratch,
        inputs.query_row_indices[:narrow],
        dirty_with=other_query,
    )


@pytest.mark.parametrize(
    ("heads_q", "heads_kv"),
    # 16 rather than 32 query heads for MQA: pure FP4 with 32 heads over one
    # KV head does not decode reproducibly, and every assertion here is a
    # bitwise comparison of two decodes.
    [(8, 8), (32, 8), (16, 1)],
    ids=["mha", "gqa", "mqa"],
)
def test_quantizer_scratch_matches_internal_allocation_per_head_geometry(
    heads_q: int,
    heads_kv: int,
) -> None:
    """``heads_q // heads_kv`` chooses the scale slots, so vary that axis.

    Every geometry allocates its own buffer, which is what ``fp4_decode``
    asks for: the storage shape depends on ``heads_q`` and ``head_dim`` but
    not on ``heads_kv``, so one buffer carried across a change of
    ``heads_kv`` would pass every shape check while leaving the previous
    geometry's scales in slots this one needs zeroed.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 is required")

    torch.manual_seed(0x5CA1E + heads_kv)
    rows, pages, head_dim = 3, 2, 128
    query = torch.randn(
        rows, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    k_pages = torch.randn(
        rows * pages,
        _PAGE_SIZE,
        heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)
    call = {
        "key_pages_fp4": key_pages_fp4,
        "key_scales": key_scales,
        "value_pages_fp4": value_pages_fp4,
        "value_scales": value_scales,
        "fp4_page_table": torch.arange(
            rows * pages, device="cuda", dtype=torch.int32
        ).view(rows, pages),
        "seqused_fp4": torch.full(
            (rows,), pages * _PAGE_SIZE, device="cuda", dtype=torch.int32
        ),
    }
    scratch = _quantizer_scratch(rows, heads_q, head_dim, query.device)
    other_query = _shuffled(query)

    allocated = fp4_decode(query, **call)
    fp4_decode(other_query, **call, **scratch)
    reused = fp4_decode(query, **call, **scratch)

    assert torch.equal(reused, allocated)
    _assert_quantizes_identically(
        query, heads_kv, scratch, dirty_with=other_query
    )


def test_quantizer_scratch_is_sized_by_query_row_indices(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """With indices the required rows come from them, not from the query.

    The query is wider than the batch here, so a buffer sized to the index
    tensor is enough, and one sized below it has to be refused rather than
    quantized past its end.
    """
    inputs = contract_decode_inputs
    indexed_rows = inputs.query_row_indices.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    call = _residual_call_kwargs(inputs)
    assert indexed_rows < inputs.query.shape[0]

    exact = _quantizer_scratch(
        indexed_rows, heads, head_dim, inputs.query.device
    )
    allocated = fp4_decode(inputs.query, **call)
    assert torch.equal(fp4_decode(inputs.query, **call, **exact), allocated)
    _assert_quantizes_identically(
        inputs.query,
        heads_kv,
        exact,
        inputs.query_row_indices,
        dirty_with=_shuffled(inputs.query),
    )

    undersized = _quantizer_scratch(
        indexed_rows - 1, heads, head_dim, inputs.query.device
    )
    for name, wrong in undersized.items():
        with pytest.raises(ValueError, match=name):
            fp4_decode(inputs.query, **call, **{name: wrong})


@pytest.mark.parametrize(
    "supplied",
    [
        ("query_fp4_scratch",),
        ("query_scales_scratch",),
        ("query_fp4_scratch", "query_scales_scratch"),
    ],
    ids=["packed-only", "scales-only", "both"],
)
def test_either_quantizer_scratch_may_be_supplied_alone(
    contract_decode_inputs: ContractDecodeInputs,
    supplied: tuple[str, ...],
) -> None:
    """Neither buffer depends on the other; the missing one is allocated."""
    inputs = contract_decode_inputs
    rows = inputs.page_table.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    call = _residual_call_kwargs(inputs)
    scratch = _quantizer_scratch(
        rows + 2, heads, head_dim, inputs.query.device
    )
    chosen = {name: scratch[name] for name in supplied}

    allocated = fp4_decode(inputs.query, **call)
    assert torch.equal(fp4_decode(inputs.query, **call, **chosen), allocated)
    _assert_quantizes_identically(
        inputs.query,
        heads_kv,
        chosen,
        inputs.query_row_indices,
        dirty_with=_shuffled(inputs.query),
    )


def test_a_prefilled_query_scales_scratch_must_not_match_internal_allocation(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """The scale slots nothing writes are real. Do not make this test pass.

    Every other test of these buffers asks a scratch to reproduce a fresh
    allocation byte for byte, and all of them would pass just as happily if
    the unwritten slots did not exist, which would make them vacuous. This is
    the one that shows they do: the same query through a buffer that arrived
    holding a nonzero pattern instead of zeros must come out different.
    Zeroing the buffer here would silence the test and take with it the
    evidence that "zeroed once" is an obligation rather than a superstition.

    Only the scale bytes move. The decode output does not, because the slots
    in question belong to rows of the 128-row MMA tile that carry no query
    head, so the tensor core consumes them and the kernel throws their
    accumulator rows away.
    """
    inputs = contract_decode_inputs
    rows = inputs.query_row_indices.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    prefilled = torch.full(
        _quantizer_scratch_shapes(rows, heads, head_dim)[
            "query_scales_scratch"
        ],
        0xA5,
        dtype=torch.uint8,
        device=inputs.query.device,
    )

    _, expected_scales = _quantize.quantize_query(
        inputs.query,
        row_indices=inputs.query_row_indices,
        heads_kv=heads_kv,
    )
    _, actual_scales = _quantize.quantize_query(
        inputs.query,
        row_indices=inputs.query_row_indices,
        heads_kv=heads_kv,
        query_scales_scratch=prefilled,
    )

    assert not torch.equal(actual_scales, expected_scales)


def test_a_prefilled_query_fp4_scratch_still_matches_internal_allocation(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    """The packed query is the half that carries no zeroing obligation.

    Every byte of it is rewritten each call, which is why ``fp4_decode`` asks
    a caller to zero only the scales.
    """
    inputs = contract_decode_inputs
    rows = inputs.query_row_indices.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    heads_kv = inputs.key_pages_fp4.shape[2]
    prefilled = torch.full(
        _quantizer_scratch_shapes(rows, heads, head_dim)["query_fp4_scratch"],
        0xA5,
        dtype=torch.uint8,
        device=inputs.query.device,
    )

    _assert_quantizes_identically(
        inputs.query,
        heads_kv,
        {"query_fp4_scratch": prefilled},
        inputs.query_row_indices,
    )


def test_quantizer_scratch_is_validated(
    contract_decode_inputs: ContractDecodeInputs,
) -> None:
    inputs = contract_decode_inputs
    rows = inputs.page_table.shape[0]
    heads, head_dim = inputs.query.shape[1], inputs.query.shape[2]
    device = inputs.query.device
    call = _residual_call_kwargs(inputs)
    shapes = _quantizer_scratch_shapes(rows, heads, head_dim)

    def reject(name: str, wrong: torch.Tensor) -> None:
        with pytest.raises(ValueError, match=name):
            fp4_decode(inputs.query, **call, **{name: wrong})

    for name, shape in shapes.items():
        reject(
            name,
            torch.zeros(
                (rows - 1, *shape[1:]), dtype=torch.uint8, device=device
            ),
        )
        # Only the row count may differ from what this batch needs; every
        # inner extent belongs to the layout the decode kernel checks.
        for axis in range(1, len(shape)):
            grown = list(shape)
            grown[axis] += 1
            reject(name, torch.zeros(grown, dtype=torch.uint8, device=device))
        reject(name, torch.zeros(shape, dtype=torch.int8, device=device))

        # Contiguous, correctly shaped, and still refused: both buffers reach
        # kernels compiled with assumed_align=16, and only the start of an
        # allocation is guaranteed to be that aligned.
        offset = torch.zeros(
            (rows + 1, *shape[1:]), dtype=torch.uint8, device=device
        )[1:]
        assert offset.is_contiguous() and offset.storage_offset() != 0
        reject(name, offset)

        strided = torch.zeros(
            (*shape[:-1], shape[-1] * 2), dtype=torch.uint8, device=device
        )[..., ::2]
        assert not strided.is_contiguous()
        reject(name, strided)

        if torch.cuda.device_count() > 1:
            reject(
                name, torch.zeros(shape, dtype=torch.uint8, device="cuda:1")
            )

    # The FP4-query path rejects a scratch before it looks at the query, so
    # placeholders are enough and a real quantize would only cost time.
    del call["query_row_indices"]
    placeholder = torch.empty(0, device=device)
    for name, shape in shapes.items():
        with pytest.raises(
            ValueError, match=f"{name} applies only to the BF16 query path"
        ):
            fp4_decode(
                **call,
                query_fp4=placeholder,
                query_scales=placeholder,
                **{
                    name: torch.zeros(
                        shape, dtype=torch.uint8, device=device
                    )
                },
            )


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
