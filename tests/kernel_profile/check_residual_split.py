"""Check the residual against a forced single split once split-k carries it.

Split-k must count the one residual block in exactly one split. Counting it
twice, or dropping it, is invisible to a reference that carries fp4 noise of
its own, so the comparison is against the same case forced onto one split,
where the residual is unambiguous.

usage: PYTHONPATH=src python tests/kernel_profile/check_residual_split.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4_decode_kernel import _decode, _kernel, _quantize, fp4_decode


PAGE = 128
HEAD_DIM = 128


def build_case(batch: int, pages: int, q_heads: int, kv_heads: int, residual: int):
    torch.manual_seed(0xBEEF)
    total_pages = batch * pages
    query = (
        torch.randn(batch, q_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    )
    k_bf16 = (
        torch.randn(
            total_pages, PAGE, kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        * 0.3
    )
    v_bf16 = torch.randn_like(k_bf16) * 0.3
    k_fp4, k_sf = _quantize.quantize_key_pages(k_bf16)
    v_fp4, v_sf = _quantize.quantize_value_pages(v_bf16)
    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch, pages
    )
    # The residual tail sits past the fp4 pages, so the fp4 length covers every
    # page but the last, which the residual then supplies in bf16.
    seqused_fp4 = torch.full(
        (batch,), (pages - 1) * PAGE, dtype=torch.int32, device="cuda"
    )
    seqused_residual = torch.full((batch,), residual, dtype=torch.int32, device="cuda")
    # A one-element slice counts as contiguous whatever its stride, so the ids
    # need a fresh unit-stride buffer rather than .contiguous().
    residual_page_ids = torch.empty(batch, dtype=torch.int32, device="cuda")
    residual_page_ids.copy_(page_table[:, -1])
    return dict(
        query=query,
        key_pages_fp4=k_fp4,
        key_scales=k_sf,
        value_pages_fp4=v_fp4,
        value_scales=v_sf,
        fp4_page_table=page_table,
        seqused_fp4=seqused_fp4,
        residual_key_pages_bf16=k_bf16,
        residual_value_pages_bf16=v_bf16,
        residual_page_ids=residual_page_ids,
        seqused_residual=seqused_residual,
    )


def run(args, splits: int):
    """Run with the split count pinned, so the residual's share can be chosen."""
    original = _decode.split_k_heuristic
    _decode.split_k_heuristic = lambda *a, **k: splits
    try:
        return fp4_decode(**args)
    finally:
        _decode.split_k_heuristic = original


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main() -> None:
    # The residual has to carry real weight for a miscount to be visible, so
    # most of these are a page or two of fp4 against a full residual block, run
    # at split counts the heuristic would never pick for so little work.
    # Every distinct (heads, split count, residual) triple is another kernel to
    # compile, so these hold the shape fixed and vary only what the residual is
    # worth: all of the mass, half, a single token, none.
    cases = [
        ("res only,  0 fp4 pages", 1, 1, 32, 8, 128, 4),
        ("res half,  1 fp4 page ", 1, 2, 32, 8, 128, 4),
        ("res 1 tok, 1 fp4 page ", 1, 2, 32, 8, 1, 4),
        ("res none,  1 fp4 page ", 1, 2, 32, 8, 0, 4),
        ("res half,  batch 4    ", 4, 2, 32, 8, 128, 4),
        # Two fp4 pages over four splits is the first case where split-k really
        # reassociates, since each split quantizes P against its own row max.
        # The pair below is the same shape with and without a residual, so the
        # residual's contribution to the divergence is the difference.
        ("res third, 2 fp4 pages", 1, 3, 32, 8, 128, 4),
        ("res none,  2 fp4 pages", 1, 3, 32, 8, 0, 4),
    ]
    failures = 0
    for name, batch, pages, q_heads, kv_heads, residual, splits in cases:
        args = build_case(batch, pages, q_heads, kv_heads, residual)
        split_out = run(args, splits)
        single_out = run(args, 1)
        cos = cosine(split_out, single_out)
        max_abs = (split_out.float() - single_out.float()).abs().max().item()
        scale = single_out.float().abs().max().item()
        # Split-k reassociates the same sums, so the two differ a little even
        # with no residual at all; a residual counted twice or dropped moves
        # the answer by its share of the softmax mass, which is far larger.
        ok = cos > 0.999 and max_abs <= 5e-2 * max(scale, 1e-6)
        failures += not ok
        print(
            f"{name}: {'ok  ' if ok else 'FAIL'} "
            f"cos={cos:.6f} max|d|={max_abs:.3e} ref_max={scale:.3e}"
        )
    print("FAILURES:", failures)


if __name__ == "__main__":
    main()
