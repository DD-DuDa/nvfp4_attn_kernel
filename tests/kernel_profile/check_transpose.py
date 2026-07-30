"""Compare the transposed softmax path against the untransposed one.

The untransposed path is covered by the correctness suite, so agreement on the
same inputs is the acceptance signal for the transposed layout. The suite's own
numerical cases cannot serve that role directly: every one of them carries a
BF16 residual tail, which the transposed path declines, so they all silently
fall back and would pass no matter what the transposed code did.

usage: PYTHONPATH=src python tests/kernel_profile/check_transpose.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4_decode_kernel import _decode
from nvfp4_decode_kernel import _quantize
from nvfp4_decode_kernel import fp4_decode


PAGE = 128
HEAD_DIM = 128


def build_case(batch: int, pages_per_row: int, q_heads: int, kv_heads: int, seed: int):
    torch.manual_seed(seed)
    total_pages = batch * pages_per_row
    query = torch.randn(batch, q_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    k_bf16 = (
        torch.randn(total_pages, PAGE, kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    v_bf16 = torch.randn_like(k_bf16) * 0.3
    k_fp4, k_sf = _quantize.quantize_key_pages(k_bf16)
    v_fp4, v_sf = _quantize.quantize_value_pages(v_bf16)
    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch, pages_per_row
    )
    seqused = torch.full((batch,), pages_per_row * PAGE, dtype=torch.int32, device="cuda")
    args = (query, k_fp4, k_sf, v_fp4, v_sf, page_table, seqused)
    return args, k_bf16, v_bf16


def run(args, transposed: bool):
    _decode._TRANSPOSE_S = transposed
    # Both caches: the split path keys into its own, and leaving it warm would
    # silently hand the second run the kernel the first run compiled.
    _decode._decode_compile_cache.clear()
    _decode._split_decode_compile_cache.clear()
    return fp4_decode(*args)


def bf16_reference(query, k_bf16, v_bf16, page_table, q_heads, kv_heads):
    batch, pages_per_row = page_table.shape
    pages = page_table.long()
    k = k_bf16[pages].reshape(batch, pages_per_row * PAGE, kv_heads, HEAD_DIM)
    v = v_bf16[pages].reshape(batch, pages_per_row * PAGE, kv_heads, HEAD_DIM)
    repeat = q_heads // kv_heads
    k = k.repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    out = F.scaled_dot_product_attention(
        query.unsqueeze(2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    )
    return out.squeeze(2)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main() -> None:
    cases = [
        ("1 row / 1 page  / gqa 32:8", 1, 1, 32, 8),
        ("1 row / 4 pages / gqa 32:8", 1, 4, 32, 8),
        ("3 rows / 2 pages/ gqa 32:8", 3, 2, 32, 8),
        ("1 row / 2 pages / mha 8:8", 1, 2, 8, 8),
        ("1 row / 2 pages / mqa 32:1", 1, 2, 32, 1),
        # Few rows over many pages is what drives the split heuristic, which is
        # the regime long context actually runs in.
        ("1 row / 32 pages/ gqa 32:8", 1, 32, 32, 8),
        ("1 row /128 pages/ gqa 32:8", 1, 128, 32, 8),
    ]
    failures = 0
    for name, batch, pages, q_heads, kv_heads in cases:
        args, k_bf16, v_bf16 = build_case(batch, pages, q_heads, kv_heads, seed=0xDEC0DE)
        plain = run(args, False)
        trans = run(args, True)
        ref = bf16_reference(args[0], k_bf16, v_bf16, args[5], q_heads, kv_heads)

        identical = torch.equal(plain, trans)
        max_abs = (plain.float() - trans.float()).abs().max().item()
        cos_pair = cosine(plain, trans)
        cos_plain = cosine(plain, ref)
        cos_trans = cosine(trans, ref)
        # The BF16 reference carries the FP4 quantization error, so the bar for
        # the transposed path is the untransposed path's own agreement with it,
        # not an absolute cosine.
        ok = cos_pair > 0.9999 and cos_trans >= cos_plain - 1e-4
        failures += not ok
        print(
            f"{name}: {'ok  ' if ok else 'FAIL'} "
            f"bitwise={identical} max|d|={max_abs:.3e} "
            f"cos(plain,trans)={cos_pair:.6f} "
            f"cos(plain,ref)={cos_plain:.6f} cos(trans,ref)={cos_trans:.6f}"
        )
    print("FAILURES:", failures)


if __name__ == "__main__":
    main()
