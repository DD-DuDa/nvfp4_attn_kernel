"""Characterise the shape of the decode kernel's run-to-run deviation.

The determinism mode of ``check_epilogue_equivalence.py`` only reports a max
absolute difference. That is enough to know something moves but not enough to
say what: a perturbed softmax denominator scales a whole output row by one
constant, whereas a corrupted K/V tile perturbs a subset of the 128 output
channels independently. This script separates those two cases, reports which
(row, head) pairs move at all, and can force every row's residual length to a
fixed value so the residual pipeline can be exercised without the residual
contributing anything numerically.
"""

from __future__ import annotations

import argparse

import torch

from check_epilogue_equivalence import build_inputs


PAGE = 128


def run_case(inputs, rows: int, case: str) -> torch.Tensor:
    from nvfp4_decode_kernel import fp4_decode

    base = dict(
        key_pages_fp4=inputs["key_pages_fp4"],
        key_scales=inputs["key_scales"],
        value_pages_fp4=inputs["value_pages_fp4"],
        value_scales=inputs["value_scales"],
        fp4_page_table=inputs["page_table"],
        seqused_fp4=inputs["seqused_fp4"],
    )
    residual = dict(
        residual_key_pages_bf16=inputs["key_pages_bf16"],
        residual_value_pages_bf16=inputs["value_pages_bf16"],
        residual_page_ids=inputs["residual_page_ids"],
        seqused_residual=inputs["seqused_residual"],
        has_bf16=inputs["seqused_residual"] > 0,
    )
    compact_query = inputs["query"][:rows].contiguous()
    with torch.no_grad():
        if case == "pure_fp4":
            out = fp4_decode(compact_query, **base)
        elif case == "residual":
            out = fp4_decode(compact_query, **base, **residual)
        else:
            raise ValueError(case)
    torch.cuda.synchronize()
    return out


def analyse(spec: str, repeat: int, case: str, residual_len: int | None) -> None:
    heads_q, heads_kv, rows, pages = (int(x) for x in spec.split("x"))
    inputs = build_inputs(heads_q, heads_kv, rows, pages)
    if residual_len is not None:
        inputs["seqused_residual"] = torch.full(
            (rows,), residual_len, dtype=torch.int32, device="cuda"
        )
    runs = [run_case(inputs, rows, case).float().cpu() for _ in range(repeat)]
    ref = runs[0]

    delta = torch.zeros_like(ref)
    for r in runs[1:]:
        delta = torch.maximum(delta, (r - ref).abs())
    moved = delta.amax(dim=-1) > 0
    n_moved = int(moved.sum())
    tag = f"{spec} {case} reslen={residual_len}"
    print(f"{tag}: {n_moved}/{moved.numel()} (row,head) pairs move, "
          f"max {delta.max().item():.4g}")
    if n_moved == 0:
        return

    idx = moved.nonzero()
    print(f"  rows touched: {sorted(set(int(i) for i in idx[:, 0]))}")
    print(f"  n heads touched: {len(set(int(i) for i in idx[:, 1]))}")
    for r_i, h_i in idx[:3].tolist():
        a = ref[r_i, h_i]
        worst = max(runs[1:], key=lambda t: (t[r_i, h_i] - a).abs().max())
        b = worst[r_i, h_i]
        big = a.abs() > 1e-3
        ratio = b[big] / a[big]
        n_diff = int((a != b).sum())
        print(
            f"  row {r_i} head {h_i}: {n_diff}/128 channels differ, "
            f"ratio spread [{ratio.min():.6f}, {ratio.max():.6f}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="32x8x18x4")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--case", default="residual")
    parser.add_argument("--residual-len", type=str, default="none")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    for reslen in args.residual_len.split(","):
        val = None if reslen == "none" else int(reslen)
        for case in args.case.split(","):
            for spec in args.spec.split(","):
                analyse(spec, args.repeat, case, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
