"""Compare two builds of the decode kernel bit for bit.

The epilogue change is supposed to remove stores that the existing predicate
already discarded, so it should not move a single bit of output. That is a much
stronger claim than the tolerance assertions in ``tests/kernel``, and it is
cheap to check: run the same inputs against two checkouts and compare.

``PYTHONPATH`` selects which build runs, so a git worktree at the previous
commit is enough; nothing here imports from a specific version.

Usage:
  PYTHONPATH=<worktree>/src python tests/kernel_profile/check_epilogue_equivalence.py \\
      --out /tmp/epi_before.pt
  PYTHONPATH=src python tests/kernel_profile/check_epilogue_equivalence.py \\
      --out /tmp/epi_after.pt
  python tests/kernel_profile/check_epilogue_equivalence.py \\
      --compare /tmp/epi_before.pt /tmp/epi_after.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PAGE = 128
HEAD_DIM = 128
HEAD_CONFIGS = ((8, 8), (32, 8), (32, 1))


def build_inputs(heads_q: int, heads_kv: int, rows: int, pages_per_row: int):
    from nvfp4_decode_kernel import _quantize

    # Seeded per shape so the two runs see identical bytes without having to
    # ship the tensors between them.
    torch.manual_seed(0x5A1E + heads_q * 1000 + heads_kv * 10 + pages_per_row)
    device = torch.device("cuda")
    total_pages = rows * pages_per_row + 2

    query = torch.randn(
        rows + 2, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device
    ) * 0.3
    key_pages_bf16 = torch.randn(
        total_pages, PAGE, heads_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
    ) * 0.3
    value_pages_bf16 = torch.randn_like(key_pages_bf16) * 0.3
    key_pages_fp4, key_scales = _quantize.quantize_key_pages(key_pages_bf16)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(value_pages_bf16)

    page_table = torch.arange(
        rows * pages_per_row, dtype=torch.int32, device=device
    ).view(rows, pages_per_row)
    seqused_fp4 = torch.full(
        (rows,), pages_per_row * PAGE, dtype=torch.int32, device=device
    )
    residual_page_ids = torch.full(
        (rows,), total_pages - 1, dtype=torch.int32, device=device
    )
    # A zero-length row has to contribute nothing even next to nonzero rows, so
    # keep one in every residual case.
    seqused_residual = torch.tensor(
        [0 if i == 0 else 1 + (i * 37) % (PAGE - 1) for i in range(rows)],
        dtype=torch.int32,
        device=device,
    )
    query_row_indices = torch.tensor(
        [(i * 2 + 1) % (rows + 2) for i in range(rows)],
        dtype=torch.int32,
        device=device,
    )
    return {
        "query": query,
        "key_pages_bf16": key_pages_bf16,
        "value_pages_bf16": value_pages_bf16,
        "key_pages_fp4": key_pages_fp4,
        "key_scales": key_scales,
        "value_pages_fp4": value_pages_fp4,
        "value_scales": value_scales,
        "page_table": page_table,
        "seqused_fp4": seqused_fp4,
        "residual_page_ids": residual_page_ids,
        "seqused_residual": seqused_residual,
        "query_row_indices": query_row_indices,
    }


def run_shapes(inputs, rows: int) -> dict[str, torch.Tensor]:
    """The four call shapes the public entry supports."""
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

    results: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        results["pure_fp4"] = fp4_decode(compact_query, **base)
        results["residual"] = fp4_decode(compact_query, **base, **residual)
        results["indexed_rows"] = fp4_decode(
            inputs["query"],
            **base,
            **residual,
            query_row_indices=inputs["query_row_indices"],
        )
        scatter_out = torch.full_like(inputs["query"], 17.0)
        out_indices = torch.tensor(
            [(i * 3 + 2) % (rows + 2) for i in range(rows)],
            dtype=torch.int32,
            device=compact_query.device,
        )
        fp4_decode(
            inputs["query"],
            **base,
            **residual,
            query_row_indices=inputs["query_row_indices"],
            out=scatter_out,
            out_indices=out_indices,
        )
        results["direct_scatter"] = scatter_out
    torch.cuda.synchronize()
    return results


def collect() -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    # pages_per_row 2 stays on the unsplit path; 64 crosses into split-K, where
    # the epilogue runs under a different scheduler and q_stage.
    for heads_q, heads_kv in HEAD_CONFIGS:
        for rows, pages_per_row in ((3, 2), (1, 64), (5, 64)):
            inputs = build_inputs(heads_q, heads_kv, rows, pages_per_row)
            tag = f"h{heads_q}x{heads_kv}_r{rows}_p{pages_per_row}"
            for name, tensor in run_shapes(inputs, rows).items():
                outputs[f"{tag}_{name}"] = tensor.detach().cpu().clone()
            del inputs
            torch.cuda.empty_cache()
    return outputs


def compare(before_path: Path, after_path: Path) -> int:
    before = torch.load(before_path)
    after = torch.load(after_path)
    missing = set(before) ^ set(after)
    if missing:
        print(f"FAIL: case sets differ: {sorted(missing)}")
        return 1

    worst = 0.0
    failures = 0
    for name in sorted(before):
        lhs, rhs = before[name], after[name]
        identical = torch.equal(lhs, rhs)
        diff = (lhs.float() - rhs.float()).abs().max().item()
        worst = max(worst, diff)
        if not identical:
            failures += 1
            print(f"  DIFFER  max_abs_diff {diff:.6g}  {name}")
    print(f"{len(before)} cases, {failures} differ, worst max_abs_diff {worst:.6g}")
    if failures:
        print("FAIL: output is not bitwise identical")
        return 1
    print("PASS: bitwise identical on every case")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--compare", type=str, nargs=2, default=None)
    args = parser.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    if not args.out:
        parser.error("one of --out or --compare is required")
    torch.cuda.set_device(0)
    outputs = collect()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.out)
    print(f"wrote {len(outputs)} outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
