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
    # Both index maps must be injective. Duplicate out_indices would make two
    # decode rows race for one output row, which is nondeterministic and would
    # look exactly like a numerical regression.
    perm = torch.randperm(rows + 2, generator=torch.Generator().manual_seed(7))
    query_row_indices = perm[:rows].to(device=device, dtype=torch.int32)
    out_indices = perm.flip(0)[:rows].to(device=device, dtype=torch.int32)
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
        "out_indices": out_indices,
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
        fp4_decode(
            inputs["query"],
            **base,
            **residual,
            query_row_indices=inputs["query_row_indices"],
            out=scatter_out,
            out_indices=inputs["out_indices"],
        )
        results["direct_scatter"] = scatter_out
    torch.cuda.synchronize()
    return results


# pages_per_row 2 stays on the unsplit path; 64 crosses into split-K, where the
# epilogue runs under a different scheduler and q_stage. rows 64 is the unsplit
# path with a main loop long enough to matter, which is where the high-batch
# decode launches land.
# pages_per_row 2 stays on the unsplit path; 64 crosses into split-K, where the
# epilogue runs under a different scheduler and q_stage. rows 64 is the unsplit
# path with a main loop long enough to matter, which is where the high-batch
# decode launches land.
DEFAULT_SHAPES = "3x2,64x8,1x64,5x64"


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    out = []
    for item in spec.split(","):
        rows, pages = item.split("x")
        out.append((int(rows), int(pages)))
    return out


def collect(shapes: str, repeat: int) -> dict[str, dict[str, torch.Tensor]]:
    """First-run outputs plus each case's own run-to-run spread.

    The kernel is not deterministic at every shape, so a raw diff between two
    builds cannot be read on its own. Carrying each build's self-spread along
    with its output makes the comparison say what it should: whether the build
    difference is larger than the noise already present.
    """
    outputs: dict[str, torch.Tensor] = {}
    spread: dict[str, float] = {}
    for heads_q, heads_kv in HEAD_CONFIGS:
        for rows, pages_per_row in parse_shapes(shapes):
            inputs = build_inputs(heads_q, heads_kv, rows, pages_per_row)
            tag = f"h{heads_q}x{heads_kv}_r{rows}_p{pages_per_row}"
            reference = run_shapes(inputs, rows)
            for name, tensor in reference.items():
                outputs[f"{tag}_{name}"] = tensor.detach().cpu().clone()
                spread[f"{tag}_{name}"] = 0.0
            for _ in range(repeat - 1):
                for name, tensor in run_shapes(inputs, rows).items():
                    key = f"{tag}_{name}"
                    diff = (tensor.float() - reference[name].float()).abs().max().item()
                    spread[key] = max(spread[key], diff)
            del inputs, reference
            torch.cuda.empty_cache()
    return {"outputs": outputs, "spread": spread}


def compare(before_path: Path, after_path: Path) -> int:
    before = torch.load(before_path)
    after = torch.load(after_path)
    before_out, before_spread = before["outputs"], before["spread"]
    after_out, after_spread = after["outputs"], after["spread"]
    missing = set(before_out) ^ set(after_out)
    if missing:
        print(f"FAIL: case sets differ: {sorted(missing)}")
        return 1

    identical = exceeds = within = 0
    for name in sorted(before_out):
        lhs, rhs = before_out[name], after_out[name]
        diff = (lhs.float() - rhs.float()).abs().max().item()
        noise = max(before_spread.get(name, 0.0), after_spread.get(name, 0.0))
        if torch.equal(lhs, rhs):
            identical += 1
        elif diff <= noise:
            within += 1
            print(f"  within noise  diff {diff:.6g} <= self-spread {noise:.6g}  {name}")
        else:
            exceeds += 1
            print(f"  EXCEEDS NOISE diff {diff:.6g} >  self-spread {noise:.6g}  {name}")

    total = len(before_out)
    print(
        f"{total} cases: {identical} bitwise identical, "
        f"{within} differ but within the kernel's own run-to-run spread, "
        f"{exceeds} exceed it"
    )
    if exceeds:
        print("FAIL: a difference is larger than the existing nondeterminism")
        return 1
    if within:
        print("PASS: no difference exceeds the kernel's own nondeterminism")
        return 0
    print("PASS: bitwise identical on every case")
    return 0


def determinism(shapes: str, repeat: int) -> int:
    """Run the same inputs repeatedly in one process and diff the outputs.

    Staying in one process keeps the compiled kernel and the allocator state
    fixed, so a difference here is the kernel itself rather than anything about
    how the run was set up.
    """
    failures = 0
    for spec in shapes.split(","):
        heads_q, heads_kv, rows, pages = (int(x) for x in spec.split("x"))
        inputs = build_inputs(heads_q, heads_kv, rows, pages)
        ctas = rows * heads_kv
        reference = run_shapes(inputs, rows)
        worst: dict[str, float] = {}
        for _ in range(repeat - 1):
            for name, tensor in run_shapes(inputs, rows).items():
                diff = (tensor.float() - reference[name].float()).abs().max().item()
                worst[name] = max(worst.get(name, 0.0), diff)
        bad = {name: d for name, d in worst.items() if d > 0}
        label = f"h{heads_q}x{heads_kv} rows {rows} pages {pages} ({ctas} CTAs)"
        if bad:
            failures += 1
            print(f"  NONDETERMINISTIC  {label}")
            for name, diff in sorted(bad.items()):
                print(f"      {name:<16} max_abs_diff {diff:.6g}")
        else:
            print(f"  stable            {label}")
        del inputs, reference
        torch.cuda.empty_cache()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--compare", type=str, nargs=2, default=None)
    parser.add_argument(
        "--determinism",
        type=str,
        default=None,
        help="comma separated headsQxheadsKVxrowsxpages specs",
    )
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument(
        "--shapes",
        type=str,
        default=DEFAULT_SHAPES,
        help="comma separated rowsxpages specs for --out",
    )
    args = parser.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    if args.determinism:
        torch.cuda.set_device(0)
        return determinism(args.determinism, args.repeat)

    if not args.out:
        parser.error("one of --out or --compare is required")
    torch.cuda.set_device(0)
    record = collect(args.shapes, args.repeat)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, args.out)
    print(f"wrote {len(record['outputs'])} outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
