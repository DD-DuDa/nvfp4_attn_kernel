"""Check that repeating the query rows leaves the decode output unchanged.

Scheme 2 shrinks the column range a softmax thread owns, but it gets there in
two steps and only the second one changes any arithmetic. The first step just
repeats each query row across its block of the M tile and teaches the epilogue
that head ``r`` now lives at row ``r * q_replicate``. Every row still runs the
full softmax over all 128 columns, on the same values, in the same order, so
the result must be **bitwise** identical to the unreplicated kernel. Anything
else means the replication or the epilogue's row selection is wrong, and this
catches it before the arithmetic changes make exact comparison impossible.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests/kernel_profile python \
      tests/kernel_profile/check_q_replicate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from nvfp4_decode_kernel import _decode as decode_mod  # noqa: E402
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel  # noqa: E402


def use_replication(enabled: bool) -> None:
    decode_mod.FP4DecodeKernel = (
        type("QReplicateKernel", (FP4DecodeKernel,), {"_enable_q_replicate": True})
        if enabled
        else FP4DecodeKernel
    )
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def run(case: bd.Case, device, enabled: bool, prequantized: bool) -> torch.Tensor:
    use_replication(enabled)
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    out = bd.make_fp4(inputs, hybrid=False, prequantized_query=prequantized)()
    return out.detach().float().cpu().clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--grid",
        type=str,
        default="1x1024,4x1024,2x4096",
        help="comma separated batchxseqlen specs",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="32x8,8x8,32x1",
        help="comma separated headsQxheadsKV specs",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    failures = 0
    for heads_spec in args.heads.split(","):
        heads_q, heads_kv = (int(x) for x in heads_spec.split("x"))
        for grid_spec in args.grid.split(","):
            batch, seqlen = (int(x) for x in grid_spec.split("x"))
            case = bd.Case(batch, seqlen, heads_q, heads_kv)
            for prequantized in (True, False):
                # The same seed makes both builds see identical inputs; the
                # tensors are re-randomized on every build_inputs call.
                torch.manual_seed(1234)
                base = run(case, device, False, prequantized)
                torch.manual_seed(1234)
                repl = run(case, device, True, prequantized)
                label = (
                    f"h{heads_q}x{heads_kv} b{batch} s{seqlen} "
                    f"{'fp4-q' if prequantized else 'bf16-q'}"
                )
                if torch.equal(base, repl):
                    print(f"{label:<34} bitwise identical")
                else:
                    diff = (base - repl).abs().max().item()
                    rel = diff / max(base.abs().max().item(), 1e-30)
                    print(f"{label:<34} DIFFERS  max_abs {diff:.3e}  rel {rel:.3e}")
                    failures += 1
    use_replication(False)
    print("\nall bitwise identical" if not failures else f"\n{failures} case(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
