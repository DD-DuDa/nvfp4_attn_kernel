"""Probe: is the short-context split guard calibrated against a real cost?

``split_k_heuristic`` refuses to split unless each split gets 8 page blocks,
citing a 1K context measured 1.7x slower "purely from the combine launch". That
measurement was taken with eager CUDA-event timing, where the combine path adds
roughly 34 us of Python dispatch and only 3.7 us of GPU work. If the 1.7x was
the Python and not the launch, the guard is costing occupancy for nothing: at
1K a row has 8 pages, so the guard pins every short context to one split and
leaves 8 CTAs running on a 148-SM machine.

This forces each split count in turn and times it under graph replay, which
excludes host dispatch from both sides.

Usage:
  CUDA_VISIBLE_DEVICES=1 CUTE_DSL_CACHE_ENABLED=0 \
      PYTHONPATH=src:tests/kernel_profile python \
      tests/kernel_profile/probe_split_sweep.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from nvfp4_decode_kernel import _decode as decode_mod  # noqa: E402
from probe_graph_gate import LOCK_PATH, gpu_lock, graph_us  # noqa: E402

_FORCED_SPLITS = 1
_original_heuristic = decode_mod.split_k_heuristic


def _forced_heuristic(*args, **kwargs) -> int:
    return _FORCED_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--grid", type=str, default="1x1024,4x1024,16x1024,1x4096")
    parser.add_argument("--splits", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    global _FORCED_SPLITS
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    split_values = [int(value) for value in args.splits.split(",")]

    rows = []
    for point in args.grid.split(","):
        batch, seqlen = point.split("x")
        case = bd.Case(
            batch=int(batch),
            seqlen=int(seqlen),
            heads_q=args.heads_q,
            heads_kv=args.heads_kv,
        )
        inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
        pages_per_row = inputs.page_table.shape[1]
        record: dict[str, object] = {
            "case": case.label,
            "pages_per_row": pages_per_row,
            "chosen_by_heuristic": _original_heuristic(
                case.batch,
                args.heads_kv,
                pages_per_row,
                sms=torch.cuda.get_device_properties(device).multi_processor_count,
            ),
            "splits": {},
        }

        # Compilation dominates wall time with the DSL cache off, so it happens
        # outside the lock; only the timed replay needs exclusive use of the GPU.
        fa4_runs = [
            bd.make_fa4(inputs, splits)
            for splits in {1, bd.fa4_auto_splits(case, device)}
        ]
        for run in fa4_runs:
            run()

        fp4_runs: list[tuple[int, object]] = []
        for splits in split_values:
            if splits > pages_per_row:
                continue
            # `_split_decode_compile_cache` is keyed on num_splits, so distinct
            # split counts get distinct entries and the cache must not be cleared
            # between them: clearing would recompile every one on its timed pass.
            _FORCED_SPLITS = splits
            decode_mod.split_k_heuristic = _forced_heuristic
            run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
            try:
                run()
            except Exception as failure:  # a split count may be unsupported
                record["splits"][str(splits)] = {
                    "error": f"{type(failure).__name__}: {failure}"
                }
                continue
            finally:
                decode_mod.split_k_heuristic = _original_heuristic
            fp4_runs.append((splits, run))

        # The forced heuristic must stay installed while the timed calls run,
        # because each `run` re-enters the dispatch path on every invocation.
        with gpu_lock(LOCK_PATH):
            record["fa4_graph_us"] = min(
                graph_us(run, args.warmup, args.iters, args.repeats)[0]
                for run in fa4_runs
            )
            reference = None
            for splits, run in fp4_runs:
                _FORCED_SPLITS = splits
                decode_mod.split_k_heuristic = _forced_heuristic
                try:
                    value, error = graph_us(run, args.warmup, args.iters, args.repeats)
                    output = run().float()
                finally:
                    decode_mod.split_k_heuristic = _original_heuristic

                # Splitting must not change the answer; a fast wrong kernel has
                # bitten this repo before, so check against the unsplit result.
                if reference is None:
                    reference = output
                    cosine = 1.0
                else:
                    cosine = bd.cosine(output, reference)
                record["splits"][str(splits)] = {
                    "graph_us": value,
                    "error": error,
                    "cosine_vs_unsplit": cosine,
                }

        rows.append(record)
        print(
            f"== {case.label}  pages/row {pages_per_row}"
            f"  heuristic picks {record['chosen_by_heuristic']}"
            f"  fa4 {record['fa4_graph_us']:.1f} us",
            flush=True,
        )
        for splits, data in record["splits"].items():
            if "graph_us" not in data:
                print(f"   splits {splits:>3}  {data['error'][:70]}", flush=True)
                continue
            print(
                f"   splits {splits:>3}  {data['graph_us']:8.1f} us"
                f"   vs fa4 {data['graph_us'] / record['fa4_graph_us']:6.3f}x"
                f"   cosine {data['cosine_vs_unsplit']:.5f}",
                flush=True,
            )

        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
