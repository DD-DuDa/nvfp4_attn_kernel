"""Is the transposed softmax still the right layout at large GQA groups?

The transposed path gives thread ``tidx`` kv position ``tidx`` of the block, so
its parallelism is pinned at the 128 kv positions of a tile no matter how many
query rows the tile carries. Query rows past the group count therefore become
serial work inside every thread instead of more threads. The untransposed path
spends threads on query rows instead, which should win once a tile carries
enough of them.

Runs both layouts over the same inputs, alternating so drift hits both sides.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from probe_graph_gate import LOCK_PATH, gpu_lock, graph_us  # noqa: E402


def measure(inputs, transpose: bool, warmup: int, iters: int, repeats: int) -> float:
    """Median graph-replay time with the layout forced one way."""
    from nvfp4_decode_kernel import _decode as decode_mod

    decode_mod._TRANSPOSE_S = transpose
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()

    run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
    run()
    torch.cuda.synchronize()
    with gpu_lock(LOCK_PATH):
        us, error = graph_us(run, warmup, iters, repeats)
    assert not error, error
    return us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-configs", default="32:8,64:8,128:8")
    parser.add_argument("--batches", default="16")
    parser.add_argument("--seqlens", default="16384,131072")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    configs = [tuple(int(x) for x in c.split(":")) for c in args.head_configs.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    seqlens = [int(x) for x in args.seqlens.split(",")]

    rows = []
    for heads_q, heads_kv in configs:
        for batch in batches:
            for seqlen in seqlens:
                case = bd.Case(batch, seqlen, heads_q, heads_kv)
                inputs = bd.build_inputs(case, device, quantize_chunk_pages=256)
                on, off = [], []
                for _ in range(args.rounds):
                    on.append(measure(inputs, True, args.warmup, args.iters, args.repeats))
                    off.append(measure(inputs, False, args.warmup, args.iters, args.repeats))
                t_on = statistics.median(on)
                t_off = statistics.median(off)
                row = {
                    "heads_q": heads_q,
                    "heads_kv": heads_kv,
                    "group": heads_q // heads_kv,
                    "batch": batch,
                    "seqlen": seqlen,
                    "transposed_us": t_on,
                    "untransposed_us": t_off,
                    "untransposed_over_transposed": t_off / t_on,
                }
                rows.append(row)
                print(
                    f"{heads_q:>3}:{heads_kv:<2} grp{row['group']:<3}"
                    f" b{batch:<3} s{seqlen:<7}"
                    f" transposed {t_on:8.1f} us   untransposed {t_off:8.1f} us"
                    f"   ratio {row['untransposed_over_transposed']:.3f}",
                    flush=True,
                )
                del inputs
                gc.collect()
                torch.cuda.empty_cache()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
