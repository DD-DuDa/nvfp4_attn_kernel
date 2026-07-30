"""Re-measure the acceptance grid under CUDA graph replay.

Eager-mode CUDA-event timing charges each kernel for its Python dispatch, and
the two kernels do not pay the same amount: FP4 pays more on the split path and
less on the single-kernel path, FA4 pays a roughly constant amount. That makes
the eager ratio a function of which internal path each side happened to pick,
which is not what the gate is trying to measure. Graph replay removes host
dispatch from both sides and is also how a serving stack runs decode.

Timing sections take an advisory file lock so this can share the GPU with other
work; input construction and teardown stay outside it.

Usage:
  CUDA_VISIBLE_DEVICES=1 CUTE_DSL_CACHE_ENABLED=0 \
      PYTHONPATH=src:tests/kernel_profile python \
      tests/kernel_profile/probe_graph_gate.py --out docs/perf/phase7/graph-gate.json
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402

LOCK_PATH = "/tmp/nvfp4_gpu1.lock"


def install_relaxed_heuristic(min_pages_per_split: int) -> None:
    """Patch split_k_heuristic with a different guard, leaving the rest intact.

    Only the ``min_pages_per_split`` constant changes. Everything else, in
    particular the ``target`` occupancy computation, is reproduced exactly so
    the comparison isolates the guard.
    """
    import math

    from nvfp4_decode_kernel import _decode as decode_mod

    def patched(rows, heads_kv, max_pages_per_row, *, sms):
        if rows < 1 or heads_kv < 1 or max_pages_per_row < 2 or sms < 1:
            return 1
        unsplit_ctas = rows * heads_kv
        if unsplit_ctas >= sms:
            return 1
        target = max(2, math.ceil(sms / unsplit_ctas))
        for splits in (32, 16, 8, 4, 2):
            if splits <= target and splits * min_pages_per_split <= max_pages_per_row:
                return splits
        return 1

    decode_mod.split_k_heuristic = patched


@contextlib.contextmanager
def gpu_lock(path: str):
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def graph_us(run, warmup: int, iters: int, repeats: int) -> tuple[float, str]:
    """Median per-replay time of a captured graph, or the capture failure."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
    except Exception as error:
        return float("nan"), f"{type(error).__name__}: {error}"

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    samples.sort()
    return samples[len(samples) // 2], ""


def run_case(case: bd.Case, device, iters: int, warmup: int, repeats: int) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    fp4_run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)

    record: dict[str, object] = {
        "case": case.label,
        "batch": case.batch,
        "seqlen": case.seqlen,
    }
    with gpu_lock(LOCK_PATH):
        record["fp4_graph_us"], record["fp4_error"] = graph_us(
            fp4_run, warmup, iters, repeats
        )
        record["fp4_event_us"] = (
            bd.measure_event_gpu_ms(fp4_run, iters, warmup)[0] * 1e3
        )
        # The D0 baseline is the better of no-split and the FA4 heuristic.
        best = math.inf
        for splits in {1, bd.fa4_auto_splits(case, device)}:
            value, error = graph_us(
                bd.make_fa4(inputs, splits), warmup, iters, repeats
            )
            if error:
                record.setdefault("fa4_errors", []).append(f"splits={splits}: {error}")
                continue
            if value < best:
                best = value
                record["fa4_splits"] = splits
        record["fa4_graph_us"] = best if best < math.inf else float("nan")
        record["fa4_event_us"] = min(
            bd.measure_event_gpu_ms(bd.make_fa4(inputs, splits), iters, warmup)[0] * 1e3
            for splits in {1, bd.fa4_auto_splits(case, device)}
        )

    record["graph_ratio"] = record["fp4_graph_us"] / record["fa4_graph_us"]
    record["event_ratio"] = record["fp4_event_us"] / record["fa4_event_us"]

    del inputs, fp4_run
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--seqlens", type=str, default="1024,4096,16384,65536")
    parser.add_argument("--batches", type=str, default="1,4,16,64")
    parser.add_argument("--max-kv-tokens", type=int, default=8_400_000)
    parser.add_argument(
        "--min-pages-per-split",
        type=int,
        default=None,
        help=(
            "override the guard in split_k_heuristic; the shipped value is 8, "
            "which was calibrated against eager-mode host overhead"
        ),
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    if args.min_pages_per_split is not None:
        install_relaxed_heuristic(args.min_pages_per_split)

    rows = []
    for seqlen in (int(value) for value in args.seqlens.split(",")):
        for batch in (int(value) for value in args.batches.split(",")):
            if batch * seqlen > args.max_kv_tokens:
                continue
            case = bd.Case(
                batch=batch,
                seqlen=seqlen,
                heads_q=args.heads_q,
                heads_kv=args.heads_kv,
            )
            record = run_case(case, device, args.iters, args.warmup, args.repeats)
            rows.append(record)
            print(
                f"{case.label:<30}"
                f" graph fp4 {record['fp4_graph_us']:8.1f} us"
                f"  fa4 {record['fa4_graph_us']:8.1f} us"
                f"  ratio {record['graph_ratio']:6.3f}"
                f"   | event ratio {record['event_ratio']:6.3f}",
                flush=True,
            )

    print("\nper-seqlen geometric mean of fp4/fa4 (lower is better, gate is 0.5):")
    for seqlen in sorted({row["seqlen"] for row in rows}):
        group = [row for row in rows if row["seqlen"] == seqlen]
        graph_gm = statistics.geometric_mean([row["graph_ratio"] for row in group])
        event_gm = statistics.geometric_mean([row["event_ratio"] for row in group])
        print(
            f"  s{seqlen:<7} graph {graph_gm:6.3f}   event {event_gm:6.3f}"
            f"   ({len(group)} batches)"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
