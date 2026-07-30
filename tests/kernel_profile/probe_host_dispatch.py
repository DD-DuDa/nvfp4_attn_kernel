"""Probe: what does the CPU spend per fp4_decode call, and does the GPU wait?

At low batch the CUDA-event time is several times the sum of the kernel times,
which means the GPU is idle waiting for the host to issue the next launch. This
attributes that host time to Python functions.

The measurement rests on one property: when the host is the bottleneck the
launch queue never backs up, so wall time with no synchronization is host time.
The script checks that property instead of assuming it, by reporting wall,
event and kernel time side by side.

Usage:
  flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
      PYTHONPATH=src:tests/kernel_profile python \
      tests/kernel_profile/probe_host_dispatch.py"
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pstats
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402


def profile_python(run, iters: int) -> list[tuple[str, float, float, int]]:
    """Return (function, cumulative_ms_per_call, self_ms_per_call, calls)."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(iters):
        run()
    profiler.disable()
    torch.cuda.synchronize()

    stats = pstats.Stats(profiler, stream=io.StringIO())
    rows = []
    for func, (_, nc, tt, ct, _) in stats.stats.items():
        filename, lineno, name = func
        short = f"{Path(filename).name}:{lineno}({name})"
        rows.append((short, ct * 1e3 / iters, tt * 1e3 / iters, nc))
    rows.sort(key=lambda row: -row[2])
    return rows


def graph_replay_us(run, warmup: int, iters: int) -> tuple[float, str]:
    """Event time per replay of a captured graph, or the reason capture failed.

    Capture answers whether the host cost is intrinsic or just eager-mode
    dispatch: a graph replays the same launches with no Python in the loop.
    """
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
    except Exception as error:  # capture is allowed to fail; that is a result
        return float("nan"), f"{type(error).__name__}: {error}"

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / iters, ""


def measure_variant(run, iters: int, warmup: int, want_python: bool) -> dict:
    wall_ms = bd.measure_wall_ms(run, iters, warmup)
    event_ms, _ = bd.measure_event_gpu_ms(run, iters, warmup)
    kernels = bd.measure_kernel_breakdown(run, iters, warmup)
    graph_us, graph_error = graph_replay_us(run, warmup, iters)
    record = {
        "wall_us": wall_ms * 1e3,
        "event_us": event_ms * 1e3,
        "kernel_us": sum(kernels.values()) * 1e3,
        "graph_us": graph_us,
        "graph_error": graph_error,
        "kernels": {name: ms * 1e3 for name, ms in kernels.items()},
    }
    if want_python:
        record["python"] = [
            {"fn": fn, "cum_us": cum * 1e3, "self_us": own * 1e3, "calls": calls}
            for fn, cum, own, calls in profile_python(run, iters)[:25]
        ]
    return record


def run_case(case: bd.Case, device: torch.device, iters: int, warmup: int) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    record: dict[str, object] = {"case": case.label}

    record["fp4"] = measure_variant(
        bd.make_fp4(inputs, hybrid=False, prequantized_query=True),
        iters,
        warmup,
        want_python=True,
    )
    # FA4 pays its own Python dispatch cost, and the acceptance gate compares
    # event times, so the two host costs only cancel if they are similar.
    record["fa4"] = measure_variant(
        bd.make_fa4(inputs, bd.fa4_auto_splits(case, device)),
        iters,
        warmup,
        want_python=False,
    )

    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--grid", type=str, default="1x16384,32x16384")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    rows = []
    for point in args.grid.split(","):
        batch, seqlen = point.split("x")
        case = bd.Case(
            batch=int(batch),
            seqlen=int(seqlen),
            heads_q=args.heads_q,
            heads_kv=args.heads_kv,
        )
        print(f"== {case.label}", flush=True)
        record = run_case(case, device, args.iters, args.warmup)
        rows.append(record)

        for variant in ("fp4", "fa4"):
            data = record[variant]
            graph = data["graph_us"]
            graph_text = (
                f"{graph:7.1f} us" if graph == graph else f"failed ({data['graph_error'][:40]})"
            )
            print(
                f"   {variant:<4} wall {data['wall_us']:7.1f}"
                f"   event {data['event_us']:7.1f}"
                f"   kernels {data['kernel_us']:7.1f}"
                f"   graph {graph_text}"
                f"   host gap {data['wall_us'] - data['kernel_us']:7.1f} us",
                flush=True,
            )
            for name, us in data["kernels"].items():
                print(f"          {us:7.2f} us  {name[:74]}", flush=True)

        print("   fp4 Python self time per call (top 15):", flush=True)
        for entry in record["fp4"]["python"][:15]:
            print(
                f"     {entry['self_us']:7.1f} us self"
                f"  {entry['cum_us']:8.1f} us cum"
                f"  {entry['calls'] // args.iters:4d}x  {entry['fn'][:60]}",
                flush=True,
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
