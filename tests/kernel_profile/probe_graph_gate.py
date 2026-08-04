"""Re-measure the acceptance grid under CUDA graph replay.

Eager-mode CUDA-event timing charges each kernel for its Python dispatch, and
the two kernels do not pay the same amount: FP4 pays more on the split path and
less on the single-kernel path, FA4 pays a roughly constant amount. That makes
the eager ratio a function of which internal path each side happened to pick,
which is not what the gate is trying to measure. Graph replay removes host
dispatch from both sides and is also how a serving stack runs decode.

Both sides take their split count as an argument, so a row's reported
``fp4_splits`` and ``fa4_splits`` are the counts that ran.

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


def run_case(
    case: bd.Case,
    device,
    iters: int,
    warmup: int,
    repeats: int,
    hybrid: bool = False,
    bf16_query: bool = False,
    fp4_splits: str | int = "auto",
) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    # A fused residual page has no pre-quantized-query benchmark, so it implies
    # the bf16-query entry point. The two are separable otherwise.
    splits = bd.resolve_fp4_splits(fp4_splits, case, device)
    fp4_run = bd.make_fp4(
        inputs,
        hybrid=hybrid,
        prequantized_query=not (hybrid or bf16_query),
        num_splits=splits,
    )

    record: dict[str, object] = {
        "case": case.label,
        "batch": case.batch,
        "seqlen": case.seqlen,
        "fp4_splits": splits,
    }
    with gpu_lock(LOCK_PATH):
        # fp4_decode refuses a split it cannot serve instead of downgrading to
        # one tile, so an unservable request is reported and skipped rather
        # than left to abort the sweep.
        try:
            fp4_run()
            torch.cuda.synchronize()
        except ValueError as error:
            record["fp4_unavailable"] = f"{type(error).__name__}: {error}"
            del inputs, fp4_run
            gc.collect()
            torch.cuda.empty_cache()
            return record
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
        "--splits",
        dest="fp4_splits",
        type=bd.parse_split_request,
        default="auto",
        help=(
            "split count handed to fp4_decode: 'auto' for what "
            "split_k_heuristic would pick, or a power of two; graph replay "
            "removes the host dispatch the shipped heuristic was calibrated "
            "against, so the two need not agree"
        ),
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="fuse a bf16 residual page, which the transposed softmax excludes",
    )
    parser.add_argument(
        "--bf16-query",
        action="store_true",
        help="quantize the query inside the call instead of passing it as fp4",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

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
            record = run_case(
                case,
                device,
                args.iters,
                args.warmup,
                args.repeats,
                args.hybrid,
                args.bf16_query,
                args.fp4_splits,
            )
            rows.append(record)
            if "fp4_unavailable" in record:
                print(
                    f"{case.label:<30} skipped at splits={record['fp4_splits']}:"
                    f" {record['fp4_unavailable']}",
                    flush=True,
                )
                continue
            print(
                f"{case.label:<30}"
                f" graph fp4 {record['fp4_graph_us']:8.1f} us"
                f" (s{record['fp4_splits']})"
                f"  fa4 {record['fa4_graph_us']:8.1f} us"
                f"  ratio {record['graph_ratio']:6.3f}"
                f"   | event ratio {record['event_ratio']:6.3f}",
                flush=True,
            )

    # Skipped cases stay in the written rows as evidence, but they have no
    # ratio to average.
    timed = [row for row in rows if "fp4_unavailable" not in row]
    print("\nper-seqlen geometric mean of fp4/fa4 (lower is better, gate is 0.5):")
    for seqlen in sorted({row["seqlen"] for row in timed}):
        group = [row for row in timed if row["seqlen"] == seqlen]
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
