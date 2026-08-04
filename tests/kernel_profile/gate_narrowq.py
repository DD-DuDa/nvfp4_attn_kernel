"""Acceptance gate for the narrow query tile.

The kernel reads exactly the bytes it must (2.42 GB at batch 16, seqlen 128k,
measured, matching the analytic minimum) and runs at 70 percent of HBM peak, so
latency here is a race to the memory roofline and nothing else. That makes DRAM
throughput the honest process metric: it does not move when the GPU clock
drifts, and it says how much of the remaining 30 percent a change actually
claimed. Latency is still reported, because it is what ships.

Three tiers:

  dev    batch 16 at two lengths, paired against a baseline binary. Fast enough
         to run between edits.
  gate   the stage gate. Adds the low batches where split-k still fires and the
         head configurations that the tile change could quietly break.
  full   everything, against trtllm-gen and FA4, for the report.

Usage:
  PYTHONPATH=src:tests/kernel_profile python tests/kernel_profile/gate_narrowq.py \
      --tier gate --out /tmp/narrowq/gate.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from probe_graph_gate import LOCK_PATH, gpu_lock, graph_us  # noqa: E402

# (batch, seqlen, heads_q, heads_kv)
TIERS: dict[str, list[tuple[int, int, int, int]]] = {
    "dev": [
        (16, 16384, 32, 8),
        (16, 131072, 32, 8),
    ],
    "gate": [
        (1, 16384, 32, 8),
        (1, 131072, 32, 8),
        (4, 16384, 32, 8),
        (4, 131072, 32, 8),
        (16, 16384, 32, 8),
        (16, 131072, 32, 8),
        (64, 16384, 32, 8),
        (16, 131072, 8, 8),
        (16, 131072, 32, 1),
    ],
    "full": [
        (b, s, 32, 8)
        for b in (1, 2, 4, 8, 16, 32, 64, 128)
        for s in (1024, 4096, 16384, 65536, 131072)
    ]
    + [(16, 131072, 8, 8), (16, 131072, 32, 1)],
}

NCU_METRICS = (
    "dram__bytes_read.sum,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,"
    "gpu__time_duration.sum"
)
_NCU_ROW = re.compile(r"^\s{4}(\S+)\s+\S*\s*([0-9][0-9,.]*)\s*$", re.M)


def ncu_profile(batch: int, seqlen: int, heads_q: int, heads_kv: int,
                device: int) -> dict[str, float]:
    """Profile one decode launch out of process.

    Nsight serializes and replays the kernel, so this cannot share a process
    with the timing loop without poisoning it.
    """
    runner = Path(tempfile.gettempdir()) / "gate_narrowq_one.py"
    runner.write_text(
        "import contextlib, io, os, sys\n"
        "os.environ.setdefault('CUTE_DSL_CACHE_ENABLED', '0')\n"
        f"sys.path[:0] = {[str(Path.cwd() / 'src'), str(Path(__file__).resolve().parent)]!r}\n"
        "import torch, bench_decode as bd\n"
        "torch.cuda.set_device(0)\n"
        f"case = bd.Case({batch}, {seqlen}, {heads_q}, {heads_kv})\n"
        "inp = bd.build_inputs(case, torch.device('cuda:0'), quantize_chunk_pages=256)\n"
        "run = bd.make_fp4(inp, hybrid=False, prequantized_query=True)\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    [run() for _ in range(3)]\n"
        "    torch.cuda.synchronize()\n"
        "torch.cuda.profiler.start(); run(); torch.cuda.synchronize(); torch.cuda.profiler.stop()\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    proc = subprocess.run(
        ["ncu", "--profile-from-start", "off", "--target-processes", "all",
         "-k", "regex:fp4", "--metrics", NCU_METRICS, sys.executable, str(runner)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    out = {}
    for name, value in _NCU_ROW.findall(proc.stdout):
        if name.startswith(("dram__", "sm__", "gpu__")):
            out[name] = float(value.replace(",", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="dev")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-ncu", action="store_true")
    ap.add_argument("--baseline", type=Path,
                    help="A previous --out file; report deltas against it.")
    args = ap.parse_args()

    torch.cuda.set_device(torch.device(f"cuda:{args.device}"))
    base = {}
    if args.baseline and args.baseline.exists():
        base = {tuple(r["case"]): r for r in json.loads(args.baseline.read_text())["rows"]}

    rows = []
    for batch, seqlen, heads_q, heads_kv in TIERS[args.tier]:
        case = bd.Case(batch, seqlen, heads_q, heads_kv)
        inputs = bd.build_inputs(case, torch.device(f"cuda:{args.device}"),
                                 quantize_chunk_pages=256)
        run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        with contextlib.redirect_stdout(io.StringIO()):
            run()
            torch.cuda.synchronize()
        with gpu_lock(LOCK_PATH):
            samples = []
            for _ in range(3):
                us, err = graph_us(run, 30, 20, 5)
                assert not err, f"{case}: {err}"
                samples.append(us)
        row = {
            "case": [batch, seqlen, heads_q, heads_kv],
            "us": statistics.median(samples),
            "us_spread": (max(samples) - min(samples)) / statistics.median(samples),
        }
        if not args.no_ncu:
            row.update(ncu_profile(batch, seqlen, heads_q, heads_kv, args.device))
        rows.append(row)
        del inputs, run
        torch.cuda.empty_cache()

    clock = subprocess.run(
        ["nvidia-smi", "-i", str(args.device), "--query-gpu=clocks.sm",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"\ntier {args.tier}, sm clock {clock}")
    header = f"{'case':>22} {'us':>9} {'dram%':>7} {'tc%':>6}"
    if base:
        header += f" {'vs base':>8}"
    print(header)
    for row in rows:
        b, s, hq, hkv = row["case"]
        line = (f"{f'b{b} s{s//1024}k {hq}:{hkv}':>22} {row['us']:9.1f}"
                f" {row.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0):7.1f}"
                f" {row.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active', 0):6.1f}")
        prev = base.get(tuple(row["case"]))
        if prev:
            line += f" {row['us'] / prev['us']:7.3f}x"
        print(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tier": args.tier, "clock": clock, "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
