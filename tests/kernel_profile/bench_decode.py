"""Decode throughput comparison: official FA4 BF16 paged decode vs NVFP4 decode.

Both kernels attend to the same logical KV cache laid out in 128-token pages.
The FA4 baseline reads BF16 pages; the NVFP4 kernel reads E2M1 pages with E4M3
scale factors, optionally with the last page kept in BF16 (the "hybrid" tail
that serving code produces before a page is sealed and quantized).

Decode is bandwidth bound, so the report includes the KV bytes each variant has
to move and the achieved bandwidth, which is a fairer cross-precision metric
than raw latency.

Timing uses two clocks:

- ``gpu``: summed CUDA kernel time from the profiler, i.e. what the device
  actually spends. This is the number to use when comparing kernels.
- ``wall``: end-to-end Python latency of the public API. ``fp4_decode``
  validates several arguments with ``.item()``, which forces a device sync per
  call, so its wall time carries host overhead the FA4 path does not have.

Full sweep::

    PYTHONPATH=src python tests/kernel_profile/bench_decode.py --device 1

Holding KV traffic fixed while varying only the GQA ratio isolates how each
kernel spends its query axis, since the KV cache read is identical across the
four runs::

    for hq in 8 16 32 64; do
      PYTHONPATH=src python tests/kernel_profile/bench_decode.py --device 1 \
        --batches 16 --seqlens 16384 --heads-q $hq --heads-kv 8 \
        --variants fa4_bf16 fp4_pure
    done
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass

import torch

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.internal.interface import num_splits_heuristic
from nvfp4_decode_kernel import _quantize
from nvfp4_decode_kernel import fp4_decode
from nvfp4_decode_kernel._decode import _decode_compile_cache, _sfq_pack_cache


PAGE_SIZE = 128
HEAD_DIM = 128

# BF16 K plus BF16 V for one token of one KV head.
BF16_BYTES_PER_TOKEN_HEAD = 2 * HEAD_DIM * 2
# E2M1 payload plus one E4M3 scale per 16 elements, for K and V.
FP4_BYTES_PER_TOKEN_HEAD = 2 * (HEAD_DIM // 2 + HEAD_DIM // 16)


@dataclass(frozen=True)
class Case:
    batch: int
    seqlen: int
    heads_q: int
    heads_kv: int

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seqlen}"


@dataclass
class Inputs:
    case: Case
    query: torch.Tensor
    cu_seqlens_q: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    page_table: torch.Tensor
    seqused_k: torch.Tensor
    key_pages_fp4: torch.Tensor
    key_scales: torch.Tensor
    value_pages_fp4: torch.Tensor
    value_scales: torch.Tensor
    seqused_fp4_full: torch.Tensor
    seqused_fp4_hybrid: torch.Tensor
    seqused_residual: torch.Tensor
    residual_page_ids: torch.Tensor
    has_bf16: torch.Tensor
    softmax_scale: float


def build_inputs(case: Case, device: torch.device) -> Inputs:
    if case.seqlen % PAGE_SIZE:
        raise ValueError("seqlen must be a multiple of the 128-token page size")

    pages_per_row = case.seqlen // PAGE_SIZE
    num_pages = case.batch * pages_per_row

    query = torch.randn(
        case.batch,
        case.heads_q,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.3
    k_pages = torch.randn(
        num_pages,
        PAGE_SIZE,
        case.heads_kv,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.3
    v_pages = torch.randn_like(k_pages) * 0.3

    page_table = torch.arange(
        num_pages, device=device, dtype=torch.int32
    ).reshape(case.batch, pages_per_row)
    seqused_k = torch.full(
        (case.batch,), case.seqlen, device=device, dtype=torch.int32
    )

    key_pages_fp4, key_scales = _quantize.quantize_key_pages(k_pages)
    value_pages_fp4, value_scales = _quantize.quantize_value_pages(v_pages)

    # Hybrid split: every row keeps its final page in BF16, matching the
    # serving-time state where the newest page is not sealed yet.
    seqused_fp4_hybrid = seqused_k - PAGE_SIZE
    seqused_residual = torch.full(
        (case.batch,), PAGE_SIZE, device=device, dtype=torch.int32
    )
    # A size-1 slice stays "contiguous" while keeping the parent row stride, so
    # copy into a fresh buffer to guarantee the unit stride the kernel requires.
    residual_page_ids = torch.empty(
        case.batch, device=device, dtype=torch.int32
    )
    residual_page_ids.copy_(page_table[:, -1])

    return Inputs(
        case=case,
        query=query,
        cu_seqlens_q=torch.arange(
            case.batch + 1, device=device, dtype=torch.int32
        ),
        k_pages=k_pages,
        v_pages=v_pages,
        page_table=page_table,
        seqused_k=seqused_k,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        seqused_fp4_full=seqused_k.clone(),
        seqused_fp4_hybrid=seqused_fp4_hybrid,
        seqused_residual=seqused_residual,
        residual_page_ids=residual_page_ids,
        has_bf16=torch.ones(case.batch, device=device, dtype=torch.bool),
        softmax_scale=HEAD_DIM**-0.5,
    )


def fa4_auto_splits(case: Case, device: torch.device) -> int:
    """FA4's own split heuristic for this decode shape.

    Mirrors the call inside ``_flash_attn_fwd``: one m-block per (row, kv head)
    because pack-GQA folds the query heads into M. The heuristic can return 0
    when the grid is already oversubscribed, which the kernel treats as "no
    split", so clamp it to a valid split count.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    total_mblocks = case.batch * case.heads_kv
    num_n_blocks = case.seqlen // PAGE_SIZE
    return max(1, num_splits_heuristic(total_mblocks, sms, num_n_blocks, 128))


def make_fa4(inputs: Inputs, num_splits: int):
    """FA4 paged decode on the varlen path.

    The varlen entry point matters: for non-varlen inputs FA4 0.4.4 silently
    drops pack-GQA whenever ``num_splits != 1`` (see the "fix GQA + SplitKV +
    non-varlen" TODO in its interface), which costs a factor of
    qhead_per_kvhead and would make any split measurement meaningless.
    """
    case = inputs.case

    def run():
        return flash_attn_varlen_func(
            inputs.query,
            inputs.k_pages,
            inputs.v_pages,
            cu_seqlens_q=inputs.cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=inputs.seqused_k,
            max_seqlen_k=case.seqlen,
            page_table=inputs.page_table,
            softmax_scale=inputs.softmax_scale,
            causal=False,
            num_splits=num_splits,
        )

    return run


def make_fp4(inputs: Inputs, hybrid: bool):
    def run_pure():
        return fp4_decode(
            query=inputs.query,
            key_pages_fp4=inputs.key_pages_fp4,
            key_scales=inputs.key_scales,
            value_pages_fp4=inputs.value_pages_fp4,
            value_scales=inputs.value_scales,
            fp4_page_table=inputs.page_table,
            seqused_fp4=inputs.seqused_fp4_full,
            softmax_scale=inputs.softmax_scale,
        )

    def run_hybrid():
        return fp4_decode(
            query=inputs.query,
            key_pages_fp4=inputs.key_pages_fp4,
            key_scales=inputs.key_scales,
            value_pages_fp4=inputs.value_pages_fp4,
            value_scales=inputs.value_scales,
            fp4_page_table=inputs.page_table,
            seqused_fp4=inputs.seqused_fp4_hybrid,
            residual_key_pages_bf16=inputs.k_pages,
            residual_value_pages_bf16=inputs.v_pages,
            residual_page_ids=inputs.residual_page_ids,
            seqused_residual=inputs.seqused_residual,
            has_bf16=inputs.has_bf16,
            softmax_scale=inputs.softmax_scale,
        )

    return run_hybrid if hybrid else run_pure


def clear_fp4_compile_caches() -> None:
    """Force the next FP4 call in this process to compile under the profiler."""
    _decode_compile_cache.clear()
    _sfq_pack_cache.clear()


def kv_bytes(case: Case, variant: str) -> int:
    """KV bytes a variant must read from the cache for one decode step."""
    tokens = case.batch * case.seqlen
    per_head = case.heads_kv
    if variant.startswith("fa4"):
        return tokens * per_head * BF16_BYTES_PER_TOKEN_HEAD
    if variant == "fp4_hybrid":
        fp4_tokens = case.batch * (case.seqlen - PAGE_SIZE)
        bf16_tokens = case.batch * PAGE_SIZE
        return per_head * (
            fp4_tokens * FP4_BYTES_PER_TOKEN_HEAD
            + bf16_tokens * BF16_BYTES_PER_TOKEN_HEAD
        )
    return tokens * per_head * FP4_BYTES_PER_TOKEN_HEAD


def _output(result):
    return result[0] if isinstance(result, tuple) else result


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def measure_wall_ms(run, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def measure_gpu_ms(run, iters: int, warmup: int) -> tuple[float, dict[str, float]]:
    """Sum CUDA kernel time per iteration, plus a per-kernel breakdown."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()

    breakdown: dict[str, float] = {}
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        micros = event.self_device_time_total
        if micros <= 0:
            continue
        breakdown[event.key] = breakdown.get(event.key, 0.0) + micros / 1e3 / iters
    return sum(breakdown.values()), breakdown


VARIANTS = {
    "fa4_bf16": lambda inputs: make_fa4(inputs, num_splits=1),
    "fa4_bf16_split": lambda inputs: make_fa4(
        inputs, num_splits=fa4_auto_splits(inputs.case, inputs.query.device)
    ),
    "fp4_pure": lambda inputs: make_fp4(inputs, hybrid=False),
    "fp4_hybrid": lambda inputs: make_fp4(inputs, hybrid=True),
}


def run_case(
    case: Case,
    device: torch.device,
    variants: list[str],
    iters: int,
    warmup: int,
    breakdown: bool,
) -> list[dict]:
    inputs = build_inputs(case, device)
    reference = _output(make_fa4(inputs, num_splits=1)()).reshape(
        case.batch, case.heads_q, HEAD_DIM
    )

    rows = []
    for name in variants:
        run = VARIANTS[name](inputs)
        out = _output(run()).reshape(case.batch, case.heads_q, HEAD_DIM)
        gpu_ms, kernels = measure_gpu_ms(run, iters, warmup)
        wall_ms = measure_wall_ms(run, iters, warmup)
        moved = kv_bytes(case, name)
        rows.append(
            {
                "batch": case.batch,
                "seqlen": case.seqlen,
                "heads_q": case.heads_q,
                "heads_kv": case.heads_kv,
                "variant": name,
                "num_splits": (
                    fa4_auto_splits(case, device)
                    if name == "fa4_bf16_split"
                    else 1
                ),
                "gpu_ms": gpu_ms,
                "wall_ms": wall_ms,
                "kv_gib": moved / 2**30,
                "kv_gbps": moved / (gpu_ms * 1e-3) / 1e9,
                "cosine_vs_fa4": cosine(out, reference),
                "kernels": (
                    dict(sorted(kernels.items(), key=lambda kv: -kv[1]))
                    if breakdown
                    else {}
                ),
            }
        )

    del inputs, reference
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def format_table(rows: list[dict], variants: list[str]) -> str:
    by_case: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault((row["batch"], row["seqlen"]), {})[row["variant"]] = row

    header = f"{'batch':>5} {'seqlen':>7} {'spl':>4}"
    for name in variants:
        header += f" | {name + ' ms':>17} {'GB/s':>7}"
    header += f" | {'fp4 speedup':>11} {'cos':>6}"
    lines = [header, "-" * len(header)]

    for (batch, seqlen), entry in sorted(by_case.items()):
        splits = entry.get("fa4_bf16_split", {}).get("num_splits", 1)
        line = f"{batch:>5} {seqlen:>7} {splits:>4}"
        for name in variants:
            row = entry.get(name)
            if row is None:
                line += f" | {'-':>17} {'-':>7}"
                continue
            timing = f"{row['gpu_ms']:.3f} ({row['wall_ms']:.3f})"
            line += f" | {timing:>17} {row['kv_gbps']:>7.0f}"

        base = entry.get("fa4_bf16")
        best_fp4 = min(
            (entry[n] for n in ("fp4_pure", "fp4_hybrid") if n in entry),
            key=lambda r: r["gpu_ms"],
            default=None,
        )
        if base is not None and best_fp4 is not None:
            line += f" | {base['gpu_ms'] / best_fp4['gpu_ms']:>10.2f}x"
            line += f" {best_fp4['cosine_vs_fa4']:>6.4f}"
        else:
            line += f" | {'-':>11} {'-':>6}"
        lines.append(line)

    lines.append("")
    lines.append("spl is the split count FA4's heuristic picks for fa4_bf16_split.")
    lines.append("ms column is `gpu_kernel_ms (end_to_end_wall_ms)`.")
    lines.append("GB/s is KV cache bytes moved divided by gpu kernel time.")
    lines.append("speedup compares the faster fp4 variant against fa4_bf16 on gpu time.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    parser.add_argument(
        "--seqlens", type=int, nargs="+", default=[1024, 4096, 16384, 65536]
    )
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--max-kv-tokens",
        type=int,
        default=2_200_000,
        help="skip cases whose batch*seqlen exceeds this KV token budget",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument(
        "--clear-fp4-compile-cache",
        action="store_true",
        help="clear in-process FP4 caches before each case (required for IKET)",
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SystemExit(f"SM100 is required, found compute capability {capability}")

    print(f"device: {torch.cuda.get_device_name(device)} (cuda:{args.device})")
    print(f"heads_q={args.heads_q} heads_kv={args.heads_kv} head_dim={HEAD_DIM}")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()

    torch.manual_seed(0)
    rows: list[dict] = []
    for batch in args.batches:
        for seqlen in args.seqlens:
            if batch * seqlen > args.max_kv_tokens:
                continue
            case = Case(batch, seqlen, args.heads_q, args.heads_kv)
            print(f"running {case.label} ...", flush=True)
            if args.clear_fp4_compile_cache:
                clear_fp4_compile_caches()
            rows.extend(
                run_case(
                    case,
                    device,
                    args.variants,
                    args.iters,
                    args.warmup,
                    args.breakdown,
                )
            )

    print()
    print(format_table(rows, args.variants))

    if args.breakdown:
        print()
        print("per-kernel breakdown (ms/iter):")
        for row in rows:
            print(f"  {row['variant']} b{row['batch']} s{row['seqlen']}")
            for name, ms in row["kernels"].items():
                print(f"    {ms:8.4f}  {name[:96]}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
