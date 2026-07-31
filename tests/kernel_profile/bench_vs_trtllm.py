"""Compare this repository's FP4 decode against the trtllm-gen NVFP4 decode.

Both sides are handed an already-quantized paged NVFP4 K/V cache and an
already-quantized query, so neither pays for quantization. The remaining
asymmetries are inherent to the two kernels and are reported alongside the
timings:

- trtllm-gen keeps the query in FP8 E4M3 and only emits an FP8 output, while
  this kernel keeps the query in FP4 E2M1 and emits BF16.
- trtllm-gen wants a plain ``[pages, heads_kv, page, dim/16]`` E4M3 block-scale
  array; this kernel wants the tcgen05-native swizzled scale layout. Each side
  is therefore quantized from the same BF16 pages with its own quantizer.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests/kernel_profile \
      python tests/kernel_profile/bench_vs_trtllm.py --seqlens 16384
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from probe_graph_gate import gpu_lock, graph_us, LOCK_PATH  # noqa: E402

FP8 = torch.float8_e4m3fn
# trtllm-gen scratch for softmax stats; vLLM sizes this at 128 MB.
WORKSPACE_BYTES = 128 * 1024 * 1024
# NVFP4 amax maps to E2M1 max (6) times E4M3 max (448).
NVFP4_GLOBAL_RANGE = 6.0 * 448.0


def kv_bytes(case: bd.Case) -> float:
    """Compulsory NVFP4 K+V traffic for one decode, identical for both kernels.

    Half a byte per element of packed E2M1 plus one E4M3 scale per 16 elements,
    on each of K and V.
    """
    elements = case.batch * case.seqlen * case.heads_kv * bd.HEAD_DIM
    return elements * 2 * (0.5 + 1.0 / 16.0)


def nvfp4_quantize_hnd(
    pages: torch.Tensor, *, heads_kv: int, page_size: int, chunk_pages: int
):
    """Quantize BF16 pages into the trtllm NVFP4 KV format.

    ``pages`` is bench_decode's ``[pages, 128, heads, dim]`` layout; the result
    is HND ``[pages, heads, page_size, ...]``, re-chunked to ``page_size``. Only
    the page granularity changes, the token order is identical.

    ``global_scale`` is a descale in this API: a value is reconstructed as
    ``e2m1 * e4m3_block_scale * global_scale``. Sizing it as ``amax / (6 * 448)``
    is what keeps the per-16 block scales inside the E4M3 range.

    The HND transpose and the quantizer both materialize a full-size copy, so
    this runs page-chunked to keep the peak well under the KV cache itself.
    """
    from flashinfer.fp4_quantization import nvfp4_kv_quantize

    dim = pages.shape[-1]
    tokens = pages.reshape(-1, page_size, heads_kv, dim)
    num_pages = tokens.shape[0]
    global_scale = (pages.abs().max().float() / NVFP4_GLOBAL_RANGE).reshape(1)

    data = torch.empty(
        num_pages, heads_kv, page_size, dim // 2, dtype=torch.uint8, device=pages.device
    )
    scales = torch.empty(
        num_pages, heads_kv, page_size, dim // 16, dtype=FP8, device=pages.device
    )
    for start in range(0, num_pages, chunk_pages):
        stop = min(start + chunk_pages, num_pages)
        block = tokens[start:stop].permute(0, 2, 1, 3).contiguous()
        block_data, block_scales = nvfp4_kv_quantize(
            block.reshape(-1, dim), global_scale
        )
        data[start:stop].copy_(block_data.view(-1, heads_kv, page_size, dim // 2))
        scales[start:stop].copy_(
            block_scales.view(-1, heads_kv, page_size, dim // 16).view(FP8)
        )
    return data, scales, global_scale.item()


def make_trtllm(inputs: bd.Inputs, *, page_size: int, chunk_pages: int = 512):
    """Build a zero-argument callable that runs one trtllm-gen NVFP4 decode."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    case = inputs.case
    device = inputs.query.device

    quantize_kwargs = dict(
        heads_kv=case.heads_kv, page_size=page_size, chunk_pages=chunk_pages
    )
    k_data, k_sf, k_descale = nvfp4_quantize_hnd(inputs.k_pages, **quantize_kwargs)
    v_data, v_sf, v_descale = nvfp4_quantize_hnd(inputs.v_pages, **quantize_kwargs)

    q_amax = inputs.query.abs().max().float().item()
    q_descale = q_amax / 448.0
    query_fp8 = (inputs.query.float() / q_descale).to(FP8)

    pages_per_row = case.seqlen // page_size
    block_tables = (
        torch.arange(case.batch * pages_per_row, device=device, dtype=torch.int32)
        .reshape(case.batch, pages_per_row)
        .contiguous()
    )
    seq_lens = torch.full(
        (case.batch,), case.seqlen, device=device, dtype=torch.int32
    )
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    out = torch.empty(case.batch, case.heads_q, bd.HEAD_DIM, dtype=FP8, device=device)

    # The NVFP4 kernel only emits E4M3, so the result has to be pre-divided by
    # an output scale. Attention output is a convex combination of V rows, so
    # |out| <= amax(V) and this choice never saturates.
    out_descale = inputs.v_pages.abs().max().float().item() / 448.0
    bmm1_scale = q_descale * k_descale * inputs.softmax_scale
    bmm2_scale = v_descale / out_descale

    def run():
        return trtllm_batch_decode_with_kv_cache(
            query=query_fp8,
            kv_cache=(k_data, v_data),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=case.seqlen,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            kv_cache_sf=(k_sf, v_sf),
        )

    return run, out, out_descale


def reference_bf16(inputs: bd.Inputs) -> torch.Tensor:
    """Full-precision attention over the same logical K/V, for a sanity check."""
    case = inputs.case
    k = inputs.k_pages.reshape(case.batch, case.seqlen, case.heads_kv, bd.HEAD_DIM)
    v = inputs.v_pages.reshape(case.batch, case.seqlen, case.heads_kv, bd.HEAD_DIM)
    group = case.heads_q // case.heads_kv
    k = k.repeat_interleave(group, dim=2).permute(0, 2, 1, 3).float()
    v = v.repeat_interleave(group, dim=2).permute(0, 2, 1, 3).float()
    q = inputs.query.unsqueeze(2).float()  # [b, hq, 1, d]
    scores = (q @ k.transpose(-1, -2)) * inputs.softmax_scale
    return (scores.softmax(-1) @ v).squeeze(2)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def run_case(case: bd.Case, device, args) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    fp4_run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
    trtllm_run, trtllm_out, out_descale = make_trtllm(
        inputs, page_size=args.trtllm_page_size
    )

    record: dict[str, object] = {
        "case": case.label,
        "batch": case.batch,
        "seqlen": case.seqlen,
        "heads_q": case.heads_q,
        "heads_kv": case.heads_kv,
        "kv_gib": kv_bytes(case) / 2**30,
    }

    if args.check:
        reference = reference_bf16(inputs)
        record["fp4_cos"] = cosine(fp4_run(), reference)
        trtllm_run()
        record["trtllm_cos"] = cosine(trtllm_out.float() * out_descale, reference)

    with gpu_lock(LOCK_PATH):
        record["fp4_us"], record["fp4_error"] = graph_us(
            fp4_run, args.warmup, args.iters, args.repeats
        )
        record["trtllm_us"], record["trtllm_error"] = graph_us(
            trtllm_run, args.warmup, args.iters, args.repeats
        )
        if args.with_fa4:
            best = math.inf
            for splits in {1, bd.fa4_auto_splits(case, device)}:
                value, error = graph_us(
                    bd.make_fa4(inputs, splits), args.warmup, args.iters, args.repeats
                )
                if not error:
                    best = min(best, value)
            record["fa4_us"] = best if best < math.inf else float("nan")

    record["ratio"] = record["fp4_us"] / record["trtllm_us"]
    for side in ("fp4", "trtllm"):
        record[f"{side}_tbps"] = kv_bytes(case) / (record[f"{side}_us"] * 1e-6) / 1e12
    del inputs, fp4_run, trtllm_run, trtllm_out
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--head-configs",
        type=str,
        default="32:8",
        help="comma-separated heads_q:heads_kv pairs, e.g. 32:8,32:4,32:1,32:32",
    )
    parser.add_argument("--seqlens", type=str, default="16384")
    parser.add_argument("--batches", type=str, default="1,4,16,64")
    parser.add_argument("--max-kv-tokens", type=int, default=8_400_000)
    parser.add_argument(
        "--trtllm-page-size",
        type=int,
        default=128,
        help="page granularity handed to trtllm-gen; the token order is identical",
    )
    parser.add_argument("--with-fa4", action="store_true")
    parser.add_argument("--check", action="store_true", help="report cosine vs bf16")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    head_configs = [
        tuple(int(part) for part in entry.split(":"))
        for entry in args.head_configs.split(",")
    ]

    rows = []
    for heads_q, heads_kv in head_configs:
        for seqlen in (int(value) for value in args.seqlens.split(",")):
            for batch in (int(value) for value in args.batches.split(",")):
                if batch * seqlen * heads_kv > args.max_kv_tokens * 8:
                    continue
                case = bd.Case(
                    batch=batch,
                    seqlen=seqlen,
                    heads_q=heads_q,
                    heads_kv=heads_kv,
                )
                try:
                    record = run_case(case, device, args)
                except Exception as error:  # one bad config must not end the grid
                    print(f"{case.label:<32} SKIPPED {type(error).__name__}: {error}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                rows.append(record)
                extra = ""
                if args.check:
                    extra = (
                        f"  cos fp4 {record['fp4_cos']:.4f}"
                        f" trtllm {record['trtllm_cos']:.4f}"
                    )
                if args.with_fa4:
                    extra += f"  fa4 {record['fa4_us']:9.1f} us"
                print(
                    f"{case.label:<32}"
                    f" fp4 {record['fp4_us']:9.1f} us"
                    f"  trtllm {record['trtllm_us']:9.1f} us"
                    f"  ratio {record['ratio']:6.3f}"
                    f"  | {record['fp4_tbps']:5.2f} vs {record['trtllm_tbps']:5.2f} TB/s"
                    f"{extra}",
                    flush=True,
                )

    def summarize(key: str, title: str) -> None:
        print(f"\ngeometric mean of fp4/trtllm by {title} (lower is better):")
        for value in sorted({row[key] for row in rows}):
            group = [row for row in rows if row[key] == value]
            geomean = statistics.geometric_mean([row["ratio"] for row in group])
            print(f"  {value:<10} {geomean:6.3f}   ({len(group)} cases)")

    if rows:
        summarize("seqlen", "sequence length")
        if len(head_configs) > 1:
            summarize("heads_kv", "kv heads")
        print(
            "\noverall geomean "
            f"{statistics.geometric_mean([row['ratio'] for row in rows]):.3f}"
            f"   ({len(rows)} cases)"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
