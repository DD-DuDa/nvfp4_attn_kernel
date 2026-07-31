"""Compare this repository's FP4 decode against the other NVFP4 decodes on SM100.

Three kernels read the same logical K/V:

- this repository's tcgen05 FP4 decode,
- the trtllm-gen NVFP4 decode (what vLLM and SGLang route to on SM100),
- FlashInfer's arch-generic FA2 paged decode, which dequantizes E2M1 to BF16 in
  registers and computes with ordinary ``mma.sync``.

``--with-fa4`` adds a fourth column: FlashAttention-4 over the same pages in
BF16. That is not an NVFP4 kernel, it is the "don't quantize the cache at all"
baseline (decision D0), so it moves 3.56x the bytes and only its *time* ratio
is comparable -- the TB/s column is computed over BF16 bytes and must not be
read against the NVFP4 columns.

Every NVFP4 side is handed an already-quantized paged K/V cache, so none of
them pays for quantization. The remaining asymmetries are inherent to the
kernels and are reported alongside the timings:

- trtllm-gen keeps the query in FP8 E4M3, dequantizes the FP4 K/V to FP8 before
  the MMAs, and only emits an FP8 output. This kernel keeps the query in FP4
  E2M1, feeds E2M1 straight to the block-scaled MMA, and emits BF16; FA2 takes
  a BF16 query, dequantizes to BF16 in registers, and emits BF16. The accuracy
  column therefore compares three different arithmetic pipelines, not three
  implementations of one.
- Each side wants a different E4M3 block-scale layout: trtllm-gen wants V
  scales in its 4-token interleave, FA2 wants both K and V scales linear, and
  this kernel wants the tcgen05-native swizzle. Each side is therefore
  quantized from the same BF16 pages with its own quantizer.

trtllm-gen has no MHA kernel, so ``heads_q == heads_kv`` rows report a skip for
that column and still compare the other two.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests/kernel_profile \
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
# FA2 sizes its split-K scratch from the case, and MQA at long context asks for
# more than trtllm-gen ever does, so this side gets its own larger buffer.
FA2_WORKSPACE_BYTES = 1024 * 1024 * 1024
E4M3_MAX = 448.0
# trtllm-gen dequantizes NVFP4 to FP8 before the attention MMAs, so a block's
# reconstructed ``e2m1 * block_scale`` has to stay inside E4M3. Sizing the
# global descale as ``amax / 448`` caps the block scales at 448/6, which leaves
# room for the E2M1 max of 6. The looser ``amax / (6 * 448)`` keeps the block
# scales themselves in range but lets the product overflow, silently clipping
# the largest K/V entries inside the kernel.
NVFP4_GLOBAL_RANGE = E4M3_MAX
# Ratio columns, as (record field, label). All are this kernel over the
# baseline, so below 1.0 means this kernel is faster.
BASELINES = (("ratio", "trtllm"), ("ratio_fi", "fa2"), ("ratio_fa4", "fa4"))


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
    ``e2m1 * e4m3_block_scale * global_scale``. See ``NVFP4_GLOBAL_RANGE`` for
    why it is sized so the block scales stay under 448/6 rather than 448.

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
    q_descale = q_amax / E4M3_MAX
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
    # an output scale.
    bmm1_scale = q_descale * k_descale * inputs.softmax_scale

    def build(out_descale: float):
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

        return run

    # Attention output is a convex combination of V rows, so |out| <= amax(V)
    # and this bound never saturates. It leaves most of the E4M3 range unused,
    # but tightening it measurably *hurts*: the kernel stops scaling the output
    # linearly with bmm2_scale somewhere above ~5x, so a fitted descale clips.
    # The loose bound is the accurate one here.
    out_descale = inputs.v_pages.abs().max().float().item() / E4M3_MAX
    return build(out_descale), out, out_descale


def nvfp4_quantize_nhd_linear(pages: torch.Tensor):
    """Quantize BF16 pages into the FA2 NVFP4 KV format.

    FA2 reads NHD directly, so bench_decode's ``[pages, 128, heads, dim]``
    layout is already what it wants and no transpose copy is needed. Unlike
    trtllm-gen, FA2 reads both K and V scales linearly, so the swizzled
    quantizer cannot be shared.

    ``global_sf`` here is a scale, not a descale: ``fp4_quantize`` multiplies by
    it, and the kernel is handed the reciprocal as ``k_scale``/``v_scale``.
    """
    from flashinfer.fp4_quantization import fp4_quantize

    num_pages, page_size, heads_kv, dim = pages.shape
    amax = max(pages.abs().max().float().item(), 1e-12)
    global_sf = torch.tensor([E4M3_MAX / amax], dtype=torch.float32, device=pages.device)
    packed, scales = fp4_quantize(
        pages.reshape(-1, dim), global_sf, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    data = packed.view(torch.uint8).reshape(num_pages, page_size, heads_kv, dim // 2)
    scales = scales.view(FP8).reshape(num_pages, page_size, heads_kv, dim // 16)
    return data, scales, amax / E4M3_MAX


def make_flashinfer(inputs: bd.Inputs):
    """Build a zero-argument callable that runs one FA2 NVFP4 decode."""
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    case = inputs.case
    device = inputs.query.device

    k_data, k_sf, k_descale = nvfp4_quantize_nhd_linear(inputs.k_pages)
    v_data, v_sf, v_descale = nvfp4_quantize_nhd_linear(inputs.v_pages)

    pages_per_row = case.seqlen // bd.PAGE_SIZE
    num_pages = case.batch * pages_per_row
    indptr = (
        torch.arange(case.batch + 1, device=device, dtype=torch.int32) * pages_per_row
    )
    indices = torch.arange(num_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full(
        (case.batch,), bd.PAGE_SIZE, device=device, dtype=torch.int32
    )

    workspace = torch.zeros(FA2_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True, backend="fa2"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        case.heads_q,
        case.heads_kv,
        bd.HEAD_DIM,
        bd.PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        sm_scale=inputs.softmax_scale,
    )

    def run():
        return wrapper.run(
            inputs.query,
            (k_data, v_data),
            kv_cache_sf=(k_sf, v_sf),
            k_scale=k_descale,
            v_scale=v_descale,
        )

    return run


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


def _try_build(factory, record: dict, side: str):
    """Build a baseline and run it once, or record why this shape is unsupported.

    The single call matters: trtllm-gen resolves its cubin inside the launcher,
    so an unsupported head configuration raises at call time, not build time.
    """
    try:
        run = factory()
        run()
        torch.cuda.synchronize()
        return run
    except Exception as error:
        record[f"{side}_error"] = f"{type(error).__name__}: {error}"
        return None


def _fmt(value, spec: str) -> str:
    """Render a metric, or a dash of the same width when the side has no kernel."""
    width = spec.split(".")[0] or "1"
    if value is None or not math.isfinite(value):
        return f"{'-':>{width}}"
    return f"{value:{spec}}"


def run_case(case: bd.Case, device, args) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    fp4_run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)

    record: dict[str, object] = {
        "case": case.label,
        "batch": case.batch,
        "seqlen": case.seqlen,
        "heads_q": case.heads_q,
        "heads_kv": case.heads_kv,
        "group": case.heads_q // case.heads_kv,
        "kv_gib": kv_bytes(case) / 2**30,
    }

    # A baseline that has no kernel for this shape must not take the whole row
    # down: the other two are still a valid comparison. trtllm-gen in
    # particular reports a missing cubin only once the launcher runs, so each
    # side is built and then called once behind the same guard.
    trtllm_out = out_descale = None

    def build_trtllm():
        nonlocal trtllm_out, out_descale
        run, trtllm_out, out_descale = make_trtllm(
            inputs, page_size=args.trtllm_page_size
        )
        return run

    trtllm_run = _try_build(build_trtllm, record, "trtllm")
    fi_run = _try_build(lambda: make_flashinfer(inputs), record, "fi")

    if args.check:
        reference = reference_bf16(inputs)
        record["fp4_cos"] = cosine(fp4_run(), reference)
        if trtllm_run is not None:
            trtllm_run()
            record["trtllm_cos"] = cosine(trtllm_out.float() * out_descale, reference)
        if fi_run is not None:
            record["fi_cos"] = cosine(fi_run(), reference)
        if args.with_fa4:
            record["fa4_cos"] = cosine(
                bd._output(bd.make_fa4(inputs, 1)()), reference
            )
        del reference

    with gpu_lock(LOCK_PATH):
        record["fp4_us"], record["fp4_error"] = graph_us(
            fp4_run, args.warmup, args.iters, args.repeats
        )
        for side, run in (("trtllm", trtllm_run), ("fi", fi_run)):
            if run is None:
                record[f"{side}_us"] = float("nan")
                continue
            record[f"{side}_us"], record[f"{side}_error"] = graph_us(
                run, args.warmup, args.iters, args.repeats
            )
        if args.with_fa4:
            # Decision D0: the FA4 baseline is the better of num_splits=1 and
            # the heuristic, which is the strictest reading of "what BF16
            # already gives you".
            best, best_splits = math.inf, None
            for splits in sorted({1, bd.fa4_auto_splits(case, device)}):
                value, error = graph_us(
                    bd.make_fa4(inputs, splits), args.warmup, args.iters, args.repeats
                )
                if not error and value < best:
                    best, best_splits = value, splits
            record["fa4_us"] = best if best < math.inf else float("nan")
            record["fa4_splits"] = best_splits
        else:
            record["fa4_us"] = float("nan")

    for side in ("fp4", "trtllm", "fi"):
        micros = record[f"{side}_us"]
        record[f"{side}_tbps"] = kv_bytes(case) / (micros * 1e-6) / 1e12
    # FA4 reads the cache in BF16, so its bandwidth is over 3.5x the bytes. The
    # time ratio is the cross-precision comparison that means something; the
    # TB/s columns are not comparable across the two byte counts.
    record["fa4_tbps"] = (
        bd.kv_bytes(case, "fa4") / (record["fa4_us"] * 1e-6) / 1e12
    )
    record["ratio"] = record["fp4_us"] / record["trtllm_us"]
    record["ratio_fi"] = record["fp4_us"] / record["fi_us"]
    record["ratio_fa4"] = record["fp4_us"] / record["fa4_us"]
    del inputs, fp4_run, trtllm_run, fi_run, trtllm_out
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
                        f" trt {_fmt(record.get('trtllm_cos'), '.4f')}"
                        f" fi {_fmt(record.get('fi_cos'), '.4f')}"
                        f" fa4 {_fmt(record.get('fa4_cos'), '.4f')}"
                    )
                for side in ("trtllm", "fi"):
                    if record.get(f"{side}_error"):
                        extra += f"  [{side}: {record[f'{side}_error'][:60]}]"
                print(
                    f"{case.label:<32}"
                    f" fp4 {record['fp4_us']:9.1f}"
                    f"  trt {_fmt(record['trtllm_us'], '9.1f')}"
                    f"  fi {_fmt(record['fi_us'], '9.1f')}"
                    f"  fa4 {_fmt(record['fa4_us'], '9.1f')} us"
                    f"  | {record['fp4_tbps']:5.2f}"
                    f" {_fmt(record['trtllm_tbps'], '5.2f')}"
                    f" {_fmt(record['fi_tbps'], '5.2f')}"
                    f" {_fmt(record['fa4_tbps'], '5.2f')} TB/s"
                    f"  | ratio trt {_fmt(record['ratio'], '6.3f')}"
                    f" fi {_fmt(record['ratio_fi'], '6.3f')}"
                    f" fa4 {_fmt(record['ratio_fa4'], '6.3f')}"
                    f"{extra}",
                    flush=True,
                )

    def summarize(key: str, title: str) -> None:
        print(f"\ngeomean of fp4/baseline by {title} (lower is better):")
        for value in sorted({row[key] for row in rows}):
            group = [row for row in rows if row[key] == value]
            cells = []
            for field, name in BASELINES:
                usable = [row[field] for row in group if math.isfinite(row[field])]
                cells.append(
                    f"{name} {statistics.geometric_mean(usable):6.3f} ({len(usable)})"
                    if usable
                    else f"{name}      -  (0)"
                )
            print(f"  {value:<10} {'   '.join(cells)}")

    if rows:
        summarize("seqlen", "sequence length")
        summarize("group", "gqa group size")
        for field, name in BASELINES:
            usable = [row[field] for row in rows if math.isfinite(row[field])]
            if usable:
                print(
                    f"\noverall geomean vs {name} "
                    f"{statistics.geometric_mean(usable):.3f}"
                    f"   ({len(usable)} of {len(rows)} cases)"
                )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
