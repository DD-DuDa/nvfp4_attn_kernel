"""Benchmark FA4 and NVFP4 paged decode with reproducible result metadata.

Clean performance gates use CUDA events. ``--breakdown`` is a separate
Torch-Profiler pass for per-kernel attribution and must not run under IKET.
FA4 always uses the varlen entry point so split-K keeps pack-GQA enabled.
``fp4_decode`` splits only as far as its ``num_splits`` argument says, so
``--fp4-splits`` passes that count explicitly and the reported ``num_splits``
is the count that ran.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.interface import num_splits_heuristic
from nvfp4_decode_kernel import _quantize, fp4_decode
from nvfp4_decode_kernel._decode import (
    _decode_compile_cache,
    _split_decode_compile_cache,
    split_k_heuristic,
)


PAGE_SIZE = 128
HEAD_DIM = 128
BF16_BYTES_PER_TOKEN_HEAD = 2 * HEAD_DIM * 2
FP4_BYTES_PER_TOKEN_HEAD = 2 * (HEAD_DIM // 2 + HEAD_DIM // 16)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SEQLENS = (1024, 4096, 16384, 65536)
DEFAULT_HEAD_CONFIGS = ((8, 8), (32, 8), (32, 1))  # MHA, GQA-4, MQA
DEFAULT_MAX_KV_TOKENS = 8_400_000
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Case:
    batch: int
    seqlen: int
    heads_q: int
    heads_kv: int

    @property
    def attention_type(self) -> str:
        if self.heads_q == self.heads_kv:
            return "mha"
        if self.heads_kv == 1:
            return "mqa"
        return f"gqa{self.heads_q // self.heads_kv}"

    @property
    def label(self) -> str:
        return (
            f"{self.attention_type}_b{self.batch}_s{self.seqlen}"
            f"_hq{self.heads_q}_hkv{self.heads_kv}"
        )


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
    query_fp4: torch.Tensor
    query_scales: torch.Tensor
    seqused_fp4_full: torch.Tensor
    seqused_fp4_hybrid: torch.Tensor
    seqused_residual: torch.Tensor
    residual_page_ids: torch.Tensor
    has_bf16: torch.Tensor
    softmax_scale: float


def build_inputs(
    case: Case, device: torch.device, *, quantize_chunk_pages: int
) -> Inputs:
    if case.seqlen % PAGE_SIZE:
        raise ValueError("seqlen must be a multiple of the 128-token page size")
    if case.heads_q % case.heads_kv:
        raise ValueError("heads_q must be divisible by heads_kv")

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
    key_pages_fp4, key_scales = quantize_pages_chunked(
        k_pages, is_value=False, chunk_pages=quantize_chunk_pages
    )
    value_pages_fp4, value_scales = quantize_pages_chunked(
        v_pages, is_value=True, chunk_pages=quantize_chunk_pages
    )
    query_fp4, query_scales = _quantize.quantize_query(
        query, heads_kv=case.heads_kv
    )

    seqused_fp4_hybrid = seqused_k - PAGE_SIZE
    seqused_residual = torch.full(
        (case.batch,), PAGE_SIZE, device=device, dtype=torch.int32
    )
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
        query_fp4=query_fp4,
        query_scales=query_scales,
        seqused_fp4_full=seqused_k.clone(),
        seqused_fp4_hybrid=seqused_fp4_hybrid,
        seqused_residual=seqused_residual,
        residual_page_ids=residual_page_ids,
        has_bf16=torch.ones(case.batch, device=device, dtype=torch.bool),
        softmax_scale=HEAD_DIM**-0.5,
    )


def quantize_pages_chunked(
    pages: torch.Tensor, *, is_value: bool, chunk_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    quantizer = (
        _quantize.quantize_value_pages
        if is_value
        else _quantize.quantize_key_pages
    )
    page_count = pages.shape[0]
    if page_count <= chunk_pages:
        return quantizer(pages)

    fp4_output = None
    scale_storage = None
    scale_output = None
    for start in range(0, page_count, chunk_pages):
        stop = min(start + chunk_pages, page_count)
        fp4_chunk, scale_chunk = quantizer(pages[start:stop])
        if fp4_output is None:
            fp4_output = torch.empty(
                (page_count, *fp4_chunk.shape[1:]),
                dtype=fp4_chunk.dtype,
                device=fp4_chunk.device,
            )
            page_stride = scale_chunk.stride(0)
            scale_storage = torch.empty(
                page_count * page_stride,
                dtype=torch.uint8,
                device=scale_chunk.device,
            )
            scale_output = scale_storage.as_strided(
                (page_count, *scale_chunk.shape[1:]),
                scale_chunk.stride(),
            ).view(scale_chunk.dtype)
        fp4_output[start:stop].copy_(fp4_chunk)
        scale_output[start:stop].copy_(scale_chunk)
    assert fp4_output is not None and scale_output is not None
    return fp4_output, scale_output


def fa4_auto_splits(case: Case, device: torch.device) -> int:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    total_mblocks = case.batch * case.heads_kv
    num_n_blocks = case.seqlen // PAGE_SIZE
    return max(1, num_splits_heuristic(total_mblocks, sms, num_n_blocks, 128))


def fp4_auto_splits(case: Case, device: torch.device) -> int:
    """The split count ``split_k_heuristic`` would pick for this case.

    ``fp4_decode`` does not consult the heuristic; it splits only as far as its
    ``num_splits`` argument says. A caller that wants the heuristic's answer has
    to ask for it here and pass the same value on, so that what is reported and
    what runs are one number.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return split_k_heuristic(
        case.batch,
        case.heads_kv,
        case.seqlen // PAGE_SIZE,
        sms=sms,
    )


def parse_split_request(value: str) -> str | int:
    """Parse a split-count request: ``auto`` or an explicit power of two."""
    if value == "auto":
        return "auto"
    splits = int(value) if value.isdigit() else 0
    if splits < 1 or splits & (splits - 1):
        raise SystemExit(
            f"split count must be 'auto' or a positive power of two, got {value!r}"
        )
    return splits


def resolve_fp4_splits(
    request: str | int, case: Case, device: torch.device
) -> int:
    return fp4_auto_splits(case, device) if request == "auto" else int(request)


def make_fa4(inputs: Inputs, num_splits: int) -> Callable[[], object]:
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


def make_fp4(
    inputs: Inputs, hybrid: bool, *, prequantized_query: bool, num_splits: int = 1
) -> Callable[[], torch.Tensor]:
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
            trusted_metadata=True,
            num_splits=num_splits,
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
            trusted_metadata=True,
            num_splits=num_splits,
        )

    def run_prequantized():
        return fp4_decode(
            key_pages_fp4=inputs.key_pages_fp4,
            key_scales=inputs.key_scales,
            value_pages_fp4=inputs.value_pages_fp4,
            value_scales=inputs.value_scales,
            fp4_page_table=inputs.page_table,
            seqused_fp4=inputs.seqused_fp4_full,
            query_fp4=inputs.query_fp4,
            query_scales=inputs.query_scales,
            softmax_scale=inputs.softmax_scale,
            trusted_metadata=True,
            num_splits=num_splits,
        )

    if prequantized_query:
        if hybrid:
            raise ValueError(
                "pre-quantized query benchmark currently covers pure FP4 KV"
            )
        return run_prequantized
    return run_hybrid if hybrid else run_pure


def clear_fp4_compile_caches() -> None:
    """Force the next FP4 call in this process to compile under IKET."""
    _decode_compile_cache.clear()
    _split_decode_compile_cache.clear()


def kv_bytes(case: Case, variant: str) -> int:
    tokens = case.batch * case.seqlen
    if variant.startswith("fa4"):
        return tokens * case.heads_kv * BF16_BYTES_PER_TOKEN_HEAD
    if variant == "fp4_hybrid_bf16q":
        fp4_tokens = case.batch * (case.seqlen - PAGE_SIZE)
        bf16_tokens = case.batch * PAGE_SIZE
        return case.heads_kv * (
            fp4_tokens * FP4_BYTES_PER_TOKEN_HEAD
            + bf16_tokens * BF16_BYTES_PER_TOKEN_HEAD
        )
    return tokens * case.heads_kv * FP4_BYTES_PER_TOKEN_HEAD


def _output(result):
    return result[0] if isinstance(result, tuple) else result


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def measure_wall_ms(run: Callable[[], object], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def measure_event_gpu_ms(
    run: Callable[[], object], iters: int, warmup: int, repeats: int = 5
) -> tuple[float, float]:
    """Return the median per-iteration GPU time and the observed spread.

    One timing region is a single sample however many iterations it averages,
    so any one-off host stall or driver interruption inside it inflates the
    result with nothing to reveal that it happened. Independent repeats let the
    median reject those, and the returned spread lets a caller tell a real
    change from a noisy measurement.
    """
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    samples.sort()
    median = samples[len(samples) // 2]
    return median, (samples[-1] - samples[0]) / median if median > 0 else 0.0


def measure_kernel_breakdown(
    run: Callable[[], object], iters: int, warmup: int
) -> dict[str, float]:
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
        if micros > 0:
            breakdown[event.key] = (
                breakdown.get(event.key, 0.0) + micros / 1e3 / iters
            )
    return dict(sorted(breakdown.items(), key=lambda item: -item[1]))


def query_quantization_ms(kernels: dict[str, float]) -> float:
    return sum(
        ms
        for name, ms in kernels.items()
        if "quantize_query_kernel" in name
    )


def variant_factory(
    name: str,
    inputs: Inputs,
    device: torch.device,
    fp4_splits: str | int = 1,
) -> tuple[Callable[[], object] | None, int | None, str | None]:
    """Build a variant's runner and the split count it will actually use.

    The returned count is the one handed to the kernel, not a prediction of
    what it might choose, so a reported ``num_splits`` always describes the
    measured run.
    """
    if name == "fa4_bf16":
        return make_fa4(inputs, num_splits=1), 1, None
    if name == "fa4_split":
        splits = fa4_auto_splits(inputs.case, device)
        return make_fa4(inputs, num_splits=splits), splits, None
    if name in ("fp4_pure_bf16q", "fp4_hybrid_bf16q", "fp4_pure_fp4q"):
        splits = resolve_fp4_splits(fp4_splits, inputs.case, device)
        hybrid = name == "fp4_hybrid_bf16q"
        prequantized_query = name.endswith("_fp4q")
        return make_fp4(
            inputs,
            hybrid=hybrid,
            prequantized_query=prequantized_query,
            num_splits=splits,
        ), splits, None
    raise KeyError(name)


DEFAULT_VARIANTS = (
    "fa4_bf16",
    "fa4_split",
    "fp4_pure_bf16q",
    "fp4_pure_fp4q",
)


def input_contract(variant: str) -> str:
    if variant.startswith("fa4"):
        return "bf16_q_bf16_kv"
    if variant.endswith("_fp4q"):
        return "fp4_q_fp4_kv"
    return "bf16_q_fp4_kv"


def unavailable_row(
    case: Case, variant: str, num_splits: int | None, reason: str | None
) -> dict:
    return {
        **asdict(case),
        "attention_type": case.attention_type,
        "variant": variant,
        "input_contract": input_contract(variant),
        "status": "unavailable",
        "unavailable_reason": reason,
        "num_splits": num_splits,
        "gpu_ms": None,
        "gpu_ms_spread": None,
        "wall_ms": None,
        "kv_gib": kv_bytes(case, variant) / 2**30,
        "kv_gbps": None,
        "cosine_vs_fa4": None,
        "kernels": {},
        "query_quantization_ms": None,
        "query_quantization_fraction": None,
        "case_peak_memory_bytes": None,
    }


def run_case(
    case: Case,
    device: torch.device,
    variants: list[str],
    iters: int,
    warmup: int,
    breakdown: bool,
    structural_only: bool,
    quantize_chunk_pages: int,
    fp4_splits: str | int = 1,
) -> list[dict]:
    torch.cuda.reset_peak_memory_stats(device)
    inputs = build_inputs(
        case, device, quantize_chunk_pages=quantize_chunk_pages
    )
    reference = _output(make_fa4(inputs, num_splits=1)()).reshape(
        case.batch, case.heads_q, HEAD_DIM
    )

    rows: list[dict] = []
    for name in variants:
        run, num_splits, unavailable_reason = variant_factory(
            name, inputs, device, fp4_splits
        )
        if run is None:
            print(f"  {name}: unavailable ({unavailable_reason})", flush=True)
            rows.append(unavailable_row(case, name, num_splits, unavailable_reason))
            continue

        # A split count the kernel cannot serve for this variant is refused at
        # the call, not silently downgraded, so one unservable request must not
        # end the sweep.
        try:
            out = _output(run()).reshape(case.batch, case.heads_q, HEAD_DIM)
        except ValueError as error:
            print(f"  {name}: unavailable ({error})", flush=True)
            rows.append(unavailable_row(case, name, num_splits, str(error)))
            continue
        gpu_ms, gpu_ms_spread = (
            (None, None)
            if structural_only
            else measure_event_gpu_ms(run, iters, warmup)
        )
        wall_ms = (
            None if structural_only else measure_wall_ms(run, iters, warmup)
        )
        kernels = (
            measure_kernel_breakdown(run, iters, warmup) if breakdown else {}
        )
        quant_ms = query_quantization_ms(kernels)
        moved = kv_bytes(case, name)
        rows.append(
            {
                **asdict(case),
                "attention_type": case.attention_type,
                "variant": name,
                "input_contract": input_contract(name),
                "status": "ok",
                "unavailable_reason": None,
                "num_splits": num_splits,
                "gpu_ms": gpu_ms,
                "gpu_ms_spread": gpu_ms_spread,
                "wall_ms": wall_ms,
                "kv_gib": moved / 2**30,
                "kv_gbps": (
                    moved / (gpu_ms * 1e-3) / 1e9
                    if gpu_ms is not None
                    else None
                ),
                "cosine_vs_fa4": cosine(out, reference),
                "kernels": kernels,
                "query_quantization_ms": quant_ms if kernels else None,
                "query_quantization_fraction": (
                    quant_ms / sum(kernels.values())
                    if kernels and sum(kernels.values()) > 0
                    else None
                ),
                "case_peak_memory_bytes": None,
            }
        )

    peak_memory = torch.cuda.max_memory_allocated(device)
    for row in rows:
        row["case_peak_memory_bytes"] = peak_memory
    del inputs, reference
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(rows: list[dict]) -> dict:
    ok = [row for row in rows if row["status"] == "ok"]
    by_case: dict[tuple[int, int, int, int], dict[str, dict]] = {}
    for row in ok:
        key = (
            row["batch"],
            row["seqlen"],
            row["heads_q"],
            row["heads_kv"],
        )
        by_case.setdefault(key, {})[row["variant"]] = row

    ratios: list[dict] = []
    for key, variants in sorted(by_case.items()):
        if "fa4_bf16" not in variants or "fa4_split" not in variants:
            continue
        baseline = min(
            (variants["fa4_bf16"], variants["fa4_split"]),
            key=lambda row: row["gpu_ms"],
        )
        for fp4_name in ("fp4_pure_bf16q", "fp4_pure_fp4q"):
            fp4_row = variants.get(fp4_name)
            if (
                fp4_row is None
                or fp4_row["status"] != "ok"
                or fp4_row["gpu_ms"] is None
                or baseline["gpu_ms"] is None
            ):
                continue
            ratios.append(
                {
                    "batch": key[0],
                    "seqlen": key[1],
                    "heads_q": key[2],
                    "heads_kv": key[3],
                    "attention_type": fp4_row["attention_type"],
                    "fp4_variant": fp4_name,
                    "fa4_baseline_variant": baseline["variant"],
                    "fa4_baseline_num_splits": baseline["num_splits"],
                    "ratio_vs_fa4_better": fp4_row["gpu_ms"] / baseline["gpu_ms"],
                }
            )

    per_path: dict[str, dict] = {}
    for fp4_name in ("fp4_pure_bf16q", "fp4_pure_fp4q"):
        path_rows = [row for row in ratios if row["fp4_variant"] == fp4_name]
        by_head_config: dict[str, dict] = {}
        for heads_q, heads_kv in sorted(
            {(row["heads_q"], row["heads_kv"]) for row in path_rows}
        ):
            config_rows = [
                row
                for row in path_rows
                if row["heads_q"] == heads_q and row["heads_kv"] == heads_kv
            ]
            per_seqlen = {}
            for seqlen in DEFAULT_SEQLENS:
                batch_values = [
                    row["ratio_vs_fa4_better"]
                    for row in config_rows
                    if row["seqlen"] == seqlen
                ]
                if batch_values:
                    per_seqlen[str(seqlen)] = _gmean(batch_values)
            config_values = [
                row["ratio_vs_fa4_better"] for row in config_rows
            ]
            by_head_config[f"{heads_q}:{heads_kv}"] = {
                "heads_q": heads_q,
                "heads_kv": heads_kv,
                "attention_type": config_rows[0]["attention_type"],
                "per_seqlen_batch_geomean": per_seqlen,
                "diagnostic_overall_geomean": _gmean(config_values),
                "worst_point": max(
                    config_rows,
                    key=lambda row: row["ratio_vs_fa4_better"],
                ),
            }
        values = [row["ratio_vs_fa4_better"] for row in path_rows]
        per_path[fp4_name] = {
            "implemented": any(
                row["variant"] == fp4_name and row["status"] == "ok"
                for row in rows
            ),
            "timed": bool(values),
            "by_head_config": by_head_config,
            "diagnostic_cross_head_geomean": (
                _gmean(values) if values else None
            ),
            "worst_point": (
                max(path_rows, key=lambda row: row["ratio_vs_fa4_better"])
                if values
                else None
            ),
        }
    return {"ratios": ratios, "paths": per_path}


def coverage_report(
    rows: list[dict],
    args: argparse.Namespace,
    head_configs: list[tuple[int, int]],
) -> dict:
    observed = {
        (
            row["batch"],
            row["seqlen"],
            row["heads_q"],
            row["heads_kv"],
        )
        for row in rows
    }
    requested = {
        (batch, seqlen, heads_q, heads_kv)
        for heads_q, heads_kv in head_configs
        for batch in args.batches
        for seqlen in args.seqlens
        if batch * seqlen <= args.max_kv_tokens
    }
    phase0_required = {
        (batch, seqlen, heads_q, heads_kv)
        for heads_q, heads_kv in DEFAULT_HEAD_CONFIGS
        for batch in DEFAULT_BATCHES
        for seqlen in DEFAULT_SEQLENS
    }
    return {
        "requested_complete": requested <= observed,
        "requested_missing": [
            list(item) for item in sorted(requested - observed)
        ],
        "phase0_required_grid_complete": phase0_required <= observed,
        "phase0_required_grid_missing": [
            list(item) for item in sorted(phase0_required - observed)
        ],
    }


def format_table(rows: list[dict]) -> str:
    by_case: dict[tuple[int, int, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (
            row["batch"],
            row["seqlen"],
            row["heads_q"],
            row["heads_kv"],
        )
        by_case.setdefault(key, {})[row["variant"]] = row

    header = (
        f"{'type':>5} {'batch':>5} {'seqlen':>7} {'hq/hkv':>7}"
        f" | {'FA4 best':>15} | {'FP4 BF16-Q':>15} | {'FP4-Q':>12}"
    )
    lines = [header, "-" * len(header)]
    for key, variants in sorted(by_case.items()):
        fa4 = [
            variants[name]
            for name in ("fa4_bf16", "fa4_split")
            if (
                name in variants
                and variants[name]["status"] == "ok"
                and variants[name]["gpu_ms"] is not None
            )
        ]
        baseline = min(fa4, key=lambda row: row["gpu_ms"]) if fa4 else None
        bf16q = variants.get("fp4_pure_bf16q")
        fp4q = variants.get("fp4_pure_fp4q")
        fa4_text = (
            f"{baseline['gpu_ms']:.4f} {baseline['variant']}"
            if baseline
            else "-"
        )
        bf16q_text = (
            f"{bf16q['gpu_ms']:.4f} s{bf16q['num_splits']}"
            if (
                bf16q
                and bf16q["status"] == "ok"
                and bf16q["gpu_ms"] is not None
            )
            else "-"
        )
        fp4q_text = (
            f"{fp4q['gpu_ms']:.4f} s{fp4q['num_splits']}"
            if (
                fp4q
                and fp4q["status"] == "ok"
                and fp4q["gpu_ms"] is not None
            )
            else "Phase 1"
        )
        lines.append(
            f"{next(iter(variants.values()))['attention_type']:>5}"
            f" {key[0]:>5} {key[1]:>7} {key[2]:>2}/{key[3]:<4}"
            f" | {fa4_text:>15} | {bf16q_text:>15} | {fp4q_text:>12}"
        )
    return "\n".join(lines)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(args: argparse.Namespace, device: torch.device) -> dict:
    env_root = sys.prefix
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env_root,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            "torch": package_version("torch"),
            "nvidia-cutlass-dsl": package_version("nvidia-cutlass-dsl"),
            "flash-attn-4": package_version("flash-attn-4"),
            "flashinfer-python": package_version("flashinfer-python"),
        },
        "gpu": {
            "requested_device": args.device,
            "visible_device": device.index,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "multi_processor_count": (
                torch.cuda.get_device_properties(device).multi_processor_count
            ),
            "total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "baseline_definition": (
            "minimum CUDA-event gpu_ms of FA4 num_splits=1 and FA4 heuristic; "
            "both use flash_attn_varlen_func"
        ),
        "fp4_q_gate_status": "available",
    }


def parse_head_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.head_configs:
        configs = []
        for item in args.head_configs:
            try:
                heads_q, heads_kv = (int(value) for value in item.split(":", 1))
            except ValueError as error:
                raise SystemExit(
                    f"invalid --head-config {item!r}; expected HEADS_Q:HEADS_KV"
                ) from error
            configs.append((heads_q, heads_kv))
        return configs
    if args.heads_q is not None or args.heads_kv is not None:
        return [(args.heads_q or 32, args.heads_kv or 8)]
    return list(DEFAULT_HEAD_CONFIGS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "index within CUDA_VISIBLE_DEVICES; §6.1 requires "
            "CUDA_VISIBLE_DEVICES=1, so the visible device index is 0"
        ),
    )
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--seqlens", type=int, nargs="+", default=DEFAULT_SEQLENS)
    parser.add_argument(
        "--head-config",
        dest="head_configs",
        action="append",
        help="repeatable HEADS_Q:HEADS_KV; default covers MHA, GQA, and MQA",
    )
    parser.add_argument("--heads-q", type=int, default=None)
    parser.add_argument("--heads-kv", type=int, default=None)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--max-kv-tokens",
        type=int,
        default=DEFAULT_MAX_KV_TOKENS,
        help="skip cases whose batch*seqlen exceeds this KV token budget",
    )
    parser.add_argument(
        "--quantize-chunk-pages",
        type=int,
        default=8192,
        help=(
            "quantize large benchmark inputs in bounded page chunks; this is "
            "setup only and is outside measured decode calls"
        ),
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--fp4-splits",
        type=parse_split_request,
        default="auto",
        help=(
            "split count handed to every FP4 variant: 'auto' for what "
            "split_k_heuristic would pick, or a power of two; the reported "
            "num_splits is this value, not a prediction"
        ),
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="run a separate Torch-Profiler pass; never use under IKET",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help=(
            "launch workloads without timing-shaped output; use under IKET, "
            "where instrumented durations are not performance evidence"
        ),
    )
    parser.add_argument(
        "--clear-fp4-compile-cache",
        action="store_true",
        help="clear in-process FP4 caches before each case (required for IKET)",
    )
    parser.add_argument(
        "--require-phase0-grid",
        action="store_true",
        help="exit nonzero unless the complete Phase 0 MHA/GQA/MQA grid ran",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.structural_only and args.breakdown:
        raise SystemExit("--structural-only and --breakdown are mutually exclusive")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SystemExit(f"SM100 is required, found compute capability {capability}")

    head_configs = parse_head_configs(args)
    print(f"device: {torch.cuda.get_device_name(device)} (cuda:{args.device})")
    print(f"head_configs={head_configs} head_dim={HEAD_DIM}")
    print(f"iters={args.iters} warmup={args.warmup}")
    print(f"fp4_splits={args.fp4_splits}")
    print()

    torch.manual_seed(0)
    rows: list[dict] = []
    skipped: list[dict] = []
    for heads_q, heads_kv in head_configs:
        for batch in args.batches:
            for seqlen in args.seqlens:
                if batch * seqlen > args.max_kv_tokens:
                    skipped.append(
                        {
                            "batch": batch,
                            "seqlen": seqlen,
                            "heads_q": heads_q,
                            "heads_kv": heads_kv,
                            "reason": "max_kv_tokens",
                        }
                    )
                    continue
                case = Case(batch, seqlen, heads_q, heads_kv)
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
                        args.structural_only,
                        args.quantize_chunk_pages,
                        args.fp4_splits,
                    )
                )

    summary = summarize(rows)
    print()
    print(format_table(rows))
    print()
    print(json.dumps(summary["paths"], indent=2))

    coverage = coverage_report(rows, args, head_configs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance(args, device),
        "coverage": {
            "requested_batches": list(args.batches),
            "requested_seqlens": list(args.seqlens),
            "requested_head_configs": [list(config) for config in head_configs],
            "skipped": skipped,
            **coverage,
        },
        "results": rows,
        "summary": summary,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    if args.require_phase0_grid and not coverage["phase0_required_grid_complete"]:
        raise SystemExit(
            "Phase 0 required grid is incomplete: "
            f"{len(coverage['phase0_required_grid_missing'])} cases missing"
        )


if __name__ == "__main__":
    main()
