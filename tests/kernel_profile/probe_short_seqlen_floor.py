"""Probe: what is the fixed GPU-side cost of FP4 decode at short seqlen?

At `seqlen 1024` the decode kernel takes a flat ~41 us for every batch from 1
to 16 while FA4 takes 10-16 us, and ~28 us of that survives with all softmax
arithmetic removed. This probe attributes that floor.

It answers four questions with measurements rather than reasoning:

  breakdown  which CUDA kernels run, with their launch geometry, so the decode
             kernel and any combine kernel are separated by name and the grid
             is read off the trace instead of being inferred
  variants   A/B of the two structural choices the non-split path makes that
             the split path does not: `q_stage = 2` and the persistent tile
             scheduler; plus forced split counts at a shape where the
             heuristic declines to split
  seqsweep   does the floor move with seqlen at fixed batch, which separates a
             per-launch cost from per-CTA work that does not shrink

Nothing here is importable by production code. The variant kernels are
subclasses patched into `_decode` at runtime, so `src/` is untouched.

Usage:
  flock /tmp/nvfp4_gpu1.lock -c "CUTE_DSL_CACHE_ENABLED=0 \
    PYTHONPATH=src:tests/kernel_profile python \
    tests/kernel_profile/probe_short_seqlen_floor.py --mode breakdown"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr
from cutlass.cute.experimental import iket

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from nvfp4_decode_kernel import _decode as decode_mod  # noqa: E402
from nvfp4_decode_kernel._fa4 import utils as fa4_utils  # noqa: E402
from nvfp4_decode_kernel._fa4.pack_gqa import PackGQA  # noqa: E402
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel  # noqa: E402


PAGE_SIZE = 128


class QStage1Kernel(FP4DecodeKernel):
    """Non-split decode with one Q stage instead of two.

    `__init__` already honours `_force_q_stage_1`; the split path sets
    `q_stage = 1` unconditionally, so this makes the non-split path match it.
    """

    _force_q_stage_1 = True


class NonPersistentKernel(FP4DecodeKernel):
    """Non-split decode on `SingleTileScheduler` instead of the persistent one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_persistent = False


class QStage1NonPersistentKernel(FP4DecodeKernel):
    _force_q_stage_1 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_persistent = False


class EpilogueProbeKernel(FP4DecodeKernel):
    """Non-split decode whose output store touches only reachable rows.

    Production stages the whole 128x128 O tile into one 512-element per-thread
    fragment inside a single 32-thread warp that has already dropped to 24
    registers, then walks 64 row-pairs and predicates all but the first few
    away. Decode with `seqlen_q == 1` and PackGQA can only ever produce
    `qhead_per_kvhead` rows, all of them in stage 0, so both knobs below remove
    work that the existing predicate already discards. Numerics must not move.

    per_m_fragment  load smem->rmem one row-pair at a time instead of the whole
                    tile up front, which is what forces the spill
    row_limit       stop the row loop, and skip whole Q stages, past the last
                    row a decode can produce
    """

    per_m_fragment = True
    row_limit = True

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O,
        mbar_ptr: cute.Pointer,
        block_info,
        num_splits: int,
        SeqlenInfoCls,
        TileSchedulerCls,
        mOutIndices=None,
    ):
        assert not self.use_tma_O
        assert self.pack_gqa and self.seqlen_q_static_one
        assert not self.use_out_indices
        # seqlen_q is statically one, so PackGQA folds exactly this many rows.
        valid_rows = self.qhead_per_kvhead
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx, split_idx
                    ]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                tidx = cute.arch.thread_idx()[0] % (
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                )
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOsO = gmem_thr_copy_O.partition_S(sO)
                cO = cute.make_identity_tensor(
                    (self.m_block_size, self.head_dim_v_padded)
                )
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = fa4_utils.predicate_k(tOcO, limit=mO.shape[1])
                tOcO_row = tOcO[0, None, 0]
                threads_per_row = gmem_tiled_copy_O.layout_tv_tiled.shape[0][0]
                num_threads = gmem_tiled_copy_O.size
                rows_per_step = num_threads // threads_per_row
                steps = cute.size(tOsO.shape[1])
                if const_expr(self.row_limit):
                    steps = min(
                        steps, (valid_rows + rows_per_step - 1) // rows_per_step
                    )
                packer = PackGQA(
                    self.m_block_size,
                    self.head_dim_v_padded,
                    self.check_hdim_v_oob,
                    self.qhead_per_kvhead,
                )
                for stage in cutlass.range_constexpr(self.q_stage):
                    iket.range_push("epi_wait_corr")
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_corr_epi_full_offset + stage,
                        epi_consumer_phase,
                    )
                    iket.range_pop()
                    stage_is_live = not self.row_limit or (
                        stage * self.m_block_size < valid_rows
                    )
                    if const_expr(stage_is_live):
                        block = self.q_stage * m_block + stage
                        tPrOPtr = packer.compute_ptr(
                            mO_cur[None, 0],
                            tOcO_row,
                            tidx,
                            block,
                            threads_per_row,
                            num_threads,
                        )
                        if const_expr(not self.per_m_fragment):
                            tOrO_all = cute.make_fragment_like(
                                tOsO[None, None, None, 0], self.o_dtype
                            )
                            cute.autovec_copy(
                                tOsO[None, None, None, stage], tOrO_all
                            )
                        for m in cutlass.range_constexpr(steps):
                            o_ptr_i64 = fa4_utils.shuffle_sync(
                                tPrOPtr[m // threads_per_row],
                                m % threads_per_row,
                                width=threads_per_row,
                            )
                            o_gmem_ptr = cute.make_ptr(
                                mO.element_type,
                                o_ptr_i64,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            if const_expr(self.per_m_fragment):
                                tOrO = cute.make_fragment_like(
                                    tOsO[None, m, None, 0], self.o_dtype
                                )
                                cute.autovec_copy(tOsO[None, m, None, stage], tOrO)
                            else:
                                tOrO = tOrO_all[None, m, None]
                            if (
                                t0OcO[0, m, 0][0]
                                < valid_rows
                                - block * self.m_block_size
                                - tOcO_row[0][0]
                            ):
                                row_gmem = cute.make_tensor(
                                    o_gmem_ptr, (self.head_dim_v_padded,)
                                )
                                elems_per_load = cute.size(tOrO.shape[0][0])
                                row_copy = cute.tiled_divide(
                                    row_gmem, (elems_per_load,)
                                )
                                for k in cutlass.range_constexpr(
                                    cute.size(tOrO.shape[1])
                                ):
                                    ki = tOcO[0, 0, k][1] // elems_per_load
                                    cute.copy(
                                        gmem_thr_copy_O,
                                        tOrO[None, k],
                                        row_copy[None, ki],
                                        pred=tOpO[None, m, k]
                                        if const_expr(self.check_hdim_v_oob)
                                        else None,
                                    )
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                    )
                epi_consumer_phase ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


class EpilogueControlKernel(EpilogueProbeKernel):
    """Both knobs off: a control that must reproduce production."""

    per_m_fragment = False
    row_limit = False


class EpiloguePerMKernel(EpilogueProbeKernel):
    per_m_fragment = True
    row_limit = False


class EpilogueRowLimitKernel(EpilogueProbeKernel):
    per_m_fragment = False
    row_limit = True


class EpilogueFastQStage1Kernel(EpilogueProbeKernel):
    _force_q_stage_1 = True


KERNEL_CLASSES = {
    "prod": FP4DecodeKernel,
    "qstage1": QStage1Kernel,
    "nonpersistent": NonPersistentKernel,
    "qstage1_nonpersistent": QStage1NonPersistentKernel,
    "epi_control": EpilogueControlKernel,
    "epi_perm": EpiloguePerMKernel,
    "epi_rowlimit": EpilogueRowLimitKernel,
    "epi_fast": EpilogueProbeKernel,
    "epi_fast_qstage1": EpilogueFastQStage1Kernel,
}


def install(kernel_class=FP4DecodeKernel, forced_splits: int | None = None) -> None:
    """Patch the decode module and drop every compiled-kernel cache."""
    decode_mod.FP4DecodeKernel = kernel_class
    if forced_splits is None:
        decode_mod.split_k_heuristic = _ORIGINAL_HEURISTIC
    else:
        decode_mod.split_k_heuristic = lambda *a, **k: forced_splits
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


_ORIGINAL_HEURISTIC = decode_mod.split_k_heuristic


def restore() -> None:
    install()


def kernel_trace(
    run: Callable[[], object], iters: int, warmup: int
) -> list[dict]:
    """Per-kernel time and launch geometry, read from a chrome trace.

    `measure_kernel_breakdown` gives times but not grid shapes, and the grid is
    the whole question here, so this parses the exported trace instead.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as handle:
        prof.export_chrome_trace(handle.name)
        trace = json.loads(Path(handle.name).read_text())

    per_kernel: dict[str, dict] = {}
    for event in trace.get("traceEvents", []):
        if event.get("cat") not in ("kernel", "gpu_memset", "gpu_memcpy"):
            continue
        name = event["name"]
        args = event.get("args", {})
        entry = per_kernel.setdefault(
            name,
            {
                "name": name,
                "launches": 0,
                "total_us": 0.0,
                "grid": args.get("grid"),
                "block": args.get("block"),
                "registers": args.get("registers per thread"),
                "smem": args.get("shared memory"),
            },
        )
        entry["launches"] += 1
        entry["total_us"] += float(event.get("dur", 0.0))

    rows = []
    for entry in per_kernel.values():
        entry["us_per_iter"] = entry["total_us"] / iters
        entry["launches_per_iter"] = entry["launches"] / iters
        del entry["total_us"]
        rows.append(entry)
    rows.sort(key=lambda row: -row["us_per_iter"])
    return rows


def short_name(name: str) -> str:
    if "fp4_decode_kernel" in name or "FP4DecodeKernel" in name:
        return "fp4_decode"
    if "combine" in name.lower():
        return "split_k_combine"
    if "quantize_query" in name:
        return "quantize_query"
    if "flash" in name.lower() or "FlashAttention" in name:
        return "fa4"
    return name[:44]


def describe(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"{short_name(row['name'])} {row['us_per_iter']:.2f}us"
            f" grid={row['grid']} block={row['block']}"
            f" regs={row['registers']} smem={row['smem']}"
            f" x{row['launches_per_iter']:.0f}"
        )
    return "\n      ".join(parts)


def total_us(rows: list[dict]) -> float:
    return sum(row["us_per_iter"] for row in rows)


def cosine_vs_fa4(inputs: bd.Inputs, run: Callable[[], object]) -> float:
    case = inputs.case
    reference = bd._output(bd.make_fa4(inputs, num_splits=1)()).reshape(
        case.batch, case.heads_q, bd.HEAD_DIM
    )
    out = bd._output(run()).reshape(case.batch, case.heads_q, bd.HEAD_DIM)
    return bd.cosine(out, reference)


def parse_grid(text: str) -> list[bd.Case]:
    cases = []
    for point in text.split(","):
        batch, seqlen = point.split("x")
        cases.append(
            bd.Case(
                batch=int(batch),
                seqlen=int(seqlen),
                heads_q=ARGS.heads_q,
                heads_kv=ARGS.heads_kv,
            )
        )
    return cases


def mode_breakdown(device: torch.device) -> list[dict]:
    records = []
    restore()
    for case in parse_grid(ARGS.grid):
        inputs = bd.build_inputs(
            case, device, quantize_chunk_pages=ARGS.quantize_chunk_pages
        )
        heuristic_splits = _ORIGINAL_HEURISTIC(
            case.batch,
            case.heads_kv,
            case.seqlen // PAGE_SIZE,
            sms=torch.cuda.get_device_properties(device).multi_processor_count,
        )
        run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        fp4_rows = kernel_trace(run_fp4, ARGS.iters, ARGS.warmup)
        fa4_rows = kernel_trace(
            bd.make_fa4(inputs, bd.fa4_auto_splits(case, device)),
            ARGS.iters,
            ARGS.warmup,
        )
        fa4_single = kernel_trace(bd.make_fa4(inputs, 1), ARGS.iters, ARGS.warmup)
        record = {
            "case": case.label,
            "batch": case.batch,
            "seqlen": case.seqlen,
            "heuristic_splits": heuristic_splits,
            "pages_per_row": case.seqlen // PAGE_SIZE,
            "fp4_total_us": total_us(fp4_rows),
            "fp4_kernels": fp4_rows,
            "fa4_split_total_us": total_us(fa4_rows),
            "fa4_split_kernels": fa4_rows,
            "fa4_single_total_us": total_us(fa4_single),
            "fa4_single_kernels": fa4_single,
        }
        records.append(record)
        print(
            f"== {case.label}  heuristic_splits={heuristic_splits}\n"
            f"   fp4 {record['fp4_total_us']:7.2f} us\n"
            f"      {describe(fp4_rows)}\n"
            f"   fa4(splits={bd.fa4_auto_splits(case, device)})"
            f" {record['fa4_split_total_us']:7.2f} us\n"
            f"      {describe(fa4_rows)}\n"
            f"   fa4(splits=1) {record['fa4_single_total_us']:7.2f} us\n"
            f"      {describe(fa4_single)}",
            flush=True,
        )
        del inputs, run_fp4
        gc.collect()
        torch.cuda.empty_cache()
    return records


VARIANTS = (
    ("prod", "prod", None),
    ("qstage1", "qstage1", None),
    ("nonpersistent", "nonpersistent", None),
    ("qstage1_nonpersistent", "qstage1_nonpersistent", None),
    ("prod_split2", "prod", 2),
    ("prod_split4", "prod", 4),
    ("prod_split8", "prod", 8),
    ("epi_control", "epi_control", None),
    ("epi_perm", "epi_perm", None),
    ("epi_rowlimit", "epi_rowlimit", None),
    ("epi_fast", "epi_fast", None),
    ("epi_fast_qstage1", "epi_fast_qstage1", None),
    ("epi_fast_split2", "epi_fast", 2),
    ("epi_fast_split4", "epi_fast", 4),
    ("epi_fast_split8", "epi_fast", 8),
)


def mode_variants(device: torch.device) -> list[dict]:
    records = []
    wanted = set(ARGS.variants.split(",")) if ARGS.variants else None
    for case in parse_grid(ARGS.grid):
        inputs = bd.build_inputs(
            case, device, quantize_chunk_pages=ARGS.quantize_chunk_pages
        )
        run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        fa4_rows = kernel_trace(
            bd.make_fa4(inputs, bd.fa4_auto_splits(case, device)),
            ARGS.iters,
            ARGS.warmup,
        )
        fa4_us = total_us(fa4_rows)
        print(f"== {case.label}   fa4 {fa4_us:7.2f} us", flush=True)
        for label, kernel_name, splits in VARIANTS:
            if wanted is not None and label not in wanted:
                continue
            install(KERNEL_CLASSES[kernel_name], forced_splits=splits)
            try:
                cosine = cosine_vs_fa4(inputs, run_fp4)
                rows = kernel_trace(run_fp4, ARGS.iters, ARGS.warmup)
            except Exception as error:  # noqa: BLE001
                print(f"   {label:<24} FAILED {type(error).__name__}: {error}")
                records.append(
                    {
                        "case": case.label,
                        "variant": label,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                install()
                continue
            records.append(
                {
                    "case": case.label,
                    "batch": case.batch,
                    "seqlen": case.seqlen,
                    "variant": label,
                    "forced_splits": splits,
                    "cosine_vs_fa4": cosine,
                    "total_us": total_us(rows),
                    "vs_fa4": total_us(rows) / fa4_us,
                    "kernels": rows,
                }
            )
            print(
                f"   {label:<24} {total_us(rows):7.2f} us"
                f"  {total_us(rows) / fa4_us:5.2f}x fa4"
                f"  cos {cosine:.4f}\n"
                f"      {describe(rows)}",
                flush=True,
            )
            install()
        del inputs, run_fp4
        gc.collect()
        torch.cuda.empty_cache()
    return records


def mode_seqsweep(device: torch.device) -> list[dict]:
    records = []
    variants = (ARGS.variants or "prod").split(",")
    for case in parse_grid(ARGS.grid):
        inputs = bd.build_inputs(
            case, device, quantize_chunk_pages=ARGS.quantize_chunk_pages
        )
        run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        fa4_rows = kernel_trace(
            bd.make_fa4(inputs, bd.fa4_auto_splits(case, device)),
            ARGS.iters,
            ARGS.warmup,
        )
        line = [f"{case.label:<28} fa4 {total_us(fa4_rows):7.2f}"]
        entry = {
            "case": case.label,
            "batch": case.batch,
            "seqlen": case.seqlen,
            "pages_per_row": case.seqlen // PAGE_SIZE,
            "fa4_us": total_us(fa4_rows),
        }
        for name in variants:
            label, kernel_name, splits = next(
                item for item in VARIANTS if item[0] == name
            )
            install(KERNEL_CLASSES[kernel_name], forced_splits=splits)
            rows = kernel_trace(run_fp4, ARGS.iters, ARGS.warmup)
            entry[f"{label}_us"] = total_us(rows)
            entry[f"{label}_kernels"] = {
                short_name(row["name"]): row["us_per_iter"] for row in rows
            }
            entry[f"{label}_grid"] = {
                short_name(row["name"]): row["grid"] for row in rows
            }
            line.append(f"{label} {total_us(rows):7.2f}")
            install()
        records.append(entry)
        print("  ".join(line), flush=True)
        del inputs, run_fp4
        gc.collect()
        torch.cuda.empty_cache()
    return records


def mode_equivalence(device: torch.device) -> list[dict]:
    """Bit-compare a variant's output against production's.

    The epilogue probe claims to skip only stores the existing predicate
    already discards, so its output must be identical, not merely close.
    """
    records = []
    for case in parse_grid(ARGS.grid):
        inputs = bd.build_inputs(
            case, device, quantize_chunk_pages=ARGS.quantize_chunk_pages
        )
        run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        install()
        reference = bd._output(run_fp4()).clone()
        for label in (ARGS.variants or "epi_fast").split(","):
            _, kernel_name, splits = next(
                item for item in VARIANTS if item[0] == label
            )
            install(KERNEL_CLASSES[kernel_name], forced_splits=splits)
            candidate = bd._output(run_fp4())
            identical = bool(torch.equal(reference, candidate))
            max_diff = (reference - candidate).abs().max().item()
            records.append(
                {
                    "case": case.label,
                    "variant": label,
                    "bitwise_identical": identical,
                    "max_abs_diff": max_diff,
                }
            )
            print(
                f"{case.label:<28} {label:<18}"
                f" identical={identical} max_abs_diff={max_diff:.3e}",
                flush=True,
            )
            install()
        del inputs, run_fp4, reference
        gc.collect()
        torch.cuda.empty_cache()
    return records


def mode_iket(device: torch.device) -> list[dict]:
    """Launch one instrumented decode; timing here is meaningless by design."""
    case = parse_grid(ARGS.grid)[0]
    variant = (ARGS.variants or "prod").split(",")[0]
    label, kernel_name, splits = next(
        item for item in VARIANTS if item[0] == variant
    )
    inputs = bd.build_inputs(
        case, device, quantize_chunk_pages=ARGS.quantize_chunk_pages
    )
    install(KERNEL_CLASSES[kernel_name], forced_splits=splits)
    run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
    for _ in range(ARGS.iters):
        run_fp4()
    torch.cuda.synchronize()
    install()
    print(f"iket workload done: {case.label} variant={label}", flush=True)
    return []


MODES = {
    "breakdown": mode_breakdown,
    "variants": mode_variants,
    "seqsweep": mode_seqsweep,
    "equivalence": mode_equivalence,
    "iket": mode_iket,
}


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--grid", type=str, default="1x1024")
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--quantize-chunk-pages", type=int, default=4096)
    parser.add_argument("--out", type=str, default=None)
    ARGS = parser.parse_args()

    device = torch.device("cuda", ARGS.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    records = MODES[ARGS.mode](device)
    restore()
    if ARGS.out:
        Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
        Path(ARGS.out).write_text(json.dumps(records, indent=2) + "\n")
        print(f"wrote {ARGS.out}")


if __name__ == "__main__":
    main()
