"""Probe: how much of FP4 decode is per-thread softmax arithmetic?

This measures the ceiling of the transposed-S idea without building it. The
transpose would cut each thread's softmax work from 128 elements to 4, so the
question it has to answer first is what the whole kernel would cost if that
work were free. Deleting phases answers that directly and cheaply.

The probe kernel is numerically wrong by construction. It keeps every TMEM
load, every P store, every mbarrier arrive and wait, and the whole MMA and TMA
pipeline; it only removes arithmetic. Nothing here is importable by production
code, and it patches the kernel class into ``_decode`` at runtime rather than
touching ``src/``.

Modes:
  full      identical to production, a control that must reproduce the baseline
  no_exp    drop exp2 and the row-sum accumulation
  no_pquant drop the group max, the groupwise rescale, and the FP4 convert
  free      drop all per-element softmax arithmetic; the ceiling

Usage:
  CUTE_DSL_CACHE_ENABLED=0 PYTHONPATH=src python tests/kernel_profile/probe_softmax_ceiling.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.experimental import iket

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
from nvfp4_decode_kernel import _decode as decode_mod  # noqa: E402
from nvfp4_decode_kernel._fa4.softmax import SoftmaxSm100  # noqa: E402
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel  # noqa: E402


MODES = ("full", "no_exp", "no_pquant", "free")

# Read by ProbeKernel.__init__; the harness sets it before each compile.
_PROBE_MODE = "full"


class ProbeKernel(FP4DecodeKernel):
    """Production kernel with selected softmax phases removed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = _PROBE_MODE
        if mode not in MODES:
            raise ValueError(f"unknown probe mode {mode!r}")
        self.probe_mode = mode
        self.probe_do_rowmax = mode in ("full", "no_exp", "no_pquant")
        self.probe_do_exp = mode in ("full", "no_pquant")
        self.probe_do_pquant = mode in ("full", "no_exp")

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tCtSFP: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))

        iket.range_push("sm_wait_s")
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        iket.range_pop()
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        cute.arch.fence_view_async_tmem_load()
        sfqk_stage = self.q_stage - 1 - stage
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)

        iket.range_push("sm_rowmax")
        if const_expr(self.probe_do_rowmax):
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        else:
            # The correction warp and the epilogue both read these, so they need
            # finite values even though the result is meaningless.
            row_max = Float32(0.0)
            acc_scale = Float32(0.0) if const_expr(is_first) else Float32(1.0)
            softmax.row_max[0] = Float32(0.0)
        iket.range_pop()

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        if const_expr(self.probe_do_rowmax):
            softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype),
            tSrS_t2r.layout,
        )

        if const_expr(not self.probe_do_pquant):
            # P still has to be stored so the PV MMA and its barriers keep their
            # shape; a constant fill is the cheapest defined value to store.
            tSrP_r2t_f32.fill(0.0)

        iket.range_push("sm_exp")
        if const_expr(self.probe_do_exp):
            softmax.apply_exp2_convert(
                tSrS_t2r,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,
            )
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        elif const_expr(is_first):
            softmax.row_sum[0] = Float32(1.0)
        iket.range_pop()

        iket.range_push("sm_pquant")
        if const_expr(self.probe_do_pquant):
            tSrPSF_f32 = softmax.compute_group_max(tSrS_t2r, sf_size=self.sf_vec_size)
            tSrPSF = cute.make_rmem_tensor(tSrPSF_f32.layout, cutlass.Float8E4M3FN)
            softmax.scale_groupwise(tSrS_t2r, tSrPSF_f32, sf_size=self.sf_vec_size)
            self._quant_fp4(tSrS_t2r, tSrPSF_f32, tSrP_r2t, tSrPSF)
            if const_expr(sSFP is not None):
                thread_idx = thr_tmem_load.thr_idx
                lane_id = thread_idx % 32
                warp_id = thread_idx // 32
                base_offset = lane_id * 16 + (warp_id % 4) * 4
                sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
                sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
                sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
                tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(4))
                cute.autovec_copy(tSrPSF_2d, sSFP_thread)
        iket.range_pop()

        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2]))):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(
            self.mbar_p_split(cute.size(tStP_r2t.shape[2])), cute.size(tStP_r2t.shape[2])
        ):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)

        iket.range_push("sm_wait_corr")
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )
        iket.range_pop()

        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1


def set_probe_mode(mode: str) -> None:
    global _PROBE_MODE
    _PROBE_MODE = mode
    decode_mod.FP4DecodeKernel = ProbeKernel
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def restore_production() -> None:
    decode_mod.FP4DecodeKernel = FP4DecodeKernel
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def fa4_reference_ms(
    inputs: bd.Inputs, device: torch.device, iters: int, warmup: int
) -> float:
    """Best of num_splits=1 and the FA4 heuristic, matching the D0 baseline."""
    best = float("inf")
    for splits in {1, bd.fa4_auto_splits(inputs.case, device)}:
        ms, _ = bd.measure_event_gpu_ms(bd.make_fa4(inputs, splits), iters, warmup)
        best = min(best, ms)
    return best


def run_case(
    case: bd.Case,
    device: torch.device,
    iters: int,
    warmup: int,
    quantize_chunk_pages: int,
) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=quantize_chunk_pages)
    run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)

    record: dict[str, object] = {"case": case.label}
    record["fa4_ms"] = fa4_reference_ms(inputs, device, iters, warmup)

    restore_production()
    prod_ms, prod_spread = bd.measure_event_gpu_ms(run_fp4, iters, warmup)
    record["production_ms"] = prod_ms
    record["production_spread"] = prod_spread

    for mode in MODES:
        set_probe_mode(mode)
        ms, spread = bd.measure_event_gpu_ms(run_fp4, iters, warmup)
        record[f"{mode}_ms"] = ms
        record[f"{mode}_spread"] = spread

    restore_production()
    del inputs, run_fp4
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--quantize-chunk-pages", type=int, default=4096)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument(
        "--grid",
        type=str,
        default="8x16384,32x16384,32x65536,128x16384",
        help="comma-separated batchxseqlen points",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    cases = []
    for point in args.grid.split(","):
        batch, seqlen = point.split("x")
        cases.append(
            bd.Case(
                batch=int(batch),
                seqlen=int(seqlen),
                heads_q=args.heads_q,
                heads_kv=args.heads_kv,
            )
        )

    rows = []
    for case in cases:
        print(f"== {case.label}", flush=True)
        record = run_case(
            case, device, args.iters, args.warmup, args.quantize_chunk_pages
        )
        rows.append(record)
        base = record["production_ms"]
        fa4 = record["fa4_ms"]
        print(
            f"   fa4 {fa4 * 1e3:9.1f} us   production {base * 1e3:9.1f} us"
            f"  ({base / fa4:.3f}x slower)",
            flush=True,
        )
        for mode in MODES:
            ms = record[f"{mode}_ms"]
            print(
                f"   {mode:<10} {ms * 1e3:9.1f} us"
                f"   speedup vs production {base / ms:5.3f}x"
                f"   vs fa4 {ms / fa4:5.3f}x"
                f"   spread {record[f'{mode}_spread'] * 100:4.1f}%",
                flush=True,
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
