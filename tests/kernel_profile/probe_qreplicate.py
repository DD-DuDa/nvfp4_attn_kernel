"""Probe: what does replicating Q across the M tile buy?

Scheme 2. Today a decode CTA computes a 128x128 S tile of which only
``qhead_per_kvhead`` rows carry a real query, and softmax thread ``t`` owns row
``t`` and all 128 of its columns, so it runs 128 exp2 whatever the GQA ratio.
If instead the M tile holds ``128 / qhead_per_kvhead`` copies of each real query
row, every thread can be made responsible for a small column range of one real
row, and the per-thread arithmetic collapses from 128 elements to
``qhead_per_kvhead``.

That reshuffle needs the row reductions to become cross-thread. Grouping the
copies so that one warp holds one real row keeps them inside a warp, which is
why the probe reduces with width 32 rather than going through shared memory.

This measures the cost structure, not the answer. The kernel it builds is
numerically wrong by construction: it does the reduced arithmetic on the first
slice of each thread's registers rather than on the columns the real scheme
would assign, and it does not replicate Q, sum the partial O rows, or relocate
P out of the S region. Every TMEM load, P store, mbarrier and MMA is left
alone, so what changes is only the arithmetic volume. Nothing here is
importable by production code; it patches the kernel class into ``_decode`` at
runtime.

Two costs the real scheme carries that this probe does not model: replicating Q
into SMEM once per CTA, and reducing the 128 partial O rows down to
``qhead_per_kvhead`` real rows once per CTA. Both are per-CTA rather than per
n_block, so on long contexts they should be small, but they are not zero and
this probe will therefore read slightly optimistic.

Modes:
  full        identical to production, a control that must reproduce it
  free        all per-element softmax arithmetic deleted; the ceiling
  s2          scheme 2 arithmetic, P quantized on the reduced slice, P stored whole
  s2_pstore   also stores only the P words a thread would actually touch

Usage:
  CUDA_VISIBLE_DEVICES=1 CUTE_DSL_CACHE_ENABLED=0 \
      PYTHONPATH=src:tests/kernel_profile python \
      tests/kernel_profile/probe_qreplicate.py
"""

from __future__ import annotations

import argparse
import gc
import json
import operator
import statistics
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
from nvfp4_decode_kernel._fa4 import utils as fa4_utils  # noqa: E402
from nvfp4_decode_kernel._fa4.blackwell_helpers import (  # noqa: E402
    packed_float_to_e2m1,
    packed_float_to_ue4m3,
)
from nvfp4_decode_kernel._fa4.softmax import SoftmaxSm100  # noqa: E402
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel  # noqa: E402
from probe_graph_gate import LOCK_PATH, gpu_lock, graph_us  # noqa: E402

# A trailing `_q1` forces one Q stage. Decode has seqlen_q == 1, so the second
# stage loads an out-of-range Q and then runs a full softmax and PV MMA over
# zeros; the kernel already honours `_force_q_stage_1` but only the split path
# sets it. Pairing it with s2 answers whether the two levers stack.
MODES = ("full", "free", "s2", "s2_pstore", "full_q1", "free_q1", "s2_q1")

_PROBE_MODE = "full"


class ProbeKernel(FP4DecodeKernel):
    """Production kernel with the softmax arithmetic volume rewritten."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = _PROBE_MODE
        if mode not in MODES:
            raise ValueError(f"unknown probe mode {mode!r}")
        self.probe_mode = mode
        # One thread covers one real row's share of the 128 columns, so the
        # slice length equals the number of real rows in the M tile.
        self.probe_elems = max(2, int(self.qhead_per_kvhead))
        # A 16-element scale group is now split across this many threads.
        self.probe_sf_lanes = max(1, self.sf_vec_size // self.probe_elems)

    @cute.jit
    def _probe_softmax_slice(
        self,
        tSrS_t2r: cute.Tensor,
        softmax: SoftmaxSm100,
        tSrP_r2t_f32: cute.Tensor,
        is_first: bool,
    ) -> Tuple[Float32, Float32]:
        """Row max, exp2, row sum and P quantization on one reduced slice."""
        elems = const_expr(self.probe_elems)
        frag = cute.logical_divide(tSrS_t2r, cute.make_layout(elems))
        s_slice = frag[None, 0]

        iket.range_push("sm_rowmax")
        # Local max over the thread's columns, then across the warp holding the
        # copies of this real row. Width 32 is the whole warp because the copy
        # grouping puts one real row in one warp.
        local_max = softmax._compute_row_max(s_slice.load())
        block_max = fa4_utils.warp_reduce(local_max, cute.arch.fmax, width=32)
        if const_expr(is_first):
            row_max = block_max
            acc_scale = Float32(0.0)
        else:
            row_max_old = softmax.row_max[0]
            row_max = cute.arch.fmax(row_max_old, block_max)
            acc_scale = fa4_utils.exp2f((row_max_old - row_max) * softmax.scale_log2)
        softmax.row_max[0] = row_max
        iket.range_pop()

        softmax.scale_subtract_rowmax(s_slice, row_max)

        iket.range_push("sm_exp")
        softmax.apply_exp2_convert(s_slice, e2e=False)
        local_sum = softmax._compute_row_sum(s_slice.load())
        block_sum = fa4_utils.warp_reduce(local_sum, operator.add, width=32)
        if const_expr(is_first):
            softmax.row_sum[0] = block_sum
        else:
            softmax.row_sum[0] = softmax.row_sum[0] * acc_scale + block_sum
        iket.range_pop()

        iket.range_push("sm_pquant")
        # The scale group now spans several threads, so the group max needs a
        # narrow cross-lane reduce that production does not pay.
        group_max = softmax._compute_row_max(s_slice.load()) * Float32(1.0 / 6.0)
        group_max = fa4_utils.warp_reduce(
            group_max, cute.arch.fmax, width=const_expr(self.probe_sf_lanes)
        )
        inv = Float32(1.0) / cute.arch.fmax(group_max, 1e-20)
        for i in cutlass.range_constexpr(elems):
            s_slice[i] = s_slice[i] * inv
        # One scale byte and one packed FP4 word per thread. `_quant_fp4` cannot
        # be reused here: it assumes whole 16-element groups per thread.
        sf_view = cute.recast_tensor(
            cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float8E4M3FN), cute.Int32
        )
        sf_view[0] = packed_float_to_ue4m3(group_max, group_max, group_max, group_max)
        p_view = cute.recast_tensor(tSrP_r2t_f32, cute.Int32)
        p_view[0] = packed_float_to_e2m1(
            s_slice[0],
            s_slice[1],
            s_slice[elems - 2],
            s_slice[elems - 1],
            s_slice[0],
            s_slice[1],
            s_slice[elems - 2],
            s_slice[elems - 1],
        )
        iket.range_pop()
        return row_max, acc_scale

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

        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype),
            tSrS_t2r.layout,
        )

        if const_expr(self.probe_mode == "full"):
            iket.range_push("sm_rowmax")
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
            iket.range_pop()
        elif const_expr(self.probe_mode == "free"):
            tSrP_r2t_f32.fill(0.0)
            row_max = Float32(0.0)
            acc_scale = Float32(0.0) if const_expr(is_first) else Float32(1.0)
            softmax.row_max[0] = Float32(0.0)
            if const_expr(is_first):
                softmax.row_sum[0] = Float32(1.0)
        else:
            tSrP_r2t_f32.fill(0.0)
            row_max, acc_scale = self._probe_softmax_slice(
                tSrS_t2r, softmax, tSrP_r2t_f32, is_first
            )

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        if const_expr(self.probe_mode == "full"):
            softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
            iket.range_push("sm_exp")
            softmax.apply_exp2_convert(
                tSrS_t2r,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,
            )
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
            iket.range_pop()

            iket.range_push("sm_pquant")
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

        # Relocating P out of the S region would let a thread write only the
        # words it touched, because the rest would stay zero across blocks.
        p_tiles = cute.size(tStP_r2t.shape[2])
        first_split = self.mbar_p_split(p_tiles)
        if const_expr(self.probe_mode == "s2_pstore"):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, 0], tStP_r2t[None, None, 0])
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        else:
            for i in cutlass.range_constexpr(first_split):
                cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
            for i in cutlass.range_constexpr(first_split, p_tiles):
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
    one_q_stage = mode.endswith("_q1")
    _PROBE_MODE = mode[:-3] if one_q_stage else mode
    decode_mod.FP4DecodeKernel = (
        type("ProbeKernelQStage1", (ProbeKernel,), {"_force_q_stage_1": True})
        if one_q_stage
        else ProbeKernel
    )
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def restore_production() -> None:
    decode_mod.FP4DecodeKernel = FP4DecodeKernel
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def run_case(
    case: bd.Case, device, iters: int, warmup: int, repeats: int, modes: list[str]
) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=4096)
    record: dict[str, object] = {"case": case.label, "batch": case.batch, "seqlen": case.seqlen}

    fa4_runs = [
        bd.make_fa4(inputs, splits) for splits in {1, bd.fa4_auto_splits(case, device)}
    ]
    for run in fa4_runs:
        run()

    restore_production()
    prod_run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
    prod_run()

    # Compilation stays outside the lock; only replay needs the GPU to itself.
    compiled: list[tuple[str, object]] = []
    for mode in modes:
        set_probe_mode(mode)
        run = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)
        try:
            run()
        except Exception as failure:
            record[f"{mode}_error"] = f"{type(failure).__name__}: {failure}"
            continue
        compiled.append((mode, run))
    probe_by_mode = dict(compiled)

    with gpu_lock(LOCK_PATH):
        record["fa4_us"] = min(
            graph_us(run, warmup, iters, repeats)[0] for run in fa4_runs
        )
        restore_production()
        record["production_us"] = graph_us(prod_run, warmup, iters, repeats)[0]
        for mode, run in compiled:
            set_probe_mode(mode)
            value, error = graph_us(run, warmup, iters, repeats)
            record[f"{mode}_us"] = value
            if error:
                record[f"{mode}_error"] = error
        restore_production()

    del inputs, prod_run, compiled, probe_by_mode
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
    parser.add_argument("--grid", type=str, default="4x16384,32x16384,32x65536,64x4096")
    parser.add_argument("--modes", type=str, default=",".join(MODES))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    modes = [mode for mode in args.modes.split(",") if mode]

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
        record = run_case(case, device, args.iters, args.warmup, args.repeats, modes)
        rows.append(record)

        prod = record["production_us"]
        fa4 = record["fa4_us"]
        print(
            f"   fa4 {fa4:8.1f} us   production {prod:8.1f} us  ({prod / fa4:.3f}x vs fa4)",
            flush=True,
        )
        for mode in modes:
            if f"{mode}_us" not in record:
                print(f"   {mode:<11} {str(record.get(f'{mode}_error'))[:70]}", flush=True)
                continue
            value = record[mode + "_us"]
            print(
                f"   {mode:<11} {value:8.1f} us"
                f"   {prod / value:5.3f}x vs production"
                f"   {value / fa4:5.3f}x vs fa4",
                flush=True,
            )
        if "free_us" in record and "s2_us" in record:
            span = prod - record["free_us"]
            if span > 0:
                captured = (prod - record["s2_us"]) / span
                record["s2_capture_of_ceiling"] = captured
                print(f"   s2 captures {captured * 100:5.1f}% of the ceiling", flush=True)

    ratios = {
        mode: [r[f"{mode}_us"] / r["fa4_us"] for r in rows if f"{mode}_us" in r]
        for mode in ("production", *modes)
        if mode != "production"
    }
    ratios["production"] = [r["production_us"] / r["fa4_us"] for r in rows]
    print("\ngeometric mean vs fa4 across the grid (lower is better):")
    for mode, values in ratios.items():
        if values:
            print(f"  {mode:<11} {statistics.geometric_mean(values):6.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
