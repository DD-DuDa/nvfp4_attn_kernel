"""Which structural parameters raise achieved FP4 bandwidth once softmax is free?

`probe_softmax_ceiling.py` showed that deleting per-element softmax arithmetic
moves the decode kernel from softmax-bound to bandwidth-bound, but only to
about 5.4 TB/s against a 6.5 TB/s streaming ceiling. This probe holds softmax
arithmetic at zero and sweeps the structural knobs that could close the rest:
Q-stage count, KV pipeline depth, epilogue staging, split count, N-block size,
and which of K or V is the expensive stream.

Every configuration is a runtime-patched subclass, so `src/` is untouched and
the numbers are wrong by construction. The `full_control` configuration keeps
the production softmax and no knobs, and must reproduce production.

The K/V asymmetry test pins one stream's page index to zero. That keeps every
TMA transaction, barrier, and MMA identical while making the pinned stream's
bytes L2-resident, so the time it gives back is that stream's DRAM cost.

Usage:
  CUTE_DSL_CACHE_ENABLED=0 PYTHONPATH=src:tests/kernel_profile \
      python tests/kernel_profile/probe_bandwidth_headroom.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr
from cutlass.cute.experimental import iket

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_decode as bd  # noqa: E402
import probe_softmax_ceiling as psc  # noqa: E402
from nvfp4_decode_kernel import _decode as decode_mod  # noqa: E402
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel  # noqa: E402


PAGE_SIZE = 128
FP4_BYTES_PER_TOKEN_HEAD = 2 * (128 // 2 + 128 // 16)  # 144
BF16_BYTES_PER_TOKEN_HEAD = 2 * 128 * 2  # 512

# Populated by BandwidthProbe._setup_attributes so the harness reports the
# staging the kernel actually compiled with rather than what was requested.
LAST_SETUP: dict[str, object] = {}


class BandwidthProbe(psc.ProbeKernel):
    """Softmax-free decode kernel with the structural knobs exposed.

    The knobs live on the class rather than the instance because `_decode`
    constructs the kernel itself; the harness builds a subclass carrying `_cfg`
    and then clears the compile caches so the next call retraces.
    """

    _cfg: dict[str, object] = {}

    def __init__(self, *args, **kwargs):
        cfg = type(self)._cfg
        n_block = int(cfg.get("n_block") or 0)
        # A tile wider than the default has to be installed after the base
        # constructor, because the base asserts that the two-stage TMEM layout
        # fits in 512 columns and a 256-wide S region does not.
        widen = n_block > PAGE_SIZE
        if n_block and not widen:
            kwargs["n_block_size"] = n_block
        super().__init__(*args, **kwargs)
        self.probe_pin_k = bool(cfg.get("pin_k"))
        self.probe_pin_v = bool(cfg.get("pin_v"))
        self.probe_pin_sf = bool(cfg.get("pin_sf"))
        if widen:
            self._widen_n_block(n_block)

    def _widen_n_block(self, n_block: int) -> None:
        """Fold the dead stage-1 S region into one wider S stage.

        Decode runs `softmax_loop` only for stage 0 when `q_stage == 1`, so the
        stage-1 S region is dead. Its columns can back a single S tile twice as
        wide: S takes [0, n_block), O follows at [n_block, n_block + 128), and
        the Q scale factors, which normally squat in the stage-1 S region, move
        above O. This reasons only about TMEM; whether the TMA and the MMA can
        be built for an N tile wider than the 128-token page is exactly what
        the experiment is meant to find out.
        """
        assert self.q_stage == 1, "a wide N tile needs the single-Q-stage layout"
        head_dim_v = self.head_dim_v_padded
        self.n_block_size = n_block
        self.cta_tiler = (self.q_stage * self.m_block_size, n_block, self.head_dim_padded)
        self.mma_tiler_qk = (self.m_block_size, n_block, self.head_dim_padded)
        self.mma_tiler_pv = (self.m_block_size, head_dim_v, n_block)
        self.tmem_o_offset = [n_block]
        self.tmem_s_offset = [0, n_block + head_dim_v]
        self.tmem_total = n_block + head_dim_v
        self.tmem_s_to_p_offset = n_block // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]
        self.tmem_p_bf16_offset = list(self.tmem_p_offset)
        self.tmem_vec_offset = self.tmem_s_offset
        assert self.tmem_s_offset[1] + self.tmem_s_to_p_offset <= 512, (
            "the relocated scale-factor region does not fit in TMEM"
        )

    def _setup_attributes(self):
        super()._setup_attributes()
        cfg = type(self)._cfg
        if cfg.get("epi_stage") is not None:
            self.epi_stage = int(cfg["epi_stage"])
        default_kv_stage = self.kv_stage
        if cfg.get("kv_stage") is not None:
            self.kv_stage = int(cfg["kv_stage"])
        LAST_SETUP.update(
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            default_kv_stage=default_kv_stage,
            epi_stage=self.epi_stage,
            n_block_size=self.n_block_size,
            m_block_size=self.m_block_size,
            tmem_total=self.tmem_o_offset[-1] + self.head_dim_v_padded,
        )

    @cute.jit
    def load_KV(
        self,
        tma_atom,
        tXgX,
        tXsX,
        paged_kv_manager,
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state,
        K_or_V: str,
        page_idx: Optional[Int32] = None,
        tma_atom_sf=None,
        tXgSF=None,
        tXsSF=None,
    ):
        """Production `load_KV` with an optional L2 pin on the page index.

        Pinning rewrites only the gmem coordinate. The TMA transaction count,
        the byte count declared to the mbarrier, and every consumer stay
        identical, so the difference against the unpinned run isolates the
        pinned stream's DRAM traffic.
        """
        assert K_or_V in ("K", "V")
        assert not self.uneven_kv_smem, "probe assumes the even KV smem layout"
        assert self.use_tma_KV, "probe covers the TMA paged path only"
        paged = const_expr(page_idx is not None)
        pin_data = const_expr(
            paged and (self.probe_pin_k if K_or_V == "K" else self.probe_pin_v)
        )
        pin_sf = const_expr(paged and self.probe_pin_sf)

        stage, phase = producer_state.index, producer_state.phase
        iket.range_push("load_wait_kv")
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        iket.range_pop()

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V],
            )
        tXsX_cur = tXsX[None, stage]
        if const_expr(not paged):
            tXgX_cur = tXgX[None, block]
        elif const_expr(pin_data):
            tXgX_cur = tXgX[None, 0, Int32(0)]
        else:
            tXgX_cur = tXgX[None, 0, page_idx]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

        if const_expr(tma_atom_sf is not None and tXgSF is not None and tXsSF is not None):
            tXsSF_cur = tXsSF[None, stage]
            if const_expr(not paged):
                tXgSF_cur = tXgSF[None, block]
            elif const_expr(pin_sf or pin_data):
                tXgSF_cur = tXgSF[None, 0, Int32(0)]
            else:
                tXgSF_cur = tXgSF[None, 0, page_idx]
            cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage)


_BASE_SPLIT_HEURISTIC = decode_mod.split_k_heuristic


def _clear_caches() -> None:
    decode_mod._decode_compile_cache.clear()
    decode_mod._split_decode_compile_cache.clear()


def apply_config(cfg: dict) -> None:
    """Install a probe class with `cfg`'s knobs and force a fresh compile."""
    psc._PROBE_MODE = cfg.get("softmax", "free")
    probe_cls = type(
        "BandwidthProbeConfigured",
        (BandwidthProbe,),
        {"_cfg": cfg, "_force_q_stage_1": bool(cfg.get("q_stage1", False))},
    )
    decode_mod.FP4DecodeKernel = probe_cls
    splits = cfg.get("splits")
    decode_mod.split_k_heuristic = (
        _BASE_SPLIT_HEURISTIC if splits is None else (lambda *a, **k: int(splits))
    )
    _clear_caches()


def restore_production() -> None:
    decode_mod.FP4DecodeKernel = FP4DecodeKernel
    decode_mod.split_k_heuristic = _BASE_SPLIT_HEURISTIC
    _clear_caches()


def kernel_ms(
    run: Callable[[], object], iters: int, warmup: int
) -> tuple[float, float, dict]:
    """Kernel-only milliseconds, split into decode and split-K combine."""
    breakdown = bd.measure_kernel_breakdown(run, iters, warmup)
    total = sum(breakdown.values())
    combine = sum(ms for name, ms in breakdown.items() if "combine" in name.lower())
    return total, total - combine, breakdown


_SMEM_RE = re.compile(r"Total shared memory used: ([0-9.]+) KB")


def measure_config(
    name: str,
    cfg: dict,
    run: Callable[[], object],
    iters: int,
    warmup: int,
    reference: Optional[torch.Tensor] = None,
) -> dict:
    LAST_SETUP.clear()
    apply_config(cfg)
    record: dict[str, object] = {"config": name, "cfg": dict(cfg)}
    captured = io.StringIO()
    try:
        # The kernel prints its shared-memory budget while tracing; capturing
        # it is the only way to see what staging actually compiled.
        with contextlib.redirect_stdout(captured):
            output = run()
            torch.cuda.synchronize()
        # Only the production-softmax configurations can be checked; the
        # softmax-free ones are wrong by construction.
        if reference is not None and cfg.get("softmax") == "full":
            record["cosine_vs_fa4"] = bd.cosine(
                output.reshape(reference.shape), reference
            )
    except Exception as error:  # noqa: BLE001 - a failed config is a result
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}".strip()
        record["traceback_tail"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-4000:]
        restore_production()
        return record
    record["status"] = "ok"
    record["smem_kb"] = [float(value) for value in _SMEM_RE.findall(captured.getvalue())]
    record["setup"] = dict(LAST_SETUP)
    total, decode, breakdown = kernel_ms(run, iters, warmup)
    record["kernel_ms"] = total
    record["decode_ms"] = decode
    record["breakdown"] = breakdown
    restore_production()
    return record


def run_case(
    case: bd.Case,
    device: torch.device,
    configs: dict[str, dict],
    iters: int,
    warmup: int,
    quantize_chunk_pages: int,
    checkpoint: Optional[Callable[[dict], None]] = None,
) -> dict:
    inputs = bd.build_inputs(case, device, quantize_chunk_pages=quantize_chunk_pages)
    run_fp4 = bd.make_fp4(inputs, hybrid=False, prequantized_query=True)

    tokens = case.batch * case.seqlen * case.heads_kv
    fp4_bytes = tokens * FP4_BYTES_PER_TOKEN_HEAD
    bf16_bytes = tokens * BF16_BYTES_PER_TOKEN_HEAD
    result: dict[str, object] = {
        "case": case.label,
        "batch": case.batch,
        "seqlen": case.seqlen,
        "heads_q": case.heads_q,
        "heads_kv": case.heads_kv,
        "fp4_bytes": fp4_bytes,
        "fa4_bytes": bf16_bytes,
    }

    reference = bd.make_fa4(inputs, 1)()
    reference = (reference[0] if isinstance(reference, tuple) else reference).reshape(
        case.batch, case.heads_q, 128
    )
    best_fa4 = float("inf")
    for splits in sorted({1, bd.fa4_auto_splits(case, device)}):
        total, _, _ = kernel_ms(bd.make_fa4(inputs, splits), iters, warmup)
        best_fa4 = min(best_fa4, total)
    result["fa4_ms"] = best_fa4
    result["fa4_tbps"] = bf16_bytes / (best_fa4 * 1e-3) / 1e12

    restore_production()
    total, _, _ = kernel_ms(run_fp4, iters, warmup)
    result["production_ms"] = total
    result["production_tbps"] = fp4_bytes / (total * 1e-3) / 1e12
    print(
        f"   {'fa4':<26} {best_fa4 * 1e3:9.1f} us"
        f"  {result['fa4_tbps']:6.2f} TB/s (bf16 bytes)",
        flush=True,
    )
    print(
        f"   {'production':<26} {total * 1e3:9.1f} us"
        f"  {result['production_tbps']:6.2f} TB/s",
        flush=True,
    )

    rows = []
    for name, cfg in configs.items():
        record = measure_config(name, cfg, run_fp4, iters, warmup, reference)
        if record["status"] == "ok":
            ms = record["kernel_ms"]
            record["tbps"] = fp4_bytes / (ms * 1e-3) / 1e12
            record["vs_fa4"] = best_fa4 / ms
            setup = record["setup"]
            cosine = record.get("cosine_vs_fa4")
            print(
                f"   {name:<26} {ms * 1e3:9.1f} us"
                f"  {record['tbps']:6.2f} TB/s"
                f"  {record['vs_fa4']:5.2f}x FA4"
                f"  q{setup.get('q_stage')}/kv{setup.get('kv_stage')}"
                f"/epi{setup.get('epi_stage')}/n{setup.get('n_block_size')}"
                f"  smem={record['smem_kb']}"
                + (f"  cos={cosine:.4f}" if cosine is not None else ""),
                flush=True,
            )
        else:
            print(f"   {name:<26} FAILED  {record['error'][:400]}", flush=True)
        rows.append(record)
        # A configuration that deadlocks the GPU takes the process with it, so
        # everything measured before it has to already be on disk.
        result["configs"] = rows
        if checkpoint is not None:
            checkpoint(result)
    result["configs"] = rows

    del inputs, run_fp4
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_config_name(name: str) -> dict:
    """Turn a knob-encoded name such as `free_q1_kv16_pinv` into a config.

    Tokens: `full`/`free` select the softmax mode, `q1` forces one Q stage,
    `kvN`/`epiN` set staging, `sN` fixes the split count, `nN` sets the N-block
    size, and `pink`/`pinv`/`pinsf`/`pinall` pin a stream's page into L2.
    """
    cfg: dict[str, object] = {}
    for token in name.split("_"):
        if token == "full":
            cfg["softmax"] = "full"
        elif token == "free":
            cfg["softmax"] = "free"
        elif token == "q1":
            cfg["q_stage1"] = True
        elif token.startswith("kv") and token[2:].isdigit():
            cfg["kv_stage"] = int(token[2:])
        elif token.startswith("epi") and token[3:].isdigit():
            cfg["epi_stage"] = int(token[3:])
        elif token.startswith("s") and token[1:].isdigit():
            cfg["splits"] = int(token[1:])
        elif token.startswith("n") and token[1:].isdigit():
            cfg["n_block"] = int(token[1:])
        elif token == "pink":
            cfg["pin_k"] = True
        elif token == "pinv":
            cfg["pin_v"] = True
        elif token == "pinsf":
            cfg["pin_sf"] = True
        elif token == "pinall":
            cfg.update(pin_k=True, pin_v=True, pin_sf=True)
        else:
            raise SystemExit(f"unknown token {token!r} in config name {name!r}")
    return cfg


ALL_CONFIGS: dict[str, dict] = {
    # Controls.
    "full_control": {"softmax": "full"},
    "free_baseline": {},
    # Q staging: decode packs at most heads_q/heads_kv real rows into the M
    # tile, so with q_stage=2 the second Q stage runs over an out-of-range tile.
    "free_qstage1": {"q_stage1": True},
    "full_qstage1": {"softmax": "full", "q_stage1": True},
    "full_q1_kv16": {"softmax": "full", "q_stage1": True, "kv_stage": 16},
    # KV pipeline depth around the default.
    "free_kv4": {"kv_stage": 4},
    "free_kv6": {"kv_stage": 6},
    "free_kv8": {"kv_stage": 8},
    "free_kv10": {"kv_stage": 10},
    "free_kv12": {"kv_stage": 12},
    "free_kv16": {"kv_stage": 16},
    "free_kv20": {"kv_stage": 20},
    # Epilogue staging frees 32 KB per dropped stage for more KV depth.
    "free_epi1": {"epi_stage": 1},
    "free_q1_kv16": {"q_stage1": True, "kv_stage": 16},
    "free_q1_epi1_kv18": {"q_stage1": True, "epi_stage": 1, "kv_stage": 18},
    "free_q1_epi1_kv20": {"q_stage1": True, "epi_stage": 1, "kv_stage": 20},
    # Split count.
    "free_split1": {"splits": 1},
    "free_split2": {"splits": 2},
    "free_split4": {"splits": 4},
    "free_split8": {"splits": 8},
    "free_split16": {"splits": 16},
    "free_q1_split2": {"q_stage1": True, "splits": 2},
    "free_q1_split4": {"q_stage1": True, "splits": 4},
    # N-block size. The page size is fixed at 128 by the contract, so anything
    # else has to reconcile the TMA box with the page extent.
    "free_nblock256": {"q_stage1": True, "n_block": 256},
    "free_nblock64": {"n_block": 64},
    # K/V asymmetry: pin one stream into L2 and see what its DRAM cost was.
    "free_pin_k": {"pin_k": True},
    "free_pin_v": {"pin_v": True},
    "free_pin_sf": {"pin_sf": True},
    "free_pin_all": {"pin_k": True, "pin_v": True, "pin_sf": True},
    "free_q1_pin_all": {"q_stage1": True, "pin_k": True, "pin_v": True, "pin_sf": True},
}


def build_configs(selection: Optional[list[str]]) -> dict[str, dict]:
    if not selection:
        return dict(ALL_CONFIGS)
    return {
        name: ALL_CONFIGS[name] if name in ALL_CONFIGS else parse_config_name(name)
        for name in selection
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--quantize-chunk-pages", type=int, default=4096)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--grid", type=str, default="32x65536")
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    configs = build_configs(args.configs.split(",") if args.configs else None)

    results = []
    for point in args.grid.split(","):
        batch, seqlen = point.split("x")
        case = bd.Case(
            batch=int(batch),
            seqlen=int(seqlen),
            heads_q=args.heads_q,
            heads_kv=args.heads_kv,
        )
        print(f"== {case.label}", flush=True)

        def checkpoint(partial: dict) -> None:
            if not args.out:
                return
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results + [partial], indent=2))

        result = run_case(
            case,
            device,
            configs,
            args.iters,
            args.warmup,
            args.quantize_chunk_pages,
            checkpoint=checkpoint,
        )
        results.append(result)
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2))

    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
