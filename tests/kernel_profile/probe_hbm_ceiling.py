"""What HBM read bandwidth is actually reachable for the FP4 decode footprint?

The decode kernel's ceiling cannot be inferred from FA4 alone: FA4 moves BF16
K/V in a different layout and may itself be short of the hardware limit. This
script measures streaming reads directly, first on a plain buffer and then on
tensors with the exact shapes and strides the decode kernel reads:

  FP4 K       [pages, 128, heads_kv, 64]  -> a CTA reads 64 contiguous bytes
                                            every heads_kv*64 bytes
  FP4 V       [pages, heads_kv, 128, 64]  -> a CTA reads 8192 contiguous bytes
                                            every heads_kv*8192 bytes
  E4M3 scales [pages, 32, 4, 1, 4, 2, hkv] -> 1024 contiguous bytes per page
                                            and head

Reductions are used rather than copies so the measurement is read-only. The
bytes are reinterpreted as float32; the values are meaningless, which does not
matter because the reduction is memory bound either way.

Usage:
  CUTE_DSL_CACHE_ENABLED=0 PYTHONPATH=src:tests/kernel_profile \
      python tests/kernel_profile/probe_hbm_ceiling.py
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Callable

import torch


PAGE_SIZE = 128
HEAD_DIM = 128


def time_ms(fn: Callable[[], object], iters: int, warmup: int, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    samples.sort()
    return samples[len(samples) // 2]


def tbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e12


def triton_stream(device: torch.device, gib: float, iters: int, warmup: int) -> list[dict]:
    """Vectorized streaming read, which torch's reduction does not saturate.

    `torch.sum` tops out well below the hardware limit on this card, so the
    ceiling it reports is a property of that kernel rather than of HBM. A
    hand-written load-and-reduce with 128-bit accesses and one program per
    tile is the standard way to get the real number. Reading two buffers at
    once is also measured because the decode kernel streams K and V together.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return [{"pattern": "triton_unavailable", "gib": 0.0, "ms": 0.0, "tbps": 0.0}]

    @triton.jit
    def _read(src, dst, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        # A multi-GiB float32 buffer has more than 2**31 elements, so the
        # offsets have to be 64-bit or they wrap into unmapped memory.
        offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        value = tl.load(src + offsets, mask=offsets < n_elements, other=0.0)
        tl.store(dst + pid, tl.sum(value))

    rows: list[dict] = []
    for block in (2048, 4096, 8192):
        elems = int(gib * 2**30) // 4
        buffer = torch.randn(elems, device=device, dtype=torch.float32)
        programs = triton.cdiv(elems, block)
        sink = torch.empty(programs, device=device, dtype=torch.float32)
        nbytes = elems * 4
        ms = time_ms(
            lambda: _read[(programs,)](buffer, sink, elems, BLOCK=block),
            iters,
            warmup,
        )
        rows.append(
            {
                "pattern": f"triton_stream_read_block{block}",
                "gib": nbytes / 2**30,
                "ms": ms,
                "tbps": tbps(nbytes, ms),
            }
        )
        del buffer, sink
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def plain_stream(device: torch.device, gib: float, iters: int, warmup: int) -> dict:
    """Contiguous read of one large buffer: the best case for this machine."""
    elems = int(gib * 2**30) // 4
    buffer = torch.randn(elems, device=device, dtype=torch.float32)
    nbytes = buffer.numel() * 4
    ms = time_ms(lambda: torch.sum(buffer), iters, warmup)
    result = {
        "pattern": "contiguous_f32_sum",
        "gib": nbytes / 2**30,
        "ms": ms,
        "tbps": tbps(nbytes, ms),
    }
    del buffer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def two_stream(device: torch.device, gib: float, iters: int, warmup: int) -> dict:
    """Two interleaved contiguous streams, as K and V are two separate reads."""
    elems = int(gib * 2**30) // 8
    a = torch.randn(elems, device=device, dtype=torch.float32)
    b = torch.randn(elems, device=device, dtype=torch.float32)
    nbytes = (a.numel() + b.numel()) * 4

    def run():
        torch.sum(a)
        torch.sum(b)

    ms = time_ms(run, iters, warmup)
    result = {
        "pattern": "two_contiguous_f32_sums",
        "gib": nbytes / 2**30,
        "ms": ms,
        "tbps": tbps(nbytes, ms),
    }
    del a, b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _measure(name: str, tensor: torch.Tensor, iters: int, warmup: int) -> dict:
    nbytes = tensor.numel() * tensor.element_size()
    ms = time_ms(lambda: torch.sum(tensor), iters, warmup)
    return {
        "pattern": name,
        "gib": nbytes / 2**30,
        "ms": ms,
        "tbps": tbps(nbytes, ms),
    }


def kv_layout_patterns(
    device: torch.device,
    pages: int,
    heads_kv: int,
    iters: int,
    warmup: int,
) -> list[dict]:
    """Read the real K and V page layouts, whole and one head at a time.

    ``heads_kv * 64`` bytes is the K row pitch, so a single head's slice takes
    64 bytes out of every pitch. Reading one head measures what that slice
    costs in isolation; reading all heads measures what it costs when the
    sibling CTAs' bytes are also wanted, which is the situation in the kernel.
    Each single-head read is paired with a contiguous read of exactly the same
    byte count, so the strided cost is separated from the effect of the smaller
    footprint.
    """
    rows: list[dict] = []
    # FP4 K page layout, expressed in float32 words so the reduction is fast.
    # 64 packed FP4 bytes per row and head is 16 float32 words.
    k_pages = torch.randn(
        pages, PAGE_SIZE, heads_kv, HEAD_DIM // 8, device=device, dtype=torch.float32
    )
    rows.append(_measure("fp4_k_all_heads", k_pages, iters, warmup))
    rows.append(
        _measure(
            "fp4_k_one_head_64B_of_%dB" % (heads_kv * 64),
            k_pages[:, :, 0, :],
            iters,
            warmup,
        )
    )
    rows.append(
        _measure(
            "control_contiguous_same_bytes_as_one_head",
            k_pages.view(-1)[: k_pages.numel() // heads_kv],
            iters,
            warmup,
        )
    )
    del k_pages
    gc.collect()
    torch.cuda.empty_cache()

    # FP4 V page layout: head-major, so one head's page is fully contiguous.
    v_pages = torch.randn(
        pages, heads_kv, HEAD_DIM, PAGE_SIZE // 8, device=device, dtype=torch.float32
    )
    rows.append(_measure("fp4_v_all_heads", v_pages, iters, warmup))
    rows.append(
        _measure(
            "fp4_v_one_head_8KB_of_%dKB" % (heads_kv * 8),
            v_pages[:, 0, :, :],
            iters,
            warmup,
        )
    )
    del v_pages
    gc.collect()
    torch.cuda.empty_cache()

    # E4M3 scale slab: 1024 contiguous bytes per page and head, heads packed
    # back to back inside a page.
    sf = torch.randn(pages, heads_kv, 256, device=device, dtype=torch.float32)
    rows.append(_measure("scales_all_heads", sf, iters, warmup))
    rows.append(
        _measure(
            "scales_one_head_1KB_of_%dKB" % heads_kv, sf[:, 0, :], iters, warmup
        )
    )
    rows.append(
        _measure(
            "control_contiguous_same_bytes_as_scales_one_head",
            sf.view(-1)[: sf.numel() // heads_kv],
            iters,
            warmup,
        )
    )
    del sf
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def full_footprint(
    device: torch.device,
    pages: int,
    heads_kv: int,
    iters: int,
    warmup: int,
) -> dict:
    """All four tensors the decode kernel reads, at the real byte ratio."""
    k_pages = torch.randn(
        pages, PAGE_SIZE, heads_kv, HEAD_DIM // 8, device=device, dtype=torch.float32
    )
    v_pages = torch.randn(
        pages, heads_kv, HEAD_DIM, PAGE_SIZE // 8, device=device, dtype=torch.float32
    )
    k_sf = torch.randn(pages, heads_kv, 256, device=device, dtype=torch.float32)
    v_sf = torch.randn(pages, heads_kv, 256, device=device, dtype=torch.float32)
    nbytes = 4 * (k_pages.numel() + v_pages.numel() + k_sf.numel() + v_sf.numel())

    def run():
        torch.sum(k_pages)
        torch.sum(v_pages)
        torch.sum(k_sf)
        torch.sum(v_sf)

    ms = time_ms(run, iters, warmup)
    result = {
        "pattern": "decode_footprint_all_four_tensors",
        "gib": nbytes / 2**30,
        "ms": ms,
        "tbps": tbps(nbytes, ms),
    }
    del k_pages, v_pages, k_sf, v_sf
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gib", type=float, default=8.0)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument(
        "--pages",
        type=int,
        default=16384,
        help="batch 32 x seqlen 65536 is 16384 pages",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(f"device: {props.name}  SMs={props.multi_processor_count}")
    print(f"memory: {props.total_memory / 2**30:.1f} GiB")
    print()

    rows: list[dict] = []
    rows.extend(triton_stream(device, args.gib, args.iters, args.warmup))
    rows.append(plain_stream(device, args.gib, args.iters, args.warmup))
    rows.append(two_stream(device, args.gib, args.iters, args.warmup))
    rows.extend(
        kv_layout_patterns(device, args.pages, args.heads_kv, args.iters, args.warmup)
    )
    rows.append(full_footprint(device, args.pages, args.heads_kv, args.iters, args.warmup))

    print(f"{'pattern':<40} {'GiB':>8} {'us':>10} {'TB/s':>8}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['pattern']:<40} {row['gib']:8.3f}"
            f" {row['ms'] * 1e3:10.1f} {row['tbps']:8.2f}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
