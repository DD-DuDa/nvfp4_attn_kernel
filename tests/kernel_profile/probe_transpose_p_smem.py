"""Check the byte view used to publish a transposed P tile into shared memory.

Under the transposed layout a softmax thread owns one KV position, so it holds
one FP4 nibble per query row and has to merge nibble pairs across neighbouring
lanes and store single bytes. The MMA reads the same region as an FP4 A operand
through a swizzled layout, so the byte view has to land on exactly the bytes
that layout addresses.

Two candidate byte views exist: the hand-built one the Q staging buffer already
uses (``cute.recast_ptr`` plus an explicit byte layout) and ``recast_tensor`` of
the FP4 tensor itself. This writes a distinct pattern through one and reads it
back through the other; agreement means either can be used and the hand-built
recipe is safe to mirror for P.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python \
      tests/kernel_profile/probe_transpose_p_smem.py
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bh
import torch
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

FP4 = cutlass.Float4E2M1FN
K_MAJOR = cute.nvgpu.OperandMajorMode.K
M = 128
K = 128
TILER = (M, K, K)
SF_VEC = 16
ROW_BYTES = K // 2
ATOM_K_BYTES = K // 4


@cute.kernel
def _roundtrip(mismatch: cute.Tensor, layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sP = smem.allocate_tensor(FP4, layout.outer, byte_alignment=1024, swizzle=layout.inner)

    manual = cute.make_tensor(
        cute.recast_ptr(sP.iterator, cute.make_swizzle(2, 4, 3), cutlass.Uint8),
        cute.make_layout(
            ((M, ATOM_K_BYTES), 1, K // (2 * ATOM_K_BYTES)),
            stride=((ROW_BYTES, 1), 0, ATOM_K_BYTES),
        ),
    )
    recast = cute.recast_tensor(sP, cutlass.Uint8)

    # Distinct value per cell, folded to a byte: rows differ in the low bits and
    # byte columns in the high bits, so any permutation between the two views
    # shows up as a mismatch.
    for row in cutlass.range_constexpr(M):
        if tidx < ROW_BYTES:
            kb = tidx
            manual[(row, kb % ATOM_K_BYTES), 0, kb // ATOM_K_BYTES] = cutlass.Uint8(
                (row * 7 + kb * 29) % 251
            )
    cute.arch.barrier()
    for row in cutlass.range_constexpr(M):
        if tidx < ROW_BYTES:
            kb = tidx
            want = cutlass.Uint8((row * 7 + kb * 29) % 251)
            got = recast[(row, kb * 2), 0, 0]
            if got != want:
                mismatch[0] = mismatch[0] + 1


@cute.jit
def _launch(mismatch: cute.Tensor):
    mma = bh.make_blockscaled_trivial_tiled_mma(
        FP4, K_MAJOR, K_MAJOR, cutlass.Float8E4M3FN, SF_VEC, tcgen05.CtaGroup.ONE, TILER[:2]
    )
    layout = bh.make_smem_layout_a(mma, TILER, FP4, 1)
    cute.printf("P smem layout {}\n", layout)
    sliced = cute.slice_(layout, (None, None, None, 0))
    cute.printf("P smem layout (stage 0) {}\n", sliced)
    _roundtrip(mismatch, sliced).launch(
        grid=[1, 1, 1], block=[128, 1, 1], smem=M * ROW_BYTES + 1024
    )


def main() -> int:
    torch.cuda.init()
    mismatch = torch.zeros(1, dtype=torch.int32, device="cuda")
    cute.compile(_launch, from_dlpack(mismatch))(from_dlpack(mismatch))
    torch.cuda.synchronize()
    count = int(mismatch.item())
    print(f"byte-view mismatches: {count}")
    return 0 if count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
