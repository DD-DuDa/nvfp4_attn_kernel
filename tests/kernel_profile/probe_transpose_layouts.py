"""Check that swapping the QK operand roles does not move any bytes.

Transposing S makes K the A operand and Q the B operand. That is only free if
the A and B shared-memory layout builders agree for a square tile, because the
Q staging buffer is addressed by hand in ``quantize_Q_bf16_to_fp4`` and the K
pages are written by TMA against a layout the loader assumes. If the layouts
match, both keep working untouched; if they do not, every hand-computed offset
has to be re-derived.

Usage:
  PYTHONPATH=src python tests/kernel_profile/probe_transpose_layouts.py
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bh
from cutlass.cute.nvgpu import tcgen05

from nvfp4_decode_kernel._fa4.block_scaled_layout import (
    make_smem_layout_sfa,
    make_smem_layout_sfb,
)

FP4 = cutlass.Float4E2M1FN
K_MAJOR = cute.nvgpu.OperandMajorMode.K
TILER = (128, 128, 128)
SF_VEC = 16


@cute.jit
def report():
    mma = bh.make_blockscaled_trivial_tiled_mma(
        FP4, K_MAJOR, K_MAJOR, cutlass.Float8E4M3FN, SF_VEC, tcgen05.CtaGroup.ONE, TILER[:2]
    )
    a = bh.make_smem_layout_a(mma, TILER, FP4, 1)
    b = bh.make_smem_layout_b(mma, TILER, FP4, 1)
    cute.printf("operand A layout {}\n", a)
    cute.printf("operand B layout {}\n", b)
    sfa = make_smem_layout_sfa(mma, TILER, SF_VEC, 1, mma_tile_inst_k=1)
    sfb = make_smem_layout_sfb(mma, TILER, SF_VEC, 1, mma_tile_inst_k=1)
    cute.printf("scale A layout {}\n", sfa)
    cute.printf("scale B layout {}\n", sfb)


def main() -> int:
    cute.compile(report)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
