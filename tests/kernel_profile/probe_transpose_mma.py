"""Ask the compiler which transposed-attention MMA shapes actually exist.

The transposed softmax needs S stored as (kv, query) instead of (query, kv), so
the QK GEMM has to swap its operands to ``A=K, B=Q`` and the PV GEMM has to take
P back from shared memory instead of tensor memory. Both of those depend on
support that is easy to assume and expensive to assume wrongly, so this asks the
builder directly instead of reading it off a table:

  - does a block-scaled FP4 MMA accept A from SMEM (P is written by the softmax
    warps, and only the A operand may come from TMEM, so SMEM is the only way to
    hand a register-resident P to the tensor core), and
  - does it accept an MN-major A? The softmax registers hold P transposed, so an
    MN-major A is the one layout the warps can store without a shuffle, and it
    is also what keeps O untransposed and the epilogue untouched.

Usage:
  PYTHONPATH=src python tests/kernel_profile/probe_transpose_mma.py
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bh
from cutlass.cute.nvgpu import tcgen05

K = cute.nvgpu.OperandMajorMode.K
MN = cute.nvgpu.OperandMajorMode.MN
CG = tcgen05.CtaGroup.ONE
FP4 = cutlass.Float4E2M1FN
SF = cutlass.Float8E4M3FN

# (label, is_block_scaled, a_major, b_major, a_source)
CASES = [
    ("QK^T  scaled  A=K(smem) K-major   B=Q K-major", True, K, K, None),
    ("QK^T  scaled  A=K(smem) MN-major  B=Q K-major", True, MN, K, None),
    ("QK^T  scaled  A=K(smem) K-major   B=Q MN-major", True, K, MN, None),
    ("PV    scaled  A=P(smem) K-major   B=V K-major", True, K, K, tcgen05.OperandSource.SMEM),
    ("PV    scaled  A=P(smem) MN-major  B=V K-major", True, MN, K, tcgen05.OperandSource.SMEM),
    ("PV    scaled  A=P(tmem) MN-major  B=V K-major", True, MN, K, tcgen05.OperandSource.TMEM),
    ("PV    scaled  A=P(tmem) K-major   B=V K-major (today)", True, K, K, tcgen05.OperandSource.TMEM),
    ("PV    plain   A=P(smem) MN-major  B=V K-major", False, MN, K, tcgen05.OperandSource.SMEM),
]


def build(is_scaled, a_major, b_major, a_source):
    args = (FP4, a_major, b_major)
    if is_scaled:
        args += (SF, 16, CG, (128, 128))
    else:
        args += (cutlass.Float32, CG, (128, 128))
    if a_source is not None:
        args += (a_source,)
    make = bh.make_blockscaled_trivial_tiled_mma if is_scaled else bh.make_trivial_tiled_mma
    return make(*args)


@cute.jit
def probe_one(idx: cutlass.Constexpr[int]):
    _, is_scaled, a_major, b_major, a_source = CASES[idx]
    mma = build(is_scaled, a_major, b_major, a_source)
    cute.printf("{}", cute.size(mma.thr_id.shape))


def main() -> int:
    width = max(len(c[0]) for c in CASES)
    failures = 0
    for idx, case in enumerate(CASES):
        try:
            cute.compile(probe_one, idx)
            print(f"{case[0]:<{width}}  OK")
        except Exception as exc:  # the builder raises OpError for unsupported combos
            first = str(exc).strip().splitlines()
            detail = first[0] if first else type(exc).__name__
            print(f"{case[0]:<{width}}  UNSUPPORTED  {detail[:160]}")
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} supported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
