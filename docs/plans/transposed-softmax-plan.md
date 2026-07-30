# Transposed softmax for FP4 decode

## Why the previous scheme was abandoned

Scheme 2 (repeat the query rows across the M tile so every softmax thread owns
a real row, then split the 128 KV columns among the copies) rests on a step the
hardware does not provide.

`tcgen05.ld.32x32b` has a warp read 32 tensor-memory lanes, one lane per thread,
so a thread receives exactly one row. Every `.16x*` variant reads 16 lanes with
32 threads, which is at most two threads per row (PTX ISA §9.7.17.2.3.1). There
is no load that hands different threads different *columns* of the same row.

So under replication all 32 lanes of a warp receive byte-identical data, and
splitting the columns among them requires each lane to index its 128-element
register fragment at a lane-dependent offset. A runtime index into a register
array spills to local memory; unrolling the 32 cases instead makes the warp walk
all 32 divergent blocks, which costs exactly what it saves.

Nor can more warps be brought in: the only warps that can read tensor-memory
lanes 0-31 are warps 0, 4, 8, 12, and those share `warp_id % 4`, hence one SMSP
and one MUFU pipe. Adding them adds no `exp2` throughput.

The measured 2.1x from `probe_qreplicate.py` came from each thread computing the
*first* four fragment elements, a compile-time slice. That models the arithmetic
volume correctly and the addressing not at all.

The reachable ceiling for scheme 2 is the 2x from a `.16x64b` load (the extra
row copies are redundant, so dropping half of them is free), which puts s16384
at roughly 0.71 against a 0.5 gate.

## What transposing changes

Store S transposed: rows are KV positions, columns are query rows.

Then thread `t` owns KV position `t` and needs only the `R = qhead_per_kvhead`
columns that carry real queries. Selecting those columns is a **warp-uniform**
choice — every lane wants columns `0..R-1` — so it is expressed in the tensor
memory address, not in a register index. Per-thread `exp2` drops from 128 to
`R` (4 for 32:8 GQA), which is the full 32x.

The padding does not disappear; it moves to the axis where it is free. The MMA
still computes a 128x128 tile, but the tensor cores were never the limit.

## Operand plumbing

Both GEMMs stay K-major on both operands, so no tensor changes layout.

**QK.** `S^T = K · Q^T`, so A is K and B is Q, with tiler
`(M=n_block, N=m_block, K=head_dim)` — the same 128/128/128 as today. K is
stored `[pages, 128 token, heads_kv, 64]`, which as an A operand `(M=token,
K=d)` is already d-contiguous; Q as a B operand `(N=query, K=d)` likewise. The
change is which tensor plays which role, which TMA atom builder is used
(`make_tiled_tma_atom_A` for K, `_B` for Q), and that SFK becomes the A scale
and SFQ the B scale.

**PV.** P must reach the tensor core from shared memory: `OperandSource` selects
the source of *the A operand only*, so a B operand can never come from tensor
memory, and the transposed P lives in registers. Keeping `O = P · V` with A
taken from SMEM leaves O as `(query, head_dim)` — untransposed — so the epilogue
and the output scatter are untouched.

A K-major A operand wants `n` contiguous, and thread `t` holds `n = t`, so the
nibbles for `n = t` and `n = t+1` share a byte across neighbouring lanes. One
`shfl` against `lane ^ 1` per query row merges them and even lanes store the
byte. That is `R` shuffles per KV block, against the 128 `exp2` it replaces.

Choosing K-major here rather than MN-major keeps the whole design on the one
configuration every FP4 GEMM already uses, instead of resting on a majorness
that only the builder has been observed to accept.

## Reductions

The row maximum is over KV, which is now spread across threads, so it becomes a
cross-thread reduction. The row sum is not: each thread accumulates its own
partial sum across every KV block and the sum is reduced **once per tile**,
because the running correction factor is uniform across threads and factors out
of the accumulation. Only the maximum is on the per-block critical path.

- Within a warp: 5 butterfly shuffles per query row.
- Across the 4 softmax warps: one named barrier and a shared-memory round trip,
  roughly 80-100 cycles per KV block against the ~1024 cycles of `exp2` it
  replaces.

The P scale groups are 16 consecutive KV positions, so a group is 16 lanes and
its maximum is 4 butterfly shuffles inside a warp — no cross-warp step.

Masking gets cheaper: validity depends only on the thread's own KV position, so
it is one scalar predicate per thread instead of a compare per element.

`row_max` and `row_sum` become `R`-element per-thread vectors holding identical
values in every thread, rather than one scalar for the thread's own row. The
correction factor is likewise shared, so a single thread publishes `R` floats to
`sScale` instead of 128 threads publishing one each, and the correction warps
can skip O rows at or beyond `R`.

## Sequencing

Each step keeps the suite green before the next begins.

1. Standalone probe: `S^T = K · Q^T` block-scaled, checked numerically against
   torch. Validates the operand swap, the scale-factor role swap, and the
   resulting tensor-memory layout before any kernel surgery.
2. Standalone probe: `O = P · V` with a block-scaled A read from SMEM.
3. Kernel path behind a `transpose_s` flag, non-split and no fused residual,
   with the flag defaulted off so the shipping path is untouched.
4. Numerical acceptance against the existing oracles, then the CUDA-graph gate.
5. Extend to the split path and the fused BF16 residual, or leave those on the
   untransposed path if they stay correct there.

Q replication is inert under this design — padding on the query axis is free
once it sits in columns — so `q_replicate` is forced to 1 when `transpose_s` is
on. It stays behind its off-by-default flag rather than being reverted, so the
comparison remains available.

## What could still sink it

- The cross-warp maximum is a hard serialization point for all four softmax
  warps once per KV block. If the pipeline cannot hide it, the gain shrinks. A
  lagged maximum (reuse the previous block's bound, which stays numerically
  valid as long as it is an upper bound) is the fallback, at the cost of some
  exponent headroom.
- The correction warps and the epilogue assume a per-thread row identity in
  several places beyond `sScale`; each has to be re-derived rather than
  patched.
