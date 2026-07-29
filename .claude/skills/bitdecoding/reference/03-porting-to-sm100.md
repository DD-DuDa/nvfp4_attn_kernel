# Porting BitDecoding's Ideas to This Repository

BitDecoding targets SM80/SM89/SM90 with `ldmatrix` + `mma.m16n8k16` + `lop3` in CUDA C++.
This repository targets SM100 with native NVFP4 `tcgen05` MMA in CuTe DSL, and consumes a
KV cache that serving code has already quantized. The principles survive; several
mechanisms do not.

Every claim about this repository below was checked against `HEAD`. Claims about what
would happen after a change are marked as hypotheses, because they have not been measured.

---

## 1. Query transformation — directly applicable, and it names our bug

BitDecoding reshapes the query from `[1, (g_q, h_kv)]` to `[g_q, h_kv]`, forming a larger
Q tile without changing attention semantics. The motivation is exactly our situation:
`Q_len = 1` underfills Tensor Core fragments and yields poor warp occupancy.

What this repository does instead:

```python
# src/nvfp4_decode_kernel/_kernel.py:39-46
query_padded_bf16 = torch.zeros(rows, 128, query.shape[1], query.shape[2], ...)
```

We pad the query sequence axis to 128 and use `pack_gqa=True` (`_decode.py:319`). So we
pay for grouping *and* for padding: `pack_gqa` fills M with `seqlen_q x qhead_per_kvhead`,
and `seqlen_q` here is the padded 128 rather than the true 1. Grouping multiplies the
padding instead of replacing it.

BitDecoding's formulation is the same idea with `seqlen_q = 1`: the M extent should be
`g_q`, not `128 x g_q`. This is the published form of the fix.

## 2. Warp partitioning — the principle transfers, our kernel currently disagrees

BitDecoding: fix `W_m = 1`, spend warps on N, because decode M is tiny and low-bit shifts
the bottleneck to compute.

This repository (`fp4_decode_kernel.py:201-209`):

| Warps | Role |
|---|---|
| 0–3 | softmax stage 0 |
| 4–7 | softmax stage 1 |
| 8–11 | correction |
| 12 | MMA |
| 13 | epilogue |
| 14 | TMA load |
| 15 | empty |

The two softmax groups are selected by `stage = Int32(0 if warp_idx < softmax1_warp_ids[0]
else 1)` (`:2094`), and the stage axis extends M: `cta_tiler = (q_stage * m_block_size,
n_block_size, head_dim_padded)` (`:151`). So **eight of sixteen warps are partitioned
along the axis BitDecoding argues should carry exactly one warp** — and that axis is
127/128 padding per §1.

Two honest caveats before treating this as a bug:

- FA4's `q_stage = 2` is also a software-pipelining device (ping-pong to overlap softmax
  with MMA), not purely M-parallelism. Some of its value is independent of M extent.
- Under `tcgen05`, MMA is issued by a single warp into TMEM rather than being tiled across
  warps the way `mma.m16n8k16` is. "Warps along N" therefore does not map one-to-one; the
  SM100 analogue is how softmax and correction work is distributed over the N tile, not
  the MMA atom layout.

So the transferable claim is the *diagnosis*, not the patch: the current warp budget was
inherited from a kernel tuned for a different roofline, and one full warp is idle. Whether
redistributing helps is an IKET question — measure which role is the critical path before
redesigning.

## 3. Cooperative softmax — becomes mandatory if N-parallelism widens

If softmax work is spread further across N, a row's max is no longer owned by one warp and
the resulting P distribution stops matching the PV MMA's operand layout. BitDecoding's
answer is `sTMP` for cross-warp rowmax and `sAcc` to stage and reload P.

**Table III's middle row is the warning to carry into any such experiment**: widening N
without cooperative softmax was 6.1x faster and numerically wrong. A performance-only
gate would have accepted it. Any round that changes softmax warp distribution must run
the numerical gate in the same round.

Our kernel already stages P through shared memory for the PV MMA under `quant_pv`
(`smem_sSFP`, `:319`), so the `sAcc` half of the mechanism has an analogue. The `sTMP`
cross-warp reduction does not currently exist because rows are not split across warps.

## 4. Residual block sizing — the formula generalizes, ours is fixed by contract

BitDecoding sizes the high-precision residual as `N_r = P_n x W_n x R` so that every
Tensor Core fragment is fully populated, giving 128 for 4-bit and 256 for 2-bit.

This repository has a BF16 residual tail, but its granularity is fixed at
`PAGE_SIZE = 128` by the kernel contract (`CLAUDE.md`: page size and head dim are both
128), not derived from a fragment-filling argument. For NVFP4 the two happen to be
compatible, so there is nothing to fix — but the *reason* differs, and if the MMA
configuration ever changes, `128` stops being justified by anything.

Worth checking against the principle rather than assuming: a residual length that is not
a whole number of fragments wastes the MMA on the tail block, and our residual length is
per-row and runtime-varying (`seqused_residual`).

## 5. P re-quantization — a named stall to look for

The paper explicitly identifies this as Blackwell's replacement for the dequantization
stall:

```
P_f16 = softmax(Q_f4 K_f4^T)
O_f16 = Quant(P_f16) V_f4
```

Our kernel does exactly this whenever `quant_pv` is set. So we should expect a
serialization point between softmax and the PV MMA.

This is a concrete, pre-registered hypothesis for the first IKET trace: **if the softmax
or correction roles show a large wait immediately before the PV MMA, this is that stall.**
Instrument the boundary specifically (see `.claude/skills/iket-trace/`).

## 6. What does not transfer

| Technique | Why not |
|---|---|
| `lop3` / `75316420` remapping | The paper itself bypasses it on Blackwell — native NVFP4 MMA consumes packed 4-bit with no dequantization step |
| `ldmatrix`-induced packing in a fused Residual Kernel | SM100 uses TMA and hardware-mandated scale-factor layouts; also, our contract forbids quantizing K/V inside `fp4_decode` (`CLAUDE.md` rule 2) — serving code owns cache quantization |
| Hopper `STSM` + `wgmma_SS` | Hopper-specific workaround for WGMMA's shared-memory operand constraint |
| Channel-wise / tensor-wise scaling generality | Our contract fixes E2M1 values with E4M3 block scales |

## 7. The mirroring invariant still binds us

Even though we do not use `ldmatrix`-induced packing, the deeper invariant applies:
**the kernel that writes the quantized cache and the kernel that reads it are coupled
through the instruction configuration, not merely through a data format.**

Here that pair is `quantize_kv_kernel.py` (producer) and `fp4_decode_kernel.py`
(consumer), plus `quantize_q_kernel.py` for Q. `CLAUDE.md` rule 4 already says to preserve
kernel-native K/V and scale layouts and avoid materialized scale copies — that rule is
this invariant. Treat those files as one unit under change, and expect layout drift to
surface as wrong numbers rather than as an error.
