# Why FP4 decode stalls at 1.4 TB/s

## Trustworthy baseline

Measured at commit `d565dec` with SM clocks locked to 1965 MHz and median-of-5
timing. Per-`(head config, seqlen)` batch geomean of FP4-Q against the D0
better-FA4 baseline; D3 requires `<= 0.5`.

| Head config | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|
| MHA 8:8 | 1.493 | 1.857 | 1.463 | 1.334 |
| GQA 32:8 | 1.465 | 1.831 | 1.459 | 1.326 |
| MQA 32:1 | 1.271 | 1.987 | 1.462 | 1.374 |

All-grid geomean **1.512x slower**, worst point 2.225x, 96 cases. Median
measurement spread is 1.5%, so differences below roughly 3% are not signal.

## The binding constraint is exp2 issue throughput

`MUFU.EX2` has run at 16 ops/clk/SM from A100 through B200; only B300 doubles
it. One `m128n128` S tile needs `128 * 128 = 16384` exp2, which is 1024 clocks,
or 555 ns at the locked 1.845 GHz. IKET measures `sm_exp` at 712 ns, so the
softmax is already at **78% of the hardware exp2 ceiling**. This is not an
implementation defect; it is the instruction wall that Hao AI Lab describes for
prefill, and it binds in decode too.

Full softmax cost per n_block from IKET, split by the markers added in this
phase:

| Range | Per n_block | Share of softmax lifetime |
|---|---:|---:|
| `sm_exp` | 712 ns | 34.0% |
| `sm_pquant` | 536 ns | 25.6% |
| `sm_wait_s` | 314 ns | 15.0% |
| `sm_rowmax` | 103 ns | 4.9% |

That per-CTA rate explains the observed bandwidth plateau directly. Each CTA
consumes 18.0 KB per n_block (FP4 K and V plus E4M3 scales) and spends 1351 ns
of softmax compute on it, which is 13.6 GB/s per CTA and **2.0 TB/s across 148
SMs**. Measured FP4 tops out near 1.4 TB/s, the remainder being waits and the
epilogue. FA4 reaches 6.4 TB/s at high batch because it is bandwidth bound;
we never get there.

## Almost all of that exp2 is spent on padding

Decode has one query row per head, so with `pack_gqa` the M tile carries
`heads_q / heads_kv` real rows out of 128. Every thread owns one M row and all
128 K columns, so the padded rows cost exactly as much as the real ones.

| Head config | Real rows | Useful exp2 | Useful time | Waste |
|---|---:|---:|---:|---:|
| MHA 8:8 | 1 | 128 | 4.3 ns | 128x |
| GQA 32:8 | 4 | 512 | 17.3 ns | 32x |
| MQA 32:1 | 32 | 4096 | 138.8 ns | 4x |

A natural experiment confirms this. MHA 8:8 and GQA 32:8 have the same KV head
count, so they launch the same CTAs over the same page blocks and differ only
in real query rows, 1 against 4:

| batch | seqlen | MHA us | GQA us | ratio |
|---:|---:|---:|---:|---:|
| 8 | 16384 | 122.1 | 115.1 | 0.942 |
| 32 | 65536 | 2106.4 | 2106.0 | 1.000 |
| 128 | 16384 | 1858.1 | 1855.7 | 0.999 |
| 128 | 65536 | 7269.0 | 7273.7 | 1.001 |

Four times the useful work in the same time. Runtime is set by the padded tile,
not by the query rows.

## What this rules out

To become bandwidth bound at 6 TB/s a CTA must finish an n_block in 455 ns.
`sm_exp` alone is 712 ns, so **no change to the P or PV number format can reach
that budget**. Two consequences:

1. The NVFP4 tensor core is worth almost nothing here. The MMA warp is busy
   15.8% of its lifetime and `mma_wait_p` is 76%; the arithmetic is 1.91 us in
   FP4 against 3.90 us in BF16 inside a 91 us kernel. We currently pay 25.6% of
   the softmax critical path to feed a unit that idles 86% of the time.
2. Hao AI Lab's NVFP4 QK with BF16 PV is still the right move, but it lands
   differently in decode. Their 1.39x comes from prefill, where the m128 tile is
   full and halving the QK MMA is a real saving. In decode the MMA is already
   free, so dropping P quantization only removes `sm_pquant`, taking softmax
   from 1351 ns to 815 ns. An ablation replacing the per-group max with a
   constant scale measured 1.05x at batch 32 to 128, 1.25x at batch 8 seqlen
   16384, and 1.38x at batch 8 seqlen 65536 — real, but parity with FA4 rather
   than 2x.

That ablation is not shippable as written: it fails seven `tests/kernel` cases
against the PyTorch oracle even though cosine against FA4 moved only from
0.9877 to 0.9862. It was reverted. Its value is as an upper bound on what
simplifying P quantization can buy.

## What is left

The 2x has to come from not issuing exp2 on padded rows. Each thread currently
holds one M row and 128 columns, so for GQA 124 of 128 threads spend the whole
tile on padding, and predicating them off does not reclaim issue slots inside a
warp. Redistributing a real row's columns across lanes is what cuts the issued
exp2 instruction count, and it is the only lever that moves softmax under the
bandwidth budget.

The constraint to design against is that TMEM binds M rows to datapaths: a
`tcgen05.ld` lane reads its own datapath. Different warps may read different
column ranges of the same datapaths, which is the mechanism BitDecoding uses
when it repartitions warps along N instead of M. Any such split makes a row's
max and sum span lanes, so it requires a cross-lane reduction in the same
round; BitDecoding's Table III records a variant that was 6.1x faster and
numerically wrong, which a performance-only gate would have accepted.
