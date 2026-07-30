# What FP4 decode would cost if softmax arithmetic were free

The transposed-S layout would cut each thread's softmax work from 128 elements
to 4. Before building it, this probe measures what the whole kernel costs when
that work goes to zero, which bounds what the transpose can buy.

## Method

`tests/kernel_profile/probe_softmax_ceiling.py` subclasses the production
kernel and overrides `softmax_step` to drop selected phases. It keeps every
TMEM load, every P store, every mbarrier arrive and wait, and the entire MMA
and TMA pipeline; only arithmetic is removed. The class is patched into
`_decode` at runtime, so `src/` is untouched and the results are wrong by
construction.

Modes: `full` (control), `no_exp` (drop exp2 and the row-sum), `no_pquant`
(drop the group max, the groupwise rescale, and the FP4 convert), `free` (drop
all per-element softmax arithmetic).

The control reproduces production to 0.998–1.012x across every case, so the
copied `softmax_step` is equivalent and the deltas are attributable.

Measured at commit `2257ab8`, SM clocks locked, GQA 32:8, on the same grid the
D3 gate uses. Times below are kernel-only, summed from the Torch profiler, so
host dispatch is excluded from all three columns.

## The ceiling

Per-`seqlen` geometric mean over batch, against the D0 better-FA4 baseline:

| seqlen | production vs FA4 | `free` vs FA4 | clears the 2x gate |
|---:|---:|---:|---|
| 1024 | 2.603x slower | 1.930x slower | no, by a wide margin |
| 4096 | 1.436x slower | 1.41x faster | no |
| 16384 | 1.321x slower | 2.19x faster | yes |
| 65536 | 1.353x slower | 2.37x faster | yes |

So the transpose can clear the gate at 16384 and 65536 and cannot at 1024 and
4096, no matter how well it is implemented. At 1024 the kernel is still almost
2x slower than FA4 with softmax entirely removed, which means the short-seqlen
deficit is a different problem.

At `seqlen 65536, batch 32` the `free` kernel runs 451 us against production's
2106 us and FA4's 1323 us. That is 5.36 TB/s of FP4 KV traffic against FA4's
6.49 TB/s of BF16 traffic, so `free` has crossed over into bandwidth-bound
territory but does not fully realize the 3.56x byte reduction.

## Partial fixes buy almost nothing

At `seqlen 16384`:

| batch | production | `no_exp` | `no_pquant` | `free` |
|---:|---:|---:|---:|---:|
| 16 | 1.000x | 1.530x | 1.739x | 2.268x |
| 32 | 1.000x | 1.561x | 1.487x | 3.869x |
| 64 | 1.000x | 1.592x | 1.517x | 4.158x |
| 128 | 1.000x | 1.611x | 1.531x | 4.062x |

Removing either phase alone lands near 1.5x; removing both plus the row max
jumps to 2.3–4.2x. Softmax sets the pace as long as it is slower than the
memory pipeline, and only stops mattering once it drops below it. A design that
removes half the softmax cost stays on the wrong side of that threshold, which
is why the earlier constant-scale-factor ablation measured only 1.05–1.38x.

## Two bottlenecks the probe surfaced that are not softmax

**Host dispatch dominates at low batch.** At `batch 1, seqlen 16384` the
CUDA-event measurement is 92.4 us while the GPU kernels total 24.8 us; 67.6 us
is the GPU idling on the host. The gap is 38.4 us at batch 4 and under 3 us
from batch 16 up. The event-timed all-grid baseline of 1.512x slower therefore
folds host overhead into a kernel number: the kernel-only geometric mean is
1.32x.

**A GPU-side floor at short seqlen.** At `seqlen 1024` the kernel takes 41 us
for every batch from 1 to 16, and 28 us of that survives with softmax removed,
against 10 us for FA4. Nothing about softmax explains it.

## Consequence for the plan

The transpose is worth building for long sequences and is not sufficient on its
own for the per-seqlen 2x gate. Reaching that gate needs the short-seqlen floor
and the host dispatch cost addressed as separate work items.

Raw data: `softmax-ceiling-gqa.json`, `softmax-ceiling-batchsweep-16384.json`,
`kernel-only-ceiling.json`.
