# Phase 2b — Warp rebalancing attempts

## Baseline evidence

The Phase 2 high-batch GQA trace satisfies the entry condition: softmax0 owns
the launch-tail envelope, correction spends 89.8% waiting for softmax, MMA
79.4% waiting for P/O readiness, and epilogue 99.6% waiting for correction.

## Attempt 1: assign warp 15 to the load role

Changing `load_warp_ids` from `(14,)` to `(14,15)` compiled, but the first
exact FP4-Q correctness test caused an unspecified CUDA launch failure. The
load pipeline's producer-count/barrier and work partition are not safely
expanded by changing the role tuple alone. The branch was reverted immediately;
no red state was committed.

## Attempt 2: assign warp 15 to epilogue

Changing `epilogue_warp_ids` from `(13,)` to `(13,15)` passed the focused exact
MHA/GQA/MQA equality test. A performance sweep then stalled for over ten minutes
on the first MHA cases, far outside the normal sub-minute sweep time. This
indicates a synchronization/work-partition pathology despite focused numerical
correctness. The run was terminated and the branch reverted; no timing claim
or commit was produced.

## Cooperative softmax branch

Not attempted within the remaining Phase budget. The observed barrier chain
would require a coordinated redesign of row ownership, cross-warp max/sum, P
staging and barrier counts. Per the plan's fast-but-wrong warning, a partial
softmax repartition is not acceptable and was not started without enough budget
to complete and numerically gate it.

## §6.6 disposition

Phase 2b did not beat its gate. The final green state remains the Phase 2
implementation. Both attempted warp-15 reassignments and their failure modes are
recorded, and the task proceeds to the orthogonal split-K Phase 3 without
relaxing any numerical/performance gate. The MHA Phase 2 regression remains
carried forward.
