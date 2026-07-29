# Phase 3 — split-K attempt ledger

## Implemented experiment

A pure FP4 split core was wired with `is_split_kv=True`, non-persistent
`SingleTileScheduler`, FP32 partial O/LSE, and a namespace-adapted private
CuTe combine kernel. The path compiled and launched for fixed split counts.

## Numerical failure

For representative pure FP4-Q decode, combined split output did not meet the
existing numerical gate versus the unsplit core:

- split=2 cosine observed around 0.86–0.98 depending on LSE layout experiment
- existing gate is 0.99
- max absolute errors were otherwise small (~0.02), indicating an unresolved
  LSE/partial semantic or layout mismatch rather than a launch failure

Both tested LSE host layouts and direct combine wiring failed to restore the
required cosine. Relaxing the gate is prohibited. No split path was exposed in
the public API and no red numerical state is retained.

## Excluded branches

- Reusing the installed FA4 combine without namespace adaptation violates the
  standalone production dependency rule.
- The adapted combine compiled only after adding missing private helper
  semantics, but its result remained below the numerical gate.
- Reordering the partial LSE dimensions changed the error but did not satisfy
  the gate.

## §6.6 disposition

The experimental split implementation and vendored combine were removed. The
last green Phase 2 state is restored. Phase 3 is recorded as failed within the
budget and execution continues to the orthogonal host-sync Phase 4. The final
performance target is not weakened; Phase 5 must report the low-batch deficit
and stop if §8.1 is not achieved.
