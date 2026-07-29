# IKET API Reference

Import: `from cutlass.cute.experimental import iket`.
All calls belong inside `@cute.kernel` code. Calls in host-side `@cute.jit` wrappers or
plain Python emit nothing.

## Concepts

- **Event** — one warp-level runtime record: a timestamp plus identifying metadata.
- **Marker** — a point annotation; emits one event.
- **Range** — a duration, normally two events. Each event may carry one payload.

Events are warp-level, not thread-level.

## The seven calls

| Call | Purpose | Notes |
|---|---|---|
| `mark(name)` | Point annotation | One event |
| `mark(name, payload)` | Point annotation with a value | |
| `range_push(name)` | Open a stack range | Closed by the next `range_pop()` in LIFO order |
| `range_push(name, payload)` | Same, with a value | |
| `range_pop()` | Close the innermost stack range | |
| `range_start(name)` | Open a token range | Returns a token; closed by `range_end(token)` |
| `range_start(name, payload)` | Same, with a value | Must close with `range_end(token, payload)` of matching type |
| `range_end(token)` | Close a token range | Token must come from `range_start` or `sentinel_token` |
| `range_end(token, payload)` | Same, with a value | The paired `range_start` must also carry a payload |
| `sentinel_token(name)` | Token with no runtime event | For ranges that close before their `range_start` in source order |
| `dag(name)` | Declare a dependency graph | `dag.edge("prologue", "mainloop", label="smem[0]", via="mbarrier")` |

## Choosing push/pop or start/end

Use **push/pop** when the range nests naturally and both calls sit in the same
structured scope. This is the clearest shape for phase instrumentation: setup, mainloop,
wait, issue, epilogue.

Use **start/end** when an explicit handle makes pairing clearer: the range closes at a
later synchronization point, crosses an iteration boundary, or has several mutually
exclusive close sites.

## Payloads

Allowed: Python bool / int / float literals, and CuTe DSL numeric and index scalars.
Not allowed: tensors, tuples, any aggregate.

Plain Python int literals become 32-bit integer payloads and plain float literals become
32-bit float payloads. For 64-bit, use explicit types:

```python
iket.mark("large_count", cutlass.Int64(0x100000000))
iket.mark("scale", cutlass.Float64(3.141592653589793))
```

Prefer warp-uniform payloads such as loop indices or block coordinates. If active threads
in the warp evaluate the payload differently, the dumped value comes from the first
active thread; guard with `if tidx == 0:` to pin it, and guard paired endpoints
consistently.

For token ranges the start and end payload forms must match — you cannot start with a
payload and end without one.

## Cross-iteration ranges

When a pipelined loop starts work for iteration `N` but the meaningful boundary appears
in iteration `N + 1`, initialize a sentinel before the loop. Creating a sentinel emits no
event, and calling `range_end` on the initial sentinel is valid and also emits nothing.

```python
iter_token = iket.sentinel_token("mma_k_tile")

for k_tile in cutlass.range(k_tile_count):
    ab_full = ab_consumer.wait_and_advance()
    if k_tile > 0:
        iket.range_end(iter_token)          # close previous tile at this wait boundary
    iter_token = iket.range_start("mma_k_tile")
    cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
    ab_full.release()

if k_tile_count > 0:
    iket.range_end(iter_token)              # final drain boundary
```

Use this only when the cross-iteration boundary is genuinely meaningful. If start and end
both fall inside one iteration, a plain push/pop is simpler.
