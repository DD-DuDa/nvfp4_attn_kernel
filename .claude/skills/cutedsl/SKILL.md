---
name: cutedsl
description: Use when writing, debugging, or optimizing CuTe DSL (CUTLASS Python DSL) kernels. Triggers on cute.kernel, @cute.jit, cute.arch, cutlass.cute, nvidia-cutlass-dsl, or requests to implement GEMM, attention, TMA / mbarrier / TMEM kernels in Python instead of CUDA C++.
---

# CuTe DSL

NVIDIA's Python DSL that JIT-compiles (Python → MLIR → PTX) and mirrors CUTLASS 3.x's C++ CuTe abstractions. Targets Ampere (SM80), Hopper (SM90), and Blackwell (SM100 / SM120) Tensor Cores.

## Start from an example, not a blank file

Clone CUTLASS and read a working kernel close to the target before writing anything:

- `cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/` — Blackwell GEMM patterns
- `cutlass/examples/python/CuTeDSL/hopper/` — Hopper GEMM, warp-specialization, TMA
- `cutlass/examples/python/CuTeDSL/notebooks/` — walkthroughs (repo-only, not shipped in the pip wheel)
- `cutlass/python/CuTeDSL/` — DSL source; canonical when docs disagree

## Reference links

| Resource | URL |
|---|---|
| CUTLASS repo | https://github.com/NVIDIA/cutlass |
| CuTe DSL Quick Start | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html |
| CuTe DSL Examples | https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL |
| CuTe DSL Examples for SM100 | https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell |
| CuTe DSL Examples for SM120 | https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell_geforce |
| NVIDIA blog intro | https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/ |
| Colfax — FlexAttention on CuTe DSL | https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/ |
| Chris Choy — hello → tiled kernels | http://chrischoy.org/posts/cutedsl-basics/ |
| Simon Veitner — applied intro | https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/ |
| Ian Barber — pitfalls, Triton comparison | https://ianbarber.blog/2025/07/04/cute-dsl/ |
| Dao-AILab/quack — production CuTe DSL kernels | https://github.com/Dao-AILab/quack |
| FlashInfer — RoPE, MoE, GDN in CuTe DSL | https://github.com/flashinfer-ai/flashinfer |

## Install

```bash
# CUDA 12.9
pip install nvidia-cutlass-dsl

# CUDA 13.1
pip install 'nvidia-cutlass-dsl[cu13]'

# To match the latest repo examples, use setup.sh instead of pip:
git clone https://github.com/NVIDIA/cutlass.git
./cutlass/python/CuTeDSL/setup.sh --cu12   # or --cu13
```

## Minimal kernel

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("hello\n")

@cute.jit
def host():
    kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))

cutlass.cuda.initialize_cuda_context()
host()
```

## Rules that break MLIR if violated

- **No `return` inside `@cute.kernel`.** Initialize every variable before any branch.
- Use `cutlass.range(n, unroll_full=True)` — required for SSA threading on dynamic-count loops.
- Use `cutlass.range_constexpr(n)` for compile-time loop counters.
- Avoid multiple `@cute.jit` host functions in one Python scope — confuses MLIR on launches.

## Print and debug

- `cute.printf("val: %d", x)` for runtime values. Python `print` runs at compile time and shows `?` for dynamic values.
- `cute.printf("layout: {}", t.layout)` — `{}` is the CuTe-type formatter.
- Wrap per-thread prints in `with cute.arch.elect_one():` to avoid N-thread spam.
- **Clear the cache when code changes don't take effect**: `rm -rf ~/.cache/cutedsl`. Kernels are aggressively cached.
- `compute-sanitizer` returns raw addresses, not line numbers, for CuTe DSL code — full support is still landing. For illegal-access bugs, narrow down by binary search on tile sizes / thread counts.
- Use `cute.compile` ahead-of-time to separate compile errors from launch errors and reuse kernels across invocations.
- **FlexAttention `score_mod`** must be written against the `TensorSSA` abstraction — see the CUTLASS TensorSSA notebook.

## Synchronization / TMA quick reference

- `cute.arch.barrier(barrier_id=id, number_of_threads=count)` — arrive + wait
- `cute.arch.barrier_arrive(barrier_id=id, number_of_threads=count)` — arrive only
- TMA loads use `mbarrier`s for completion tracking.
- **TMA stores require a proxy fence before issuance.**
- For the full TMA workflow (descriptor → partitioning → pipeline → multicast / store-reduce), read the Hopper / Blackwell examples — don't freelance it.

## Status

CuTe DSL graduated from public beta end of summer 2025 and is still evolving fast. Always check the CUTLASS repo release notes against the `nvidia-cutlass-dsl` version pinned in the environment.