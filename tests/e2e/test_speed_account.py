"""What the FP4 cache costs per decode step, stated so nobody has to guess.

This is a record, not a gate. It exists because the number it produces is easy
to misread, and the misreading is expensive: somebody profiles a served NVFP4
engine, finds each decode step slower than a BF16 one, and concludes the FP4
kernel is slow. Two different costs are folded into that difference and only one
of them is the kernel.

The other is CUDA graph. The NVFP4 path refuses graph capture — promotion under
capture is unverified, so ``guards`` insists on eager — while a BF16 engine in
production captures. So a naive comparison charges quantization for the whole
gap. Three arms separate them:

    BF16 with graph   the configuration a BF16 deployment actually runs
    BF16 eager        the same arithmetic, minus the graph
    NVFP4 eager       ours

BF16 graph minus BF16 eager is what the graph is worth. NVFP4 eager minus BF16
eager is what four bits cost. Only the second is attributable to this project,
and only the first is recoverable by finishing the CUDA graph work.

Latency comes from a two-point fit rather than from instrumenting the engine.
The same prompts are generated twice, once for a short completion and once for a
long one, and the slope of wall time against tokens is the per-step cost. Prefill
and every other fixed cost sits in the intercept and cancels. Hooking
``execute_model`` would give a distribution rather than a mean, at the price of
measuring a patched engine.

There is no fourth arm for BF16 through this backend. Under any cache dtype but
NVFP4 ``forward`` returns ``super().forward(...)`` on its first line and the
builder is a pass-through, so the arm would measure a Python branch.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import pytest
import torch


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
BATCH = 8
PROMPT_TOKENS = 1000
# The two points. Far enough apart that the difference is dominated by decode
# rather than by the noise on either measurement.
SHORT_TOKENS = 32
LONG_TOKENS = 288
REPEATS = 3


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass(frozen=True)
class Arm:
    label: str
    short_seconds: float
    long_seconds: float

    @property
    def micros_per_step(self) -> float:
        return (
            (self.long_seconds - self.short_seconds)
            / (LONG_TOKENS - SHORT_TOKENS)
            * 1e6
        )

    def summary(self) -> str:
        return f"{self.label}: {self.micros_per_step:,.0f} us/step"


def _prompts() -> list:
    from vllm.inputs import TokensPrompt

    return [
        TokensPrompt(
            prompt_token_ids=[
                1000 + (row * 31 + i * 7) % 20000 for i in range(PROMPT_TOKENS)
            ]
        )
        for row in range(BATCH)
    ]


def _time_generation(llm, prompts, max_tokens: int) -> float:
    """Best of several runs. The fastest is the one least disturbed."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True
    )
    best = float("inf")
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - began)
    return best


def _run_arm(label: str, kv_cache_dtype: str, eager: bool) -> Arm:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=BATCH,
        max_num_batched_tokens=MAX_MODEL_LEN * 2,
        gpu_memory_utilization=0.85,
        enforce_eager=eager,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={
            "backend": "CUSTOM" if kv_cache_dtype == "nvfp4" else "FLASH_ATTN"
        },
    )
    try:
        prompts = _prompts()
        # Unmeasured. The first generation compiles kernels and, for the graph
        # arm, captures them.
        llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        return Arm(
            label=label,
            short_seconds=_time_generation(llm, prompts, SHORT_TOKENS),
            long_seconds=_time_generation(llm, prompts, LONG_TOKENS),
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def arms() -> dict[str, Arm]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    result = {
        "bf16_graph": _run_arm("BF16 + CUDA graph", "auto", eager=False),
        "bf16_eager": _run_arm("BF16 eager", "auto", eager=True),
        "nvfp4_eager": _run_arm("NVFP4 eager", "nvfp4", eager=True),
    }
    graph_cost = (
        result["bf16_eager"].micros_per_step
        - result["bf16_graph"].micros_per_step
    )
    quant_cost = (
        result["nvfp4_eager"].micros_per_step
        - result["bf16_eager"].micros_per_step
    )
    print(
        "\n"
        + "\n".join(arm.summary() for arm in result.values())
        + f"\n  giving up the graph: {graph_cost:+,.0f} us/step"
        + f"\n  four bits:           {quant_cost:+,.0f} us/step"
    )
    return result


def test_the_three_arms_all_produced_a_usable_slope(arms):
    """A negative or absurd slope means the fit measured noise, not decode."""
    for arm in arms.values():
        assert arm.long_seconds > arm.short_seconds, (
            f"{arm.label} took {arm.long_seconds:.3f}s for {LONG_TOKENS} "
            f"tokens and {arm.short_seconds:.3f}s for {SHORT_TOKENS}, so the "
            "two-point fit has nothing to say"
        )
        assert 1e3 < arm.micros_per_step < 1e6, (
            f"{arm.label} fitted {arm.micros_per_step:,.0f} us/step, which is "
            "outside any plausible range for an 8B decode step"
        )


def test_the_graph_is_worth_something(arms):
    """Otherwise the eager comparison below has no cost to separate out.

    If a captured BF16 engine were no faster than an eager one, the whole
    reason for splitting the account would be gone — and it would more likely
    mean the graph arm never captured than that capture is free.
    """
    graph = arms["bf16_graph"].micros_per_step
    eager = arms["bf16_eager"].micros_per_step
    assert eager > graph, (
        f"BF16 eager fitted {eager:,.0f} us/step against {graph:,.0f} with a "
        "graph, so the graph arm probably did not capture"
    )
