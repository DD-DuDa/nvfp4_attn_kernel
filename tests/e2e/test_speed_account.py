"""What the FP4 cache costs per decode step, stated so nobody has to guess.

This is a record first. It exists because the number it produces is easy to
misread, and the misreading is expensive: somebody profiles a served NVFP4
engine, finds each decode step slower than a BF16 one, and concludes the FP4
kernel is slow. Two different costs are folded into that difference and only one
of them is the kernel. What the assertions at the bottom add is that the record
cannot be quietly wrong — each pins a relationship that has to hold for the
labels on these numbers to mean what they say.

The other is CUDA graph. A decode step is mostly host dispatch and a graph
removes it, so an engine that captures and one that does not are two different
speeds for the same arithmetic. A BF16 deployment captures; the NVFP4 path can
now be captured too, but an eager NVFP4 engine is still a configuration someone
can ask for. Four arms keep the two costs apart:

    BF16 with graph    the configuration a BF16 deployment actually runs
    BF16 eager         the same arithmetic, minus the graph
    NVFP4 eager        ours, with the graph given up
    NVFP4 with graph   ours, as a deployment runs it

Each adjacent pair isolates one thing. BF16 eager minus BF16 graph is what a
graph is worth, measured on a path this project does not touch. NVFP4 eager
minus BF16 eager is what four bits cost with graphs out of the picture on both
sides, and it is the only one of these attributable to this project. NVFP4
eager minus NVFP4 graph is how much of the host cost capture took back. The
last number, the ratio of the two graph arms, is the one a deployment feels,
because both of its terms are configurations somebody would actually serve.

Latency comes from a two-point fit rather than from instrumenting the engine.
The same prompts are generated twice, once for a short completion and once for a
long one, and the slope of wall time against tokens is the per-step cost. Prefill
and every other fixed cost sits in the intercept and cancels. Hooking
``execute_model`` would give a distribution rather than a mean, at the price of
measuring a patched engine.

There is no fifth arm for BF16 through this backend. Under any cache dtype but
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
# The least the captured NVFP4 arm may beat the eager one by and still be read
# as having captured. See ``test_the_nvfp4_graph_arm_captured``.
MIN_GRAPH_SPEEDUP = 2.0
# The acceptance condition of docs/tasks/3.vllm_cudagraph.md §7.2, not a bound
# chosen here. See ``test_a_captured_nvfp4_step_costs_what_a_bf16_one_does``.
MAX_NVFP4_OVER_BF16 = 1.25


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


def _forget_the_graph_pool() -> None:
    """Drop vLLM's memo of the CUDA graph memory pool an engine captured into.

    vLLM asks PyTorch for one graph pool per process and keeps the handle on
    the platform class forever. Destroying an engine that captured releases the
    pool, and the next engine to capture then hands PyTorch a handle to a pool
    that no longer exists, which fails inside the caching allocator. A served
    process builds one engine and never needs this; two of the arms here
    capture.
    """
    from vllm.platforms import current_platform

    for klass in type(current_platform).__mro__:
        if "_global_graph_pool" in vars(klass):
            klass._global_graph_pool = None


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
        _forget_the_graph_pool()
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
        "nvfp4_graph": _run_arm("NVFP4 + CUDA graph", "nvfp4", eager=False),
    }
    graph_cost = (
        result["bf16_eager"].micros_per_step
        - result["bf16_graph"].micros_per_step
    )
    quant_cost = (
        result["nvfp4_eager"].micros_per_step
        - result["bf16_eager"].micros_per_step
    )
    recovered = (
        result["nvfp4_eager"].micros_per_step
        - result["nvfp4_graph"].micros_per_step
    )
    ratio = (
        result["nvfp4_graph"].micros_per_step
        / result["bf16_graph"].micros_per_step
    )
    print(
        "\n"
        + "\n".join(arm.summary() for arm in result.values())
        + f"\n  giving up the graph:   {graph_cost:+,.0f} us/step"
        + f"\n  four bits:             {quant_cost:+,.0f} us/step"
        + f"\n  taking the graph back: {-recovered:+,.0f} us/step"
        + f"\n  NVFP4 over BF16, both captured: {ratio:.2f}x"
    )
    return result


def test_the_four_arms_all_produced_a_usable_slope(arms):
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


def test_the_nvfp4_graph_arm_captured(arms):
    """The fourth arm has to be a captured engine, not an eager one relabelled.

    The BF16 pair above is asked only for a direction, and that is enough
    there: nothing but a failure to capture explains a captured engine being
    no faster than an eager one. The failure this arm invites has a different
    shape. If the backend stopped declaring a support level vLLM can use, or
    the resolver settled on a mode that captures no full graph, the arm would
    still build, still serve, still be labelled a graph — and it would fit the
    eager arm's slope. Direction alone would then be a coin flip, so this asks
    for a size.

    ``MIN_GRAPH_SPEEDUP`` sits near the geometric midpoint of the two
    outcomes. A silently uncaptured arm gives 1x by construction, being the
    eager configuration measured a second time; capture measures 4.9x. Stating
    it as a factor rather than as microseconds keeps it from encoding how fast
    this machine is.
    """
    graph = arms["nvfp4_graph"].micros_per_step
    eager = arms["nvfp4_eager"].micros_per_step
    assert eager > graph * MIN_GRAPH_SPEEDUP, (
        f"NVFP4 fitted {graph:,.0f} us/step with a graph against {eager:,.0f} "
        f"eager, a factor of {eager / graph:.2f}x, and anything under "
        f"{MIN_GRAPH_SPEEDUP:.1f}x reads here as an arm that never captured"
    )


def test_a_captured_nvfp4_step_costs_what_a_bf16_one_does(arms):
    """The comparison a deployment actually faces, against the agreed bound.

    Both terms are configurations someone would serve, measured by the same
    code in the same process, so machine speed divides out and what is left is
    what the cache dtype costs a captured engine.

    ``MAX_NVFP4_OVER_BF16`` is inherited rather than invented: it is the pass
    condition the CUDA graph work was accepted on. The ratio measures 1.00 or
    a shade under, so the whole quarter of the budget is unused, and that
    margin is what keeps this from becoming a noise detector — tightening it
    toward the measured value would assert more than one machine can support.
    The ratio is a property of the served model as much as of the kernel, and
    ``NVFP4_TEST_MODEL`` names the model it was measured on.
    """
    nvfp4 = arms["nvfp4_graph"].micros_per_step
    bf16 = arms["bf16_graph"].micros_per_step
    assert nvfp4 <= bf16 * MAX_NVFP4_OVER_BF16, (
        f"a captured NVFP4 step fitted {nvfp4:,.0f} us against {bf16:,.0f} "
        f"for a captured BF16 one, {nvfp4 / bf16:.2f}x, over the "
        f"{MAX_NVFP4_OVER_BF16:.2f}x this path is meant to hold"
    )
