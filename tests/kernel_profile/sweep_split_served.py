#!/usr/bin/env python3
"""What a decode split count costs in a served engine, not in a probe.

``fp4_decode`` takes ``num_splits`` and production passes nothing, so every
served step runs one tile. The standalone probes disagree with that default:
under graph replay they measure split-K as a large win at low batch, because
the split path's extra launches and zero-fills are free once they are inside a
graph. A probe is not a served step, though. The graph holds the whole model
forward rather than one kernel, the batch composition moves as requests retire,
and the page table is frozen at ``max_model_len`` under capture. This measures
the thing the recommendation is actually about.

One process serves one split count, because the count is a compile-time
constant that graph capture bakes in: changing it inside a live engine would
leave the captured graphs holding the old value. The driver runs the process
once per count and ``--report`` collates the JSON they leave behind.

The count is injected by wrapping ``nvfp4_vllm.impl.fp4_decode`` before the
engine is built, which is measurement code standing in for a production
decision that has not been made yet. Nothing about the injection is trusted:
``evidence`` in the output counts how many decode calls entered the split path
while the stream was capturing, with which count, and what the split kernel's
compile cache ended up keyed on. A sweep whose arms all secretly ran at one
split is the failure this is here to rule out.

Usage:

    CUDA_VISIBLE_DEVICES=1 python tests/kernel_profile/sweep_split_served.py \
        --num-splits 4 --out /tmp/split-sweep

    python tests/kernel_profile/sweep_split_served.py --report /tmp/split-sweep

Environment overrides: ``NVFP4_TEST_MODEL``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)
PAGE_SIZE = 128
MAX_NUM_SEQS = 32


@dataclass
class Step:
    """One ``execute_model``, as the batch looked to the metadata builder."""

    num_reqs: int
    live_rows: int
    max_query_len: int
    seconds: float


@dataclass
class Recorder:
    """Everything the host can still see once the steps are graph replays."""

    steps: list[Step] = field(default_factory=list)
    pending: tuple[int, int, int] | None = None
    replays: int = 0
    split_calls_capturing: int = 0
    split_calls_serving: int = 0
    split_values: set[int] = field(default_factory=set)
    captured_widths: dict[int, int] = field(default_factory=dict)
    single_calls_capturing: int = 0
    single_calls_serving: int = 0
    compiles_under_capture: int = 0

    def reset_steps(self) -> None:
        self.steps.clear()
        self.replays = 0


def _split_for_rows(rows: int, heads_kv: int, sms: int, cap: int) -> int:
    """One CTA per SM if the batch cannot fill the machine on its own.

    The single-tile decode launches one CTA per (row, KV head), so a batch of
    ``rows`` covers ``rows * heads_kv`` of the ``sms`` the device has. Splitting
    the key axis is worth exactly the SMs that would otherwise idle, and never
    more: past that point the extra tiles queue behind each other and the
    combine and its zero-fills are pure cost.

    Nothing here reads the page table, which is what makes it safe to evaluate
    while a graph is being captured. Under capture that table is frozen at
    ``max_model_len``, so a count derived from it would be an inflated one
    baked into every later replay.
    """
    budget = sms // max(1, rows * heads_kv)
    splits = 1
    while splits * 2 <= min(budget, cap):
        splits *= 2
    return splits


def _install_patches(recorder: Recorder, num_splits: int, per_row: bool):
    """Inject the split count and count what the injection actually did.

    Order matters: ``nvfp4_vllm.impl`` holds ``fp4_decode`` as a module global
    and vLLM imports that module through the plugin, so the wrapper has to be
    in place before the engine exists — and long before capture, which is the
    only moment the count reaches a graph.
    """
    import nvfp4_vllm.builder as builder_module
    import nvfp4_vllm.impl as impl_module
    import nvfp4_decode_kernel._decode as decode_module

    original_fp4_decode = impl_module.fp4_decode
    original_split = decode_module.decode_fp4_split
    original_single = decode_module.decode_fp4
    original_build = builder_module.NVFP4MetadataBuilder.build
    original_replay = torch.cuda.CUDAGraph.replay
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def injected_fp4_decode(**kwargs):
        chosen = num_splits
        if per_row:
            chosen = _split_for_rows(
                kwargs["query"].shape[0],
                kwargs["key_pages_fp4"].shape[2],
                sms,
                num_splits,
            )
        return original_fp4_decode(num_splits=chosen, **kwargs)

    def counted_split(**kwargs):
        recorder.split_values.add(kwargs["num_splits"])
        if torch.cuda.is_current_stream_capturing():
            recorder.split_calls_capturing += 1
            recorder.captured_widths[kwargs["query_fp4"].shape[0]] = kwargs[
                "num_splits"
            ]
        else:
            recorder.split_calls_serving += 1
        return _watch_compiles(original_split, kwargs)

    def counted_single(**kwargs):
        # The split path calls this once with ``validate_only`` to reuse the
        # contract checks. Only a call that runs counts as a single tile.
        if kwargs.get("validate_only"):
            return original_single(**kwargs)
        if torch.cuda.is_current_stream_capturing():
            recorder.single_calls_capturing += 1
            recorder.captured_widths[kwargs["query_fp4"].shape[0]] = 1
        else:
            recorder.single_calls_serving += 1
        return _watch_compiles(original_single, kwargs)

    def _watch_compiles(call, kwargs):
        """A compile on a capturing stream allocates and synchronizes.

        Per-width split counts multiply the kernels an engine has to build, so
        this is the check that vLLM's warmup before each captured size really
        does build that size's kernel before the recording starts.
        """
        if not torch.cuda.is_current_stream_capturing():
            return call(**kwargs)
        before = len(decode_module._decode_compile_cache) + len(
            decode_module._split_decode_compile_cache
        )
        result = call(**kwargs)
        after = len(decode_module._decode_compile_cache) + len(
            decode_module._split_decode_compile_cache
        )
        if after != before:
            recorder.compiles_under_capture += 1
        return result

    def recorded_build(
        self, common_prefix_len, common_attn_metadata, fast_build=False
    ):
        meta = common_attn_metadata
        starts = meta.query_start_loc_cpu
        lengths = (
            starts[1 : meta.num_reqs + 1] - starts[: meta.num_reqs]
        ).tolist()
        recorder.pending = (
            meta.num_reqs,
            sum(1 for length in lengths if length > 0),
            max(lengths) if lengths else 0,
        )
        return original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )

    def counted_replay(self):
        recorder.replays += 1
        return original_replay(self)

    impl_module.fp4_decode = injected_fp4_decode
    decode_module.decode_fp4_split = counted_split
    decode_module.decode_fp4 = counted_single
    builder_module.NVFP4MetadataBuilder.build = recorded_build
    torch.cuda.CUDAGraph.replay = counted_replay


def _model_runner(llm):
    core = llm.llm_engine.engine_core
    while hasattr(core, "engine_core"):
        core = core.engine_core
    return core.model_executor.driver_worker.worker.model_runner


def _time_steps(runner, recorder: Recorder) -> None:
    """Charge each step the host time of its own ``execute_model``, drained.

    The synchronization is not optional. ``execute_model`` hands back an
    asynchronous output whose sampled tokens are copied on another stream, so
    without a drain it times the launch rather than the step — a full BF16 8B
    decode step has to read sixteen gigabytes of weights, which no B200 does in
    the 0.7 ms an undrained call reports. Draining costs the overlap between a
    step's tail and the next step's host work, which is the same on every arm.
    """
    original_execute = runner.execute_model

    def timed_execute(*args, **kwargs):
        recorder.pending = None
        began = time.perf_counter()
        output = original_execute(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        if recorder.pending is not None:
            num_reqs, live_rows, max_query_len = recorder.pending
            recorder.steps.append(
                Step(num_reqs, live_rows, max_query_len, elapsed)
            )
        return output

    runner.execute_model = timed_execute


def _prompts(batch: int, prompt_tokens: int, salt: int):
    from vllm.inputs import TokensPrompt

    return [
        TokensPrompt(
            prompt_token_ids=[
                1000 + (salt * 977 + row * 31 + index * 7) % 20000
                for index in range(prompt_tokens)
            ]
        )
        for row in range(batch)
    ]


def _generate(llm, prompts, max_tokens: int):
    from vllm import SamplingParams

    return llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        ),
        use_tqdm=False,
    )


def _steady_steps(steps: list[Step], batch: int) -> list[float]:
    """Only the steps where every row of the target batch is decoding.

    A prompt that has been prefilled starts decoding while the rest are still
    being prefilled, so the front of a run is mixed steps and narrow decode
    steps; the back is a batch shrinking as requests retire. Neither is the
    configuration being measured.
    """
    return [
        step.seconds
        for step in steps
        if step.num_reqs == batch
        and step.live_rows == batch
        and step.max_query_len == 1
    ]


def _measure_point(
    llm, recorder: Recorder, batch: int, prompt_tokens: int, args, salt: int
) -> dict:
    prompts = _prompts(batch, prompt_tokens, salt)
    samples: list[float] = []
    replays = 0
    decode_steps = 0
    wall = 0.0
    accounted = 0.0
    engine_seconds = 0.0
    completion = ()
    for _ in range(args.repeats):
        recorder.reset_steps()
        began = time.perf_counter()
        outputs = _generate(llm, prompts, args.generate_tokens)
        elapsed = time.perf_counter() - began
        wall += elapsed
        accounted += sum(step.seconds for step in recorder.steps)
        replays += recorder.replays
        steady = _steady_steps(recorder.steps, batch)
        decode_steps += len(steady)
        # Everything this call spent that was not another kind of step, so the
        # engine loop's own per-step cost — scheduling, output processing,
        # detokenization — is charged to the steps it belongs to.
        engine_seconds += elapsed - (
            sum(step.seconds for step in recorder.steps) - sum(steady)
        )
        # The first few steps of the steady window still carry the tail of
        # prefill's allocator and scheduler churn.
        samples.extend(steady[args.discard :])
        completion = tuple(outputs[0].outputs[0].token_ids[:16])
    if not samples:
        return {
            "batch": batch,
            "prompt_tokens": prompt_tokens,
            "steady_steps": 0,
            "error": "no steady-state decode step at this width",
        }
    samples.sort()
    return {
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "steady_steps": decode_steps,
        "samples": len(samples),
        "median_us": statistics.median(samples) * 1e6,
        "mean_us": statistics.fmean(samples) * 1e6,
        "p10_us": samples[len(samples) // 10] * 1e6,
        "p90_us": samples[len(samples) * 9 // 10] * 1e6,
        "engine_us": engine_seconds / decode_steps * 1e6 if decode_steps else 0,
        "replays": replays,
        "wall_seconds": wall,
        "coverage": accounted / wall if wall else 0.0,
        "first_tokens": list(completion),
    }


def _trace(llm, batch: int, prompt_tokens: int, args, salt: int) -> dict:
    """What the replayed graph actually launches, seen by CUPTI.

    The Python counters can only speak for capture. This speaks for replay: a
    split step launches a combine per layer and a single-tile step launches
    none, so the kernel inventory of a replayed step says which one was
    recorded.
    """
    from torch.profiler import ProfilerActivity, profile

    prompts = _prompts(batch, prompt_tokens, salt)
    _generate(llm, prompts, 4)
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        _generate(llm, prompts, args.trace_tokens)
    kernels = [
        {
            "name": event.key,
            "count": event.count,
            "cuda_us": event.self_device_time_total,
        }
        for event in profiler.key_averages()
        if event.self_device_time_total > 0 and event.count > 0
    ]
    kernels.sort(key=lambda entry: entry["cuda_us"], reverse=True)
    return {
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "generated": args.trace_tokens,
        "distinct_kernels": len(kernels),
        "total_launches": sum(entry["count"] for entry in kernels),
        "total_cuda_us": sum(entry["cuda_us"] for entry in kernels),
        "kernels": kernels,
    }


def run_arm(args) -> dict:
    from vllm import LLM

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    recorder = Recorder()
    _install_patches(recorder, args.num_splits, args.per_row)

    started = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
        max_model_len=args.max_model_len,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "CUSTOM"},
        # Exactly the batch sizes being measured, so a measured batch is a
        # captured width rather than a padded one, and so capture does not
        # spend minutes on sizes nothing here replays.
        compilation_config={"cudagraph_capture_sizes": sorted(args.batches)},
    )
    # A per-width count compiles one decode kernel per width, and every one of
    # them has to be built before the width it belongs to is captured.
    startup_seconds = time.perf_counter() - started
    from nvfp4_decode_kernel._decode import (
        _decode_compile_cache,
        _split_decode_compile_cache,
    )

    # Everything above happened during capture; freeze it before serving so
    # the two phases can be told apart.
    evidence = {
        "num_splits_requested": args.num_splits,
        "per_row_policy": args.per_row,
        "split_values_seen": sorted(recorder.split_values),
        "splits_by_captured_width": dict(sorted(recorder.captured_widths.items())),
        "compiles_under_capture": recorder.compiles_under_capture,
        "split_calls_under_capture": recorder.split_calls_capturing,
        "single_tile_calls_under_capture": recorder.single_calls_capturing,
        "split_compile_cache_keys": [
            list(key) for key in _split_decode_compile_cache
        ],
        "single_compile_cache_keys": [
            list(key) for key in _decode_compile_cache
        ],
    }

    runner = _model_runner(llm)
    _time_steps(runner, recorder)
    results = []
    trace = None
    try:
        # Unmeasured, and at the widest captured size: whatever the first
        # served batch still has to compile or allocate, it pays for here.
        _generate(llm, _prompts(max(args.batches), 256, 0), 8)
        for salt, prompt_tokens in enumerate(
            [] if args.trace_only else args.prompts
        ):
            for batch in args.batches:
                point = _measure_point(
                    llm, recorder, batch, prompt_tokens, args, salt * 17 + batch
                )
                results.append(point)
                print(
                    f"  splits={args.num_splits} batch={batch} "
                    f"prompt={prompt_tokens}: "
                    + (
                        point["error"]
                        if "error" in point
                        else f"{point['median_us']:,.0f} us/step over "
                        f"{point['samples']} steps, "
                        f"{point['engine_us']:,.0f} us end to end "
                        f"(coverage {point['coverage']:.2f})"
                    ),
                    flush=True,
                )
        if args.trace_tokens:
            trace = _trace(
                llm, args.trace_batch, max(args.prompts), args, salt=99
            )
    finally:
        evidence["split_calls_while_serving"] = recorder.split_calls_serving
        evidence["single_tile_calls_while_serving"] = (
            recorder.single_calls_serving
        )
        llm.llm_engine.engine_core.shutdown()

    return {
        "num_splits": args.num_splits,
        "startup_seconds": startup_seconds,
        "model": MODEL,
        "max_model_len": args.max_model_len,
        "generate_tokens": args.generate_tokens,
        "repeats": args.repeats,
        "evidence": evidence,
        "points": results,
        "trace": trace,
    }


def _label(arm: dict) -> str:
    if arm["evidence"].get("per_row_policy"):
        return f"per row (<={arm['num_splits']})"
    return f"{arm['num_splits']} split"


def report(directory: Path) -> None:
    arms = []
    for path in sorted(directory.glob("splits-*.json")) + sorted(
        directory.glob("per-row-*.json")
    ):
        arms.append(json.loads(path.read_text()))
    arms.sort(
        key=lambda arm: (
            bool(arm["evidence"].get("per_row_policy")),
            arm["num_splits"],
        )
    )
    if not arms:
        print(f"no arm JSON under {directory}")
        return

    prompts = sorted(
        {point["prompt_tokens"] for arm in arms for point in arm["points"]}
    )
    batches = sorted(
        {point["batch"] for arm in arms for point in arm["points"]}
    )
    header = "| prompt | batch | " + " | ".join(_label(arm) for arm in arms)
    print(header + " | best |")
    print("|---" * (len(arms) + 3) + "|")
    for prompt_tokens in prompts:
        for batch in batches:
            cells = []
            for arm in arms:
                point = next(
                    (
                        candidate
                        for candidate in arm["points"]
                        if candidate["batch"] == batch
                        and candidate["prompt_tokens"] == prompt_tokens
                    ),
                    None,
                )
                cells.append(
                    None
                    if point is None or "median_us" not in point
                    else point["median_us"]
                )
            best = min(
                (
                    (value, _label(arm))
                    for value, arm in zip(cells, arms)
                    if value is not None
                ),
                default=(None, None),
            )
            rendered = " | ".join(
                "-" if value is None else f"{value:,.0f}" for value in cells
            )
            print(
                f"| {prompt_tokens} | {batch} | {rendered} | "
                f"{best[1]} ({best[0]:,.0f}) |"
            )
    print()
    for arm in arms:
        print(f"{_label(arm)} evidence: {arm['evidence']}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-splits", type=int)
    parser.add_argument(
        "--per-row",
        action="store_true",
        help="choose the count per decode row count, capped at --num-splits, "
        "instead of using it as a constant",
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument(
        "--prompts", type=int, nargs="+", default=[2048, 16384]
    )
    parser.add_argument("--generate-tokens", type=int, default=96)
    parser.add_argument("--discard", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=17408)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trace-tokens", type=int, default=0)
    parser.add_argument("--trace-batch", type=int, default=4)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report is not None:
        report(args.report)
        return
    if args.num_splits is None or args.out is None:
        raise SystemExit("--num-splits and --out are required")
    if torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit("SM100 is required")

    result = run_arm(args)
    args.out.mkdir(parents=True, exist_ok=True)
    name = (
        f"per-row-{args.num_splits:02d}"
        if args.per_row
        else f"splits-{args.num_splits:02d}"
    )
    path = args.out / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")
    gc.collect()


if __name__ == "__main__":
    main()
