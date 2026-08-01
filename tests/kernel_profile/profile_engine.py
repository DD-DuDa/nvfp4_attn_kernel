"""Per-kernel GPU cost of a decode step, for an NVFP4 engine and a BF16 one.

``tests/e2e/test_speed_account.py`` says a decode step costs 23,142 us on the
FP4 cache against 13,695 eager on BF16. This says where the difference goes.

Two workloads, because they answer different questions.

``two_point`` is the default and the one the report's per-step tables come
from. Its constants are imported from that speed test rather than restated, so
the two measurements describe the same step and the CPU share can be had by
subtracting this total from that wall time. Kernels are attributed by the same
two-point difference the wall measurement uses: profile a short generation and
a long one, subtract per kernel, divide by the step difference. Prefill runs
once in each and cancels exactly, which matters because prefill quantizes whole
pages and would otherwise be counted as a decode cost.

``legacy`` reproduces the workload behind the previous implementation's audit
in ``nvfp4_attn/docs/kernel_new/docs/1.start_point.html`` — eight prompts of
604 to 686 tokens, 64 generated tokens, one profiled generation summed whole
including its prefill — so that this implementation's Self-CUDA total can be
put beside that report's 323.313 ms and 384.412 ms without either side being
restated in units it was not measured in. It carries prefill on purpose: the
number it is being compared against does.

Only CUDA activity is recorded unless ``--host`` is passed. Turning on CPU
profiling inflates the wall time this is meant to be subtracted from, and the
CPU share is better had as a residual than as a distorted direct measurement.

Writes JSON. ``compare_engine.py`` reduces two of these into a paired table.

    python tests/kernel_profile/profile_engine.py --arm nvfp4 --out /tmp/a.json
    python tests/kernel_profile/profile_engine.py --arm bf16  --out /tmp/b.json
    python tests/kernel_profile/profile_engine.py --arm bf16 --workload legacy \
        --out /tmp/legacy_bf16.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


# The workload of nvfp4_attn/docs/kernel_new/docs/1.start_point.html, which is
# what its 323.313 ms and 384.412 ms Self-CUDA totals were measured on.
LEGACY_BATCH = 8
LEGACY_PROMPT_RANGE = (604, 686)
LEGACY_TOKENS = 64


def _speed_account():
    """The e2e speed test, loaded by path, for its workload constants."""
    path = (
        Path(__file__).resolve().parents[1] / "e2e" / "test_speed_account.py"
    )
    spec = importlib.util.spec_from_file_location("_speed_account", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    os.environ.setdefault("NVFP4_RUN_VLLM_E2E", "1")
    spec.loader.exec_module(module)
    return module


def _kernels(prof, host: bool) -> dict[str, dict[str, float]]:
    from torch.autograd import DeviceType

    totals: dict[str, dict[str, float]] = {}
    for event in prof.key_averages():
        if host:
            micros = event.self_cpu_time_total
        else:
            if event.device_type != DeviceType.CUDA:
                continue
            micros = event.self_device_time_total
        if micros <= 0:
            continue
        row = totals.setdefault(event.key, {"us": 0.0, "count": 0})
        row["us"] += micros
        row["count"] += event.count
    return totals


def _legacy_prompts() -> list:
    """Eight prompts spread across the previous audit's 604–686 token range."""
    from vllm.inputs import TokensPrompt

    low, high = LEGACY_PROMPT_RANGE
    span = (high - low) / max(LEGACY_BATCH - 1, 1)
    return [
        TokensPrompt(
            prompt_token_ids=[
                1000 + (row * 31 + i * 7) % 20000
                for i in range(low + round(row * span))
            ]
        )
        for row in range(LEGACY_BATCH)
    ]


def _profile(llm, prompts, max_tokens: int, host: bool) -> tuple[dict, float]:
    from torch.profiler import ProfilerActivity, profile
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True
    )
    activities = [ProfilerActivity.CUDA]
    if host:
        activities.append(ProfilerActivity.CPU)
    torch.cuda.synchronize()
    with profile(activities=activities) as prof:
        began = time.perf_counter()
        llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - began
    return _kernels(prof, host), wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["nvfp4", "bf16"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--workload",
        choices=["two_point", "legacy"],
        default="two_point",
        help=(
            "two_point: per-decode-step, prefill differenced away. "
            "legacy: one K=8 / 64-token generation summed whole, matching "
            "the previous implementation's audit."
        ),
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help=(
            "record CPU operators too. Inflates wall time, so the result "
            "is only usable as a ranking of where host time goes."
        ),
    )
    args = parser.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    speed = _speed_account()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=speed.MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4" if args.arm == "nvfp4" else "auto",
        max_model_len=speed.MAX_MODEL_LEN,
        max_num_seqs=speed.BATCH,
        max_num_batched_tokens=speed.MAX_MODEL_LEN * 2,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        block_size=speed.PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={
            "backend": "CUSTOM" if args.arm == "nvfp4" else "FLASH_ATTN"
        },
    )
    try:
        prompts = (
            speed._prompts()
            if args.workload == "two_point"
            else _legacy_prompts()
        )
        # Unmeasured, and 8 tokens exactly as the previous audit's warmup was.
        llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        if args.workload == "legacy":
            totals, wall = _profile(llm, prompts, LEGACY_TOKENS, args.host)
            short, short_wall = {}, 0.0
            long, long_wall = totals, wall
        else:
            short, short_wall = _profile(
                llm, prompts, speed.SHORT_TOKENS, args.host
            )
            long, long_wall = _profile(
                llm, prompts, speed.LONG_TOKENS, args.host
            )
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Legacy divides by one: the figure it is compared against is a total over
    # the whole generation, not a rate.
    steps = (
        1
        if args.workload == "legacy"
        else speed.LONG_TOKENS - speed.SHORT_TOKENS
    )
    per_step = {}
    for name in set(short) | set(long):
        a = short.get(name, {"us": 0.0, "count": 0})
        b = long.get(name, {"us": 0.0, "count": 0})
        per_step[name] = {
            "us": (b["us"] - a["us"]) / steps,
            "launches": (b["count"] - a["count"]) / steps,
        }

    legacy = args.workload == "legacy"
    payload = {
        "arm": args.arm,
        "workload": args.workload,
        "host": args.host,
        "batch": LEGACY_BATCH if legacy else speed.BATCH,
        "prompt_tokens": (
            f"{LEGACY_PROMPT_RANGE[0]}-{LEGACY_PROMPT_RANGE[1]}"
            if legacy
            else speed.PROMPT_TOKENS
        ),
        "generated_tokens": LEGACY_TOKENS if legacy else None,
        "short_tokens": None if legacy else speed.SHORT_TOKENS,
        "long_tokens": None if legacy else speed.LONG_TOKENS,
        "steps": steps,
        # Under the profiler, so inflated. Recorded for provenance only; the
        # wall figures this is compared against come from the e2e test.
        "profiled_wall_us": (long_wall - short_wall) / steps * 1e6,
        "kernels": dict(
            sorted(per_step.items(), key=lambda kv: -kv[1]["us"])
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    total = sum(row["us"] for row in per_step.values())
    if legacy:
        print(
            f"\n{args.arm}: {total / 1e3:,.3f} ms Self-CUDA over one "
            f"K={LEGACY_BATCH} / {LEGACY_TOKENS}-token generation"
        )
    else:
        print(f"\n{args.arm}: {total:,.0f} us/step of GPU work")
    for name, row in list(payload["kernels"].items())[:25]:
        print(f"  {row['us']:9,.1f} us  x{row['launches']:6.1f}  {name[:88]}")


if __name__ == "__main__":
    main()
