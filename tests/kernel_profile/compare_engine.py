"""Reduce two ``profile_engine.py`` runs into one account of the decode step.

Kernels are sorted into buckets by name. The buckets are chosen so that each
one answers a question somebody would actually ask about the difference —
whether the FP4 attention kernel is slower than the BF16 one, what the cache
write costs, what the bookkeeping costs — rather than to make a tidy total.

Anything unrecognized lands in ``other`` and is printed, so a kernel that
appears after a change is noticed instead of quietly averaged in.

    python tests/kernel_profile/compare_engine.py /tmp/prof_nvfp4.json /tmp/prof_bf16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Ordered. The first pattern that matches a kernel name claims it, so more
# specific patterns come first.
BUCKETS: list[tuple[str, str, tuple[str, ...]]] = [
    # Sampling has a kernel with "combine" in its name; it is claimed above
    # this entry so the split-K pattern cannot swallow it.
    (
        "model_other",
        "RMSNorm, RoPE, SiLU, sampling",
        ("_combine_sampled_and_draft_tokens", "_gumbel_sample"),
    ),
    (
        "attention",
        "Attention over the KV cache",
        (
            "fp4decodekernel",
            "fp4_decode_kernel",
            "flashattentionforward",
            "flash_fwd",
            "fmha",
            "attention_kernel",
            "split_k_combine",
        ),
    ),
    (
        "q_quant",
        "Quantizing this step's query to FP4",
        ("quantize_q", "_pack_e2m1", "_pack_e4m3", "quantize_query"),
    ),
    (
        "scratch",
        "Zeroing per-step scratch buffers",
        ("fillfunctor",),
    ),
    (
        "kv_write",
        "Writing this step's K/V into the cache",
        ("reshape_and_cache", "_tail_write_kernel", "_tail_reset_kernel"),
    ),
    (
        "promotion",
        "Sealing a full page into FP4",
        ("_quantize_key_pages", "_quantize_value_pages", "_work_table_kernel"),
    ),
    (
        "control",
        "Slot table and metadata",
        (
            "_control_kernel",
            "_gather_block_tables",
            "_compute_slot_mappings",
            "_post_update_kernel",
            "_prepare_inputs",
        ),
    ),
    (
        "gemm",
        "Model-body GEMMs",
        ("nvjet", "gemm", "cublas", "splitk", "cutlass_80", "sm100_xmma"),
    ),
    (
        "model_other",
        "RMSNorm, RoPE, SiLU, sampling",
        (
            "rms_norm",
            "rotary",
            "silu",
            "act_and_mul",
            "softmax",
            "topk",
            "top_k",
            "sort",
            "argmax",
            "reduce_kernel",
            "embedding",
            "cat_",
            "catarray",
        ),
    ),
    ("copy", "Memcpy and elementwise copies", ("memcpy", "memset")),
]


def classify(name: str) -> str:
    lowered = name.lower()
    for key, _, patterns in BUCKETS:
        if any(pattern in lowered for pattern in patterns):
            return key
    return "other"


def reduce(payload: dict) -> tuple[dict[str, dict], list[tuple[str, float]]]:
    totals: dict[str, dict[str, float]] = {}
    unknown: list[tuple[str, float]] = []
    for name, row in payload["kernels"].items():
        key = classify(name)
        bucket = totals.setdefault(key, {"us": 0.0, "launches": 0.0})
        bucket["us"] += row["us"]
        bucket["launches"] += row["launches"]
        bucket.setdefault("top", []).append((name, row["us"], row["launches"]))
        if key == "other" and row["us"] > 0.5:
            unknown.append((name, row["us"]))
    for bucket in totals.values():
        bucket["top"] = sorted(bucket["top"], key=lambda item: -item[1])[:4]
    return totals, sorted(unknown, key=lambda item: -item[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nvfp4")
    parser.add_argument("bf16")
    args = parser.parse_args()

    a = json.loads(Path(args.nvfp4).read_text())
    b = json.loads(Path(args.bf16).read_text())
    assert a["arm"] == "nvfp4" and b["arm"] == "bf16"
    assert a["steps"] == b["steps"]
    assert a.get("workload") == b.get("workload"), (
        "the two arms were profiled on different workloads, so their totals "
        "are not comparable"
    )
    legacy = a.get("workload") == "legacy"

    fp4, fp4_unknown = reduce(a)
    bf16, bf16_unknown = reduce(b)

    # A key may appear in BUCKETS more than once, when patterns that belong to
    # one bucket have to be split around a more specific rule. Print it once.
    order: list[str] = []
    for key, _, _ in BUCKETS:
        if key not in order:
            order.append(key)
    order.append("other")
    labels = {key: text for key, text, _ in BUCKETS}
    labels["other"] = "Unclassified"

    if legacy:
        print(
            f"\nSelf-CUDA over one generation, batch {a['batch']}, "
            f"prompt {a['prompt_tokens']} tokens, "
            f"{a['generated_tokens']} generated, prefill included\n"
        )
    else:
        print(
            f"\nGPU time per decode step, batch {a['batch']}, "
            f"prompt {a['prompt_tokens']} tokens, "
            f"{a['steps']} steps of two-point difference\n"
        )
    print(f"{'bucket':<14}{'BF16 us':>10}{'NVFP4 us':>11}{'delta':>10}"
          f"{'BF16 lch':>10}{'FP4 lch':>9}  what")
    print("-" * 96)
    total_a = total_b = 0.0
    for key in order:
        x = bf16.get(key, {"us": 0.0, "launches": 0.0})
        y = fp4.get(key, {"us": 0.0, "launches": 0.0})
        if x["us"] == 0 and y["us"] == 0:
            continue
        total_a += x["us"]
        total_b += y["us"]
        print(
            f"{key:<14}{x['us']:>10,.1f}{y['us']:>11,.1f}"
            f"{y['us'] - x['us']:>+10,.1f}"
            f"{x['launches']:>10,.0f}{y['launches']:>9,.0f}  {labels[key]}"
        )
    launch_a = sum(v["launches"] for v in bf16.values())
    launch_b = sum(v["launches"] for v in fp4.values())
    print("-" * 96)
    print(
        f"{'TOTAL GPU':<14}{total_a:>10,.1f}{total_b:>11,.1f}"
        f"{total_b - total_a:>+10,.1f}{launch_a:>10,.0f}{launch_b:>9,.0f}"
    )

    if legacy:
        # nvfp4_attn/docs/kernel_new/docs/1.start_point.html, "Paired BF16
        # comparison", collected 2026-07-16 on this same workload.
        previous = {"bf16": 323.313, "nvfp4": 384.412}
        print(
            "\nagainst the previous implementation on the same workload "
            "(nvfp4_attn 1.start_point.html):\n"
        )
        print(f"{'':<14}{'previous ms':>13}{'this ms':>11}{'change':>11}")
        for label, key, now in (
            ("BF16", "bf16", total_a),
            ("NVFP4", "nvfp4", total_b),
        ):
            was = previous[key]
            print(
                f"{label:<14}{was:>13,.3f}{now / 1e3:>11,.3f}"
                f"{(now / 1e3 - was) / was * 100:>+10.1f}%"
            )
        gap_was = previous["nvfp4"] - previous["bf16"]
        gap_now = (total_b - total_a) / 1e3
        print(
            f"{'NVFP4 - BF16':<14}{gap_was:>+13,.3f}{gap_now:>+11,.3f}"
            f"{'':>11}"
        )
        print(
            f"{'as a share':<14}"
            f"{gap_was / previous['bf16'] * 100:>12.2f}%"
            f"{gap_now / (total_a / 1e3) * 100:>10.2f}%"
        )
        print(
            "\nthe BF16 arm is the same vLLM path in both repositories, so "
            "how close\nthose two BF16 numbers are is the measure of how "
            "comparable the runs are."
        )
        return

    # The wall figures come from tests/e2e/test_speed_account.py, measured on
    # this same configuration without a profiler attached. Everything in a step
    # that is not GPU time is the host failing to keep ahead of the device.
    walls = {"bf16": 13_695.0, "nvfp4": 23_142.0}
    print(
        f"\nagainst the unprofiled wall clock "
        f"(tests/e2e/test_speed_account.py):\n"
    )
    print(f"{'':<14}{'wall us':>10}{'GPU us':>11}{'host gap':>10}"
          f"{'gap share':>11}{'us/launch':>11}")
    for label, wall, gpu, launches in (
        ("BF16 eager", walls["bf16"], total_a, launch_a),
        ("NVFP4 eager", walls["nvfp4"], total_b, launch_b),
    ):
        gap = wall - gpu
        print(
            f"{label:<14}{wall:>10,.0f}{gpu:>11,.0f}{gap:>10,.0f}"
            f"{gap / wall * 100:>10.1f}%{gap / launches:>11,.1f}"
        )
    print(
        f"{'delta':<14}{walls['nvfp4'] - walls['bf16']:>+10,.0f}"
        f"{total_b - total_a:>+11,.0f}"
        f"{(walls['nvfp4'] - total_b) - (walls['bf16'] - total_a):>+10,.0f}"
        f"{'':>11}{'':>11}"
    )

    for label, rows in (("NVFP4", fp4_unknown), ("BF16", bf16_unknown)):
        if rows:
            print(f"\nunclassified in {label} (>0.5 us/step):")
            for name, micros in rows[:12]:
                print(f"  {micros:8,.2f} us  {name[:96]}")

    print("\nwhat is inside each bucket (NVFP4 arm):")
    for key in order:
        bucket = fp4.get(key)
        if not bucket:
            continue
        print(f"  {key}")
        for name, micros, launches in bucket["top"]:
            print(f"    {micros:8,.1f} us  x{launches:5.1f}  {name[:84]}")


if __name__ == "__main__":
    main()
