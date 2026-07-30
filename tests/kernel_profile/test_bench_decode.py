import pytest
import torch

from nvfp4_decode_kernel._decode import split_k_heuristic
from tests.kernel_profile.bench_decode import quantize_pages_chunked, summarize


def row(
    *,
    batch: int,
    seqlen: int,
    variant: str,
    gpu_ms: float | None,
    num_splits: int | None = 1,
    status: str = "ok",
) -> dict:
    return {
        "batch": batch,
        "seqlen": seqlen,
        "heads_q": 32,
        "heads_kv": 8,
        "attention_type": "gqa4",
        "variant": variant,
        "status": status,
        "gpu_ms": gpu_ms,
        "num_splits": num_splits,
    }


def test_summary_uses_better_fa4_and_geometric_mean() -> None:
    rows = []
    for batch, fp4_ms in ((1, 1.0), (2, 4.0)):
        rows.extend(
            [
                row(
                    batch=batch,
                    seqlen=1024,
                    variant="fa4_bf16",
                    gpu_ms=2.0,
                ),
                row(
                    batch=batch,
                    seqlen=1024,
                    variant="fa4_split",
                    gpu_ms=1.0,
                    num_splits=8,
                ),
                row(
                    batch=batch,
                    seqlen=1024,
                    variant="fp4_pure_bf16q",
                    gpu_ms=fp4_ms,
                ),
            ]
        )

    summary = summarize(rows)
    path = summary["paths"]["fp4_pure_bf16q"]
    config = path["by_head_config"]["32:8"]
    assert config["per_seqlen_batch_geomean"]["1024"] == 2.0
    assert config["diagnostic_overall_geomean"] == 2.0
    assert path["worst_point"]["ratio_vs_fa4_better"] == 4.0
    assert all(
        item["fa4_baseline_variant"] == "fa4_split"
        for item in summary["ratios"]
    )


def test_summary_reports_fp4_q_unavailable_without_fabricating_data() -> None:
    summary = summarize(
        [
            row(batch=1, seqlen=1024, variant="fa4_bf16", gpu_ms=2.0),
            row(batch=1, seqlen=1024, variant="fa4_split", gpu_ms=1.0),
            row(
                batch=1,
                seqlen=1024,
                variant="fp4_pure_fp4q",
                gpu_ms=None,
                status="unavailable",
            ),
        ]
    )
    assert summary["paths"]["fp4_pure_fp4q"] == {
        "implemented": False,
        "timed": False,
        "by_head_config": {},
        "diagnostic_cross_head_geomean": None,
        "worst_point": None,
    }


@pytest.mark.parametrize(
    ("rows", "heads_kv", "pages", "expected"),
    [
        (1, 1, 128, 8),
        (1, 8, 128, 8),
        (4, 8, 128, 4),
        (8, 8, 128, 2),
        (16, 8, 128, 2),
        (32, 8, 128, 1),
        (1, 8, 127, 1),
        (1, 8, 1, 1),
    ],
)
def test_split_k_heuristic_fills_low_batch_without_splitting_high_batch(
    rows: int,
    heads_kv: int,
    pages: int,
    expected: int,
) -> None:
    assert (
        split_k_heuristic(
            rows,
            heads_kv,
            pages,
            sms=148,
        )
        == expected
    )


def test_summary_keeps_head_config_gates_separate() -> None:
    rows = []
    for heads_q, heads_kv, fp4_ms in ((8, 8, 1.0), (32, 8, 9.0)):
        for batch in (1, 2):
            for variant, gpu_ms in (
                ("fa4_bf16", 2.0),
                ("fa4_split", 1.0),
                ("fp4_pure_bf16q", fp4_ms),
            ):
                item = row(
                    batch=batch,
                    seqlen=1024,
                    variant=variant,
                    gpu_ms=gpu_ms,
                )
                item["heads_q"] = heads_q
                item["heads_kv"] = heads_kv
                item["attention_type"] = (
                    "mha" if heads_q == heads_kv else "gqa4"
                )
                rows.append(item)
    configs = summarize(rows)["paths"]["fp4_pure_bf16q"]["by_head_config"]
    assert configs["8:8"]["per_seqlen_batch_geomean"]["1024"] == 1.0
    assert configs["32:8"]["per_seqlen_batch_geomean"]["1024"] == pytest.approx(9.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_value", [False, True])
def test_chunked_quantization_preserves_native_layout(is_value: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    torch.manual_seed(7)
    pages = torch.randn(
        5, 128, 2, 128, device="cuda", dtype=torch.bfloat16
    )
    from nvfp4_decode_kernel import _quantize

    quantizer = (
        _quantize.quantize_value_pages
        if is_value
        else _quantize.quantize_key_pages
    )
    expected_fp4, expected_scales = quantizer(pages)
    actual_fp4, actual_scales = quantize_pages_chunked(
        pages, is_value=is_value, chunk_pages=3
    )
    assert actual_fp4.shape == expected_fp4.shape
    assert actual_fp4.stride() == expected_fp4.stride()
    assert actual_scales.shape == expected_scales.shape
    assert actual_scales.stride() == expected_scales.stride()
    assert torch.equal(actual_fp4.view(torch.uint8), expected_fp4.view(torch.uint8))
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )
