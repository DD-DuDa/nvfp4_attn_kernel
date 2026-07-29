from tests.kernel_profile.bench_decode import summarize


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
    assert path["per_seqlen_geomean"]["1024"] == 2.0
    assert path["overall_geomean"] == 2.0
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
        "per_seqlen_geomean": {},
        "overall_geomean": None,
        "worst_point": None,
    }
