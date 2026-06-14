from __future__ import annotations

from scripts.run_batch_eval import (
    build_summary,
    language_readiness_for,
    percentile,
)


def test_build_summary_computes_english_wer_and_latency_stats() -> None:
    rows = [
        {
            "reference_text": "hello world",
            "hypothesis_text": "hello world",
            "latency_ms": 100.0,
        },
        {
            "reference_text": "good morning",
            "hypothesis_text": "good evening",
            "latency_ms": 200.0,
        },
    ]

    summary = build_summary(
        per_file_rows=rows,
        language="en",
        model_id_or_path="model",
        target_lang="en-US",
        language_readiness="transcription_ready",
        manifest_path="manifest.jsonl",
        warnings=[],
    )

    assert summary["metric_name"] == "wer"
    assert summary["metric_value"] == 0.25
    assert summary["avg_latency_ms"] == 150.0
    assert summary["p50_latency_ms"] == 150.0
    assert summary["p95_latency_ms"] == 195.0
    assert summary["substitutions"] == 1


def test_build_summary_labels_greek_as_adaptation_ready() -> None:
    readiness, warnings = language_readiness_for("el")

    assert readiness == "adaptation_ready"
    assert "exploratory" in warnings[0]


def test_percentile_handles_empty_values() -> None:
    assert percentile([], 50) is None


def test_percentile_handles_single_value() -> None:
    assert percentile([42.0], 95) == 42.0
