from __future__ import annotations

import pytest

from scripts.run_streaming_demo import (
    build_arg_parser,
    chunk_ms_to_samples,
    summarize_session,
    validate_language_target,
)
from src.asr.model_loader import DEFAULT_MODEL_ID


def test_parser_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--language", "en"])

    assert args.model_id_or_path == DEFAULT_MODEL_ID
    assert args.chunk_ms == 100
    assert args.inference_window_ms == 1000
    assert args.language == "en"
    assert args.target_lang == "en-US"
    assert args.device == "cuda"
    assert args.max_seconds is None


def test_parser_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-id-or-path",
            "model.nemo",
            "--chunk-ms",
            "200",
            "--inference-window-ms",
            "1500",
            "--language",
            "el",
            "--target-lang",
            "el-GR",
            "--device",
            "cpu",
            "--max-seconds",
            "5",
        ]
    )

    assert args.model_id_or_path == "model.nemo"
    assert args.chunk_ms == 200
    assert args.inference_window_ms == 1500
    assert args.language == "el"
    assert args.target_lang == "el-GR"
    assert args.device == "cpu"
    assert args.max_seconds == 5


def test_summarize_session_calculates_latency_and_rtf_stats() -> None:
    events = [
        {"latency_ms": 100.0, "rtf": 0.2},
        {"latency_ms": 200.0, "rtf": 0.4},
        {"latency_ms": 300.0, "rtf": 0.6},
    ]

    summary = summarize_session(events, total_runtime_s=4.0)

    assert summary["total_runtime_s"] == 4.0
    assert summary["inference_windows"] == 3
    assert summary["avg_latency_ms"] == 200.0
    assert summary["p50_latency_ms"] == 200.0
    assert summary["p95_latency_ms"] == pytest.approx(290.0)
    assert summary["avg_rtf"] == pytest.approx(0.4)


def test_summarize_session_handles_no_events() -> None:
    summary = summarize_session([], total_runtime_s=1.5)

    assert summary["total_runtime_s"] == 1.5
    assert summary["inference_windows"] == 0
    assert summary["avg_latency_ms"] is None
    assert summary["p50_latency_ms"] is None
    assert summary["p95_latency_ms"] is None
    assert summary["avg_rtf"] is None


def test_language_target_validation_rejects_greek_with_english_target() -> None:
    with pytest.raises(ValueError, match="Greek"):
        validate_language_target("el", "en-US")


def test_chunk_ms_to_samples() -> None:
    assert chunk_ms_to_samples(100, 16000) == 1600
