from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.make_benchmark_plots import (
    has_concurrency_latency_data,
    has_gpu_util_time_series,
    has_latency_distribution_data,
    has_rtf_chunk_data,
    load_benchmark_artifacts,
)
from scripts.profile_inference import parse_stream_counts, validate_language_target


def test_parse_stream_counts() -> None:
    assert parse_stream_counts("1,2,4,8") == [1, 2, 4, 8]


def test_parse_stream_counts_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_stream_counts("1,0")


def test_validate_language_target_rejects_greek_english_target() -> None:
    with pytest.raises(ValueError, match="Greek"):
        validate_language_target("el", "en-US")


def test_load_benchmark_artifacts() -> None:
    with _workspace_temp_dir() as temp_dir:
        (temp_dir / "profile_summary.json").write_text(
            json.dumps({"per_input_latencies_ms": [1.0]}),
            encoding="utf-8",
        )

        artifacts = load_benchmark_artifacts(str(temp_dir))

    assert artifacts == {"profile_summary": {"per_input_latencies_ms": [1.0]}}


def test_plot_data_presence_helpers() -> None:
    artifacts = {
        "profile_summary": {
            "per_input_latencies_ms": [1.0, 2.0],
            "chunk_metrics": [{"chunk_ms": 100, "rtf": 0.4}],
        },
        "gpu_snapshots": [{"gpu_utilization_pct": 50.0}],
        "concurrency_summary": [{"stream_count": 1, "p95_latency_ms": 100.0}],
    }

    assert has_latency_distribution_data(artifacts)
    assert has_rtf_chunk_data(artifacts)
    assert has_gpu_util_time_series(artifacts)
    assert has_concurrency_latency_data(artifacts)


def test_missing_plot_data_helpers_return_false() -> None:
    assert not has_latency_distribution_data({})
    assert not has_rtf_chunk_data({})
    assert not has_gpu_util_time_series({"gpu_snapshots": [{"gpu_utilization_pct": None}]})
    assert not has_concurrency_latency_data({})


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="benchmark_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)
