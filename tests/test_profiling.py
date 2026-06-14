from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.profiling import gpu_metrics
from src.profiling.latency import LatencyTracker


def test_latency_tracker_summary() -> None:
    tracker = LatencyTracker()
    tracker.record(100.0)
    tracker.record(200.0)
    tracker.record(300.0)

    summary = tracker.summary()

    assert summary["count"] == 3
    assert summary["min_latency_ms"] == 100.0
    assert summary["max_latency_ms"] == 300.0
    assert summary["mean_latency_ms"] == 200.0
    assert summary["p50_latency_ms"] == 200.0
    assert summary["p95_latency_ms"] == pytest.approx(290.0)
    assert summary["p99_latency_ms"] == pytest.approx(298.0)


def test_latency_tracker_start_stop_with_clock() -> None:
    times = iter([1.0, 1.25])
    tracker = LatencyTracker(clock=lambda: next(times))

    tracker.start("decode")
    latency_ms = tracker.stop("decode")

    assert latency_ms == 250.0
    assert tracker.summary()["mean_latency_ms"] == 250.0


def test_latency_tracker_to_json() -> None:
    tracker = LatencyTracker()
    tracker.record(123.0)
    with _workspace_temp_dir() as temp_dir:
        output_path = temp_dir / "latency.json"

        tracker.to_json(str(output_path))

        assert json.loads(output_path.read_text())["mean_latency_ms"] == 123.0


def test_snapshot_structure_when_torch_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    result = gpu_metrics.snapshot()

    assert set(result) == {
        "timestamp",
        "cuda_available",
        "gpu_name",
        "memory_allocated_mb",
        "memory_reserved_mb",
        "memory_total_mb",
        "gpu_utilization_pct",
        "gpu_temperature_c",
        "nvidia_smi_available",
    }
    assert result["cuda_available"] is False


def test_nvidia_smi_snapshot_returns_none_when_unavailable() -> None:
    def missing_runner(*_args, **_kwargs):
        raise FileNotFoundError

    assert gpu_metrics._nvidia_smi_snapshot(runner=missing_runner) is None


def test_nvidia_smi_snapshot_parses_csv() -> None:
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            stdout="NVIDIA Test GPU, 8192, 42, 55\n",
            stderr="",
        )

    result = gpu_metrics._nvidia_smi_snapshot(runner=runner)

    assert result == {
        "gpu_name": "NVIDIA Test GPU",
        "memory_total_mb": 8192.0,
        "gpu_utilization_pct": 42.0,
        "gpu_temperature_c": 55.0,
    }


def test_torch_snapshot_uses_cuda_memory(monkeypatch) -> None:
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _device_index: "Fake GPU",
        memory_allocated=lambda _device_index: 256 * 1024**2,
        memory_reserved=lambda _device_index: 512 * 1024**2,
        get_device_properties=lambda _device_index: types.SimpleNamespace(
            total_memory=1024 * 1024**2
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    result = gpu_metrics._torch_snapshot()

    assert result["cuda_available"] is True
    assert result["gpu_name"] == "Fake GPU"
    assert result["memory_allocated_mb"] == 256.0
    assert result["memory_reserved_mb"] == 512.0
    assert result["memory_total_mb"] == 1024.0


def test_concurrency_summary_marks_degraded_when_p95_doubles_baseline() -> None:
    summary = gpu_metrics._concurrency_summary_from_latencies(
        stream_count=4,
        latencies_ms=[100.0, 200.0, 500.0],
        baseline_p95_latency_ms=200.0,
        max_memory_allocated_mb=128.0,
    )

    assert summary["degraded"] is True
    assert "2x baseline" in summary["degradation_reason"]
    assert summary["baseline_p95_latency_ms"] == 200.0


def test_concurrency_summary_marks_not_degraded_when_under_threshold() -> None:
    summary = gpu_metrics._concurrency_summary_from_latencies(
        stream_count=2,
        latencies_ms=[100.0, 150.0, 180.0],
        baseline_p95_latency_ms=100.0,
        max_memory_allocated_mb=None,
    )

    assert summary["degraded"] is False
    assert summary["degradation_reason"] is None


def test_profile_inference_uses_fake_model_without_profiler(monkeypatch) -> None:
    calls: list[str] = []
    times = iter([1.0, 1.1, 2.0, 2.3])

    def fake_model(audio_input):
        calls.append(audio_input)

    def fake_snapshot():
        return {
            "memory_allocated_mb": 64.0,
            "gpu_utilization_pct": 25.0,
        }

    monkeypatch.setattr(gpu_metrics.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(gpu_metrics, "snapshot", fake_snapshot)
    monkeypatch.setattr(gpu_metrics, "_torch_profiler_context", lambda: None)

    result = gpu_metrics.profile_inference(
        model=fake_model,
        audio_inputs=["a", "b"],
        label="fake",
    )

    assert calls == ["a", "b"]
    assert result["label"] == "fake"
    assert result["num_inputs"] == 2
    assert result["per_input_latencies_ms"] == pytest.approx([100.0, 300.0])
    assert result["avg_latency_ms"] == pytest.approx(200.0)
    assert result["max_memory_allocated_mb"] == 64.0
    assert result["avg_gpu_util_pct"] == 25.0
    assert result["profiler_used"] is False


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="profiling_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)
