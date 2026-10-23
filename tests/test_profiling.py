from __future__ import annotations

import subprocess
import sys
import time
import types

import pytest

from src.profiling import gpu_metrics


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


def test_profile_inference_uses_transcribe_on_callable_model(monkeypatch) -> None:
    # A torch model is callable (forward) and has transcribe(); profiling must use transcribe().
    class CallableModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str]] = []

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("profiling called the model's forward() instead of transcribe()")

        def transcribe(self, audio, target_lang=None):
            self.calls.append((audio, target_lang))
            return ["transcript"]

    times = iter([1.0, 1.1, 2.0, 2.3])

    def fake_snapshot():
        return {
            "memory_allocated_mb": 64.0,
            "gpu_utilization_pct": 25.0,
        }

    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(gpu_metrics, "snapshot", fake_snapshot)
    monkeypatch.setattr(gpu_metrics, "_reset_peak_memory", lambda: None)
    monkeypatch.setattr(gpu_metrics, "_peak_memory_allocated_mb", lambda: 512.0)
    model = CallableModel()

    result = gpu_metrics.profile_inference(
        model=model,
        audio_inputs=["a.wav", "b.wav"],
        label="fake",
    )

    assert model.calls == [(["a.wav"], "el-GR"), (["b.wav"], "el-GR")]
    assert result["label"] == "fake"
    assert result["num_inputs"] == 2
    assert result["per_input_latencies_ms"] == pytest.approx([100.0, 300.0])
    assert result["avg_latency_ms"] == pytest.approx(200.0)
    assert result["max_memory_allocated_mb"] == 512.0
    assert result["avg_gpu_util_pct"] == 25.0


def test_profile_inference_uses_transcribe_adapter_with_target_lang(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []
    times = iter([1.0, 1.25])

    class FakeModel:
        def transcribe(self, paths2audio_files, target_lang):
            calls.append((paths2audio_files, target_lang))
            return ["transcript"]

    def fake_snapshot():
        return {
            "memory_allocated_mb": 64.0,
            "gpu_utilization_pct": 25.0,
        }

    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(gpu_metrics, "snapshot", fake_snapshot)

    result = gpu_metrics.profile_inference(
        model=FakeModel(),
        audio_inputs=["sample.wav"],
        label="el:el-GR",
        target_lang="el-GR",
    )

    assert calls == [(["sample.wav"], "el-GR")]
    assert result["target_lang"] == "el-GR"
    assert result["warnings"] == []
    assert result["per_input_latencies_ms"] == pytest.approx([250.0])


def test_concurrency_test_reports_peak_memory_per_stream_count(monkeypatch) -> None:
    peaks = iter([100.0, 250.0, 400.0])
    runs: list[int] = []

    def fake_run(model, audio_inputs, stream_count, target_lang=None):
        runs.append(stream_count)
        return [10.0 * stream_count]

    monkeypatch.setattr(gpu_metrics, "_run_concurrent_streams", fake_run)
    monkeypatch.setattr(gpu_metrics, "_reset_peak_memory", lambda: None)
    monkeypatch.setattr(gpu_metrics, "_peak_memory_allocated_mb", lambda: next(peaks))

    results = gpu_metrics.concurrency_test(
        model=object(),
        audio_inputs=["a"],
        stream_counts=(1, 2, 4),
    )

    assert runs == [1, 2, 4]
    assert [item["max_memory_allocated_mb"] for item in results] == [100.0, 250.0, 400.0]


def test_peak_memory_is_none_without_cuda(monkeypatch) -> None:
    import sys
    import types

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert gpu_metrics._peak_memory_allocated_mb() is None

