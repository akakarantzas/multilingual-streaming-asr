from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime, timezone
from statistics import mean
from typing import Any


NVIDIA_SMI_QUERY = (
    "name,memory.total,utilization.gpu,temperature.gpu"
)


def snapshot() -> dict:
    torch_info = _torch_snapshot()
    smi_info = _nvidia_smi_snapshot()
    nvidia_smi_available = smi_info is not None
    smi_info = smi_info or {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cuda_available": torch_info["cuda_available"],
        "gpu_name": torch_info["gpu_name"] or smi_info.get("gpu_name"),
        "memory_allocated_mb": torch_info["memory_allocated_mb"],
        "memory_reserved_mb": torch_info["memory_reserved_mb"],
        "memory_total_mb": torch_info["memory_total_mb"] or smi_info.get("memory_total_mb"),
        "gpu_utilization_pct": smi_info.get("gpu_utilization_pct"),
        "gpu_temperature_c": smi_info.get("gpu_temperature_c"),
        "nvidia_smi_available": nvidia_smi_available,
    }


def profile_inference(model: Any, audio_inputs: list, label: str) -> dict:
    limitations = [
        "Local process inference profile; not a production serving benchmark.",
        "GPU utilization snapshots are sampled before, between inputs, and after inference.",
        "Profiler availability depends on the installed torch build and CUDA visibility.",
    ]
    snapshots = [snapshot()]
    per_input_latencies_ms: list[float] = []
    profiler_used = False

    profiler_context = _torch_profiler_context()
    if profiler_context is not None:
        profiler_used = True

    if profiler_context is None:
        for audio_input in audio_inputs:
            per_input_latencies_ms.append(_time_model_call(model, audio_input))
            snapshots.append(snapshot())
    else:
        with profiler_context as profiler:
            for audio_input in audio_inputs:
                per_input_latencies_ms.append(_time_model_call(model, audio_input))
                snapshots.append(snapshot())
                step = getattr(profiler, "step", None)
                if callable(step):
                    step()

    snapshots.append(snapshot())
    return {
        "label": label,
        "num_inputs": len(audio_inputs),
        "avg_latency_ms": _mean(per_input_latencies_ms),
        "p50_latency_ms": _percentile(per_input_latencies_ms, 50),
        "p95_latency_ms": _percentile(per_input_latencies_ms, 95),
        "max_memory_allocated_mb": _max_present(
            item.get("memory_allocated_mb") for item in snapshots
        ),
        "avg_gpu_util_pct": _mean_present(item.get("gpu_utilization_pct") for item in snapshots),
        "per_input_latencies_ms": per_input_latencies_ms,
        "profiler_used": profiler_used,
        "limitations": limitations,
    }


def concurrency_test(
    model: Any,
    audio_inputs: list,
    stream_counts: list[int] = [1, 2, 4, 8],
) -> list[dict]:
    if not stream_counts:
        return []
    if any(stream_count <= 0 for stream_count in stream_counts):
        raise ValueError("stream_counts must contain positive integers")

    limitations = [
        "Local thread simulation.",
        "Not a Triton benchmark.",
        "May not represent independent production streams.",
        "Uses the same loaded model across threads; if the model is not thread-safe, results are invalid.",
    ]
    baseline_latencies = _run_concurrent_streams(model, audio_inputs, stream_count=1)
    baseline_p95 = _percentile(baseline_latencies, 95)

    results: list[dict] = []
    for stream_count in stream_counts:
        latencies = (
            baseline_latencies
            if stream_count == 1
            else _run_concurrent_streams(model, audio_inputs, stream_count=stream_count)
        )
        summary = _concurrency_summary_from_latencies(
            stream_count=stream_count,
            latencies_ms=latencies,
            baseline_p95_latency_ms=baseline_p95,
            max_memory_allocated_mb=snapshot().get("memory_allocated_mb"),
        )
        summary["limitations"] = limitations
        results.append(summary)
    return results


def _torch_snapshot() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "gpu_name": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "memory_total_mb": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return {
            "cuda_available": False,
            "gpu_name": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "memory_total_mb": None,
        }

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "cuda_available": True,
        "gpu_name": str(torch.cuda.get_device_name(device_index)),
        "memory_allocated_mb": _bytes_to_mb(torch.cuda.memory_allocated(device_index)),
        "memory_reserved_mb": _bytes_to_mb(torch.cuda.memory_reserved(device_index)),
        "memory_total_mb": _bytes_to_mb(properties.total_memory),
    }


def _run_concurrent_streams(model: Any, audio_inputs: list, stream_count: int) -> list[float]:
    latencies: list[float] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        local_latencies: list[float] = []
        try:
            for audio_input in audio_inputs:
                local_latencies.append(_time_model_call(model, audio_input))
        except BaseException as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            latencies.extend(local_latencies)

    threads = [threading.Thread(target=worker) for _ in range(stream_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError(
            "Concurrency simulation failed. The loaded model may not be thread-safe."
        ) from errors[0]
    return latencies


def _concurrency_summary_from_latencies(
    stream_count: int,
    latencies_ms: list[float],
    baseline_p95_latency_ms: float | None,
    max_memory_allocated_mb: float | None,
) -> dict:
    p95_latency_ms = _percentile(latencies_ms, 95)
    degraded, degradation_reason = _degradation_status(
        p95_latency_ms=p95_latency_ms,
        baseline_p95_latency_ms=baseline_p95_latency_ms,
    )
    return {
        "stream_count": stream_count,
        "avg_latency_ms": _mean(latencies_ms),
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": p95_latency_ms,
        "max_memory_allocated_mb": max_memory_allocated_mb,
        "degraded": degraded,
        "degradation_reason": degradation_reason,
        "baseline_p95_latency_ms": baseline_p95_latency_ms,
    }


def _degradation_status(
    p95_latency_ms: float | None,
    baseline_p95_latency_ms: float | None,
) -> tuple[bool, str | None]:
    if p95_latency_ms is None or baseline_p95_latency_ms is None:
        return False, None
    if baseline_p95_latency_ms <= 0:
        return False, None
    if p95_latency_ms > 2 * baseline_p95_latency_ms:
        return (
            True,
            "p95_latency_ms is greater than 2x baseline single-stream p95_latency_ms",
        )
    return False, None


def _time_model_call(model: Any, audio_input: Any) -> float:
    start = time.perf_counter()
    _run_model_inference(model, audio_input)
    return (time.perf_counter() - start) * 1000


def _run_model_inference(model: Any, audio_input: Any) -> Any:
    if callable(model):
        return model(audio_input)
    transcribe = getattr(model, "transcribe", None)
    if callable(transcribe):
        return transcribe([audio_input])
    raise RuntimeError("Model must be callable or expose a callable transcribe() method.")


def _torch_profiler_context() -> Any | None:
    try:
        import torch
    except ImportError:
        return None

    if not bool(torch.cuda.is_available()):
        return None

    profiler_module = getattr(torch, "profiler", None)
    profile = getattr(profiler_module, "profile", None)
    if not callable(profile):
        return None

    activities = []
    profiler_activity = getattr(profiler_module, "ProfilerActivity", None)
    if profiler_activity is not None:
        cpu_activity = getattr(profiler_activity, "CPU", None)
        cuda_activity = getattr(profiler_activity, "CUDA", None)
        activities = [activity for activity in (cpu_activity, cuda_activity) if activity is not None]

    try:
        if activities:
            return profile(activities=activities)
        return profile()
    except Exception:
        return None


def _nvidia_smi_snapshot(
    runner: Any = subprocess.run,
) -> dict | None:
    command = [
        "nvidia-smi",
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        return None

    if completed.returncode != 0:
        return None

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not first_line:
        return None

    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 4:
        return None

    gpu_name, memory_total_mb, gpu_utilization_pct, gpu_temperature_c = parts
    return {
        "gpu_name": gpu_name or None,
        "memory_total_mb": _parse_float(memory_total_mb),
        "gpu_utilization_pct": _parse_float(gpu_utilization_pct),
        "gpu_temperature_c": _parse_float(gpu_temperature_c),
    }


def _bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes / 1024**2)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _mean_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return _mean(present)


def _max_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return max(present)
