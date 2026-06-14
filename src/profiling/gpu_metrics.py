from __future__ import annotations

import subprocess
from datetime import datetime, timezone
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
