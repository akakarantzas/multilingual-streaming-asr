from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def format_bytes_as_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GB"


def get_gpu_memory_allocated_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0

    if not torch.cuda.is_available():
        return 0.0

    return float(torch.cuda.memory_allocated() / 1024**2)


def load_model(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    revision: str | None = None,
) -> Any:
    print(f"Loading ASR model: {model_id_or_path}")
    if revision:
        print(f"Requested model revision: {revision}")

    try:
        import torch
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import NVIDIA NeMo ASR and PyTorch. Check that NeMo, PyTorch, "
            "and their CUDA-compatible dependencies are installed in the active environment."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            'device="cuda" was requested, but PyTorch reports CUDA is unavailable. '
            "Check the NVIDIA driver, CUDA-compatible PyTorch wheel, and GPU visibility; "
            'or pass device="cpu" for smoke testing.'
        )

    torch_device = torch.device(device)

    if _looks_like_local_nemo_path(model_id_or_path):
        model_path = _validate_local_nemo_path(model_id_or_path)
        if not hasattr(ASRModel, "restore_from"):
            raise RuntimeError(
                "Installed NeMo ASRModel does not expose restore_from(). Check the "
                "installed nemo_toolkit version and Nemotron 3.5 ASR loading docs."
            )
        model = ASRModel.restore_from(
            restore_path=str(model_path),
            map_location=torch_device,
        )
    else:
        if not hasattr(ASRModel, "from_pretrained"):
            raise RuntimeError(
                "Installed NeMo ASRModel does not expose from_pretrained(). Check the "
                "installed nemo_toolkit version and Nemotron 3.5 ASR loading docs."
            )
        pretrained_kwargs: dict[str, Any] = {
            "model_name": model_id_or_path,
            "map_location": torch_device,
        }
        if revision is not None:
            if _call_accepts_parameter(ASRModel.from_pretrained, "revision"):
                pretrained_kwargs["revision"] = revision
            else:
                raise RuntimeError(
                    "A model revision was requested, but the installed NeMo "
                    "ASRModel.from_pretrained() API does not expose a revision parameter. "
                    "Check the NeMo/Nemotron 3.5 ASR loading docs or download the desired "
                    "revision locally as a .nemo checkpoint."
                )
        model = ASRModel.from_pretrained(**pretrained_kwargs)

    if hasattr(model, "to"):
        model = model.to(torch_device)
    elif device != "cpu":
        raise RuntimeError(
            "Loaded model does not expose .to(), so it could not be moved to the requested "
            f"device {device!r}. Check the installed NeMo model type."
        )

    if hasattr(model, "eval"):
        model.eval()

    if torch.cuda.is_available():
        allocated_mb = get_gpu_memory_allocated_mb()
        allocated_bytes = int(allocated_mb * 1024**2)
        print(
            "GPU memory allocated after model load: "
            f"{allocated_mb:.2f} MB ({format_bytes_as_gb(allocated_bytes)})"
        )

    return model


def _looks_like_local_nemo_path(model_id_or_path: str) -> bool:
    path = Path(model_id_or_path).expanduser()
    if path.suffix == ".nemo":
        return True
    if path.exists():
        return True
    if path.is_absolute():
        return True
    return model_id_or_path.startswith((".", "~")) or "\\" in model_id_or_path


def _validate_local_nemo_path(model_id_or_path: str) -> Path:
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local NeMo checkpoint does not exist: {path}")
    if path.suffix != ".nemo":
        raise ValueError(f"Local NeMo checkpoint must have a .nemo suffix: {path}")
    if not path.is_file():
        raise ValueError(f"Local NeMo checkpoint must be a file: {path}")
    return path


def _call_accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters
