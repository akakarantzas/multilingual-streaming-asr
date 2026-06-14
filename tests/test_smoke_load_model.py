from __future__ import annotations

from scripts.smoke_load_model import (
    build_arg_parser,
    build_troubleshooting_message,
)
from src.asr.model_loader import DEFAULT_MODEL_ID


def test_argument_parser_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.model_id_or_path == DEFAULT_MODEL_ID
    assert args.device == "cuda"
    assert args.revision is None


def test_argument_parser_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-id-or-path",
            "local-model.nemo",
            "--device",
            "cpu",
            "--revision",
            "main",
        ]
    )

    assert args.model_id_or_path == "local-model.nemo"
    assert args.device == "cpu"
    assert args.revision == "main"


def test_troubleshooting_message_mentions_required_guidance() -> None:
    message = build_troubleshooting_message(RuntimeError("example failure"))

    assert "example failure" in message
    assert "Confirm NeMo ASR is installed" in message
    assert "CUDA-compatible PyTorch" in message
    assert "Hugging Face/NVIDIA access" in message
    assert "local .nemo checkpoint path" in message
    assert "without DGX Spark" in message
