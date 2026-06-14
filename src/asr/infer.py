from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from src.asr.model_loader import (
    DEFAULT_MODEL_ID,
    get_gpu_memory_allocated_mb,
    load_model,
)


DEFAULT_OUTPUT_PATH = "experiments/baseline/single_file_result.json"
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac"}
ALLOWED_TARGET_LANGS = {"en-US", "en-GB", "el-GR", "auto"}


def run_single_file_inference(
    audio_path: str,
    model_id_or_path: str,
    output_path: str | None = None,
    target_lang: str = "en-US",
    device: str = "cuda",
) -> dict:
    audio_file = _validate_audio_path(audio_path)
    _validate_target_lang(target_lang)

    result_output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
    warnings: list[str] = []
    language_readiness = _language_readiness(target_lang, warnings)

    model = load_model(model_id_or_path=model_id_or_path, device=device)
    transcribe = getattr(model, "transcribe", None)
    if not callable(transcribe):
        raise RuntimeError(
            "Loaded ASR model does not expose model.transcribe(). Check the installed "
            "NeMo/Nemotron 3.5 ASR inference API for the restored model class before "
            "running single-file inference."
        )

    transcribe_call, target_lang_support = _build_transcribe_call(
        transcribe=transcribe,
        audio_file=audio_file,
        target_lang=target_lang,
        warnings=warnings,
    )

    start = time.perf_counter()
    raw_transcript = transcribe(*transcribe_call.args, **transcribe_call.kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    transcript = extract_transcript_text(raw_transcript)
    gpu_memory_allocated_mb = get_gpu_memory_allocated_mb()

    result = {
        "audio_filepath": str(audio_file),
        "model_id_or_path": model_id_or_path,
        "target_lang": target_lang,
        "target_lang_support": target_lang_support,
        "language_readiness": language_readiness,
        "warnings": warnings,
        "transcript": transcript,
        "latency_ms": latency_ms,
        "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
        "device": device,
    }

    print(f"Audio file path: {audio_file}")
    print(f"Transcript text: {transcript}")
    print(f"latency_ms: {latency_ms:.2f}")
    print(f"gpu_memory_allocated_mb after inference: {gpu_memory_allocated_mb:.2f}")

    result_output_path.parent.mkdir(parents=True, exist_ok=True)
    result_output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run defensive single-file ASR inference with Nemotron 3.5 ASR.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to a .wav or .flac audio file.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--target-lang",
        default="en-US",
        choices=sorted(ALLOWED_TARGET_LANGS),
        help="Initial target language hint.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the JSON inference result.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Target device for loading, usually "cuda" or "cpu".',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_single_file_inference(
            audio_path=args.audio,
            model_id_or_path=args.model_id_or_path,
            output_path=args.output,
            target_lang=args.target_lang,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Single-file inference failed: {exc}", file=sys.stderr)
        print(
            "Check that the audio file exists, has a .wav or .flac extension, "
            "NeMo ASR is installed, CUDA/PyTorch is configured if using cuda, "
            "and Hugging Face/NVIDIA access is available for the Nemotron model.",
            file=sys.stderr,
        )
        return 1
    return 0


def extract_transcript_text(transcribe_result: Any) -> str:
    if isinstance(transcribe_result, str):
        return transcribe_result

    text = getattr(transcribe_result, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(transcribe_result, Sequence):
        extracted = [extract_transcript_text(item) for item in transcribe_result]
        return "\n".join(text for text in extracted if text)

    raise RuntimeError(
        "Could not extract transcript text from model.transcribe() result. Expected "
        "a string, an object with .text, a list of strings, or a list of objects with .text."
    )


def _validate_audio_path(audio_path: str) -> Path:
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_SUFFIXES))
        raise ValueError(f"Audio file must use one of these extensions: {allowed}")
    if not path.is_file():
        raise ValueError(f"Audio path must be a file: {path}")
    return path


def _validate_target_lang(target_lang: str) -> None:
    if target_lang not in ALLOWED_TARGET_LANGS:
        allowed = ", ".join(sorted(ALLOWED_TARGET_LANGS))
        raise ValueError(f"Unsupported target_lang {target_lang!r}. Allowed values: {allowed}")


def _language_readiness(target_lang: str, warnings: list[str]) -> str:
    if target_lang in {"en-US", "en-GB"}:
        return "transcription_ready"
    if target_lang == "el-GR":
        warnings.append(
            "Greek el-GR is adaptation-ready; quality is not confirmed until "
            "fine-tuning/evaluation is measured."
        )
        return "adaptation_ready"
    return "auto"


class _TranscribeCall:
    def __init__(self, args: list[Any], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs


def _build_transcribe_call(
    transcribe: Any,
    audio_file: Path,
    target_lang: str,
    warnings: list[str],
) -> tuple[_TranscribeCall, bool]:
    kwargs: dict[str, Any] = {}
    args: list[Any] = []

    audio_values = [str(audio_file)]
    if _call_has_named_parameter(transcribe, "audio"):
        kwargs["audio"] = audio_values
    elif _call_has_named_parameter(transcribe, "paths2audio_files"):
        kwargs["paths2audio_files"] = audio_values
    else:
        args.append(audio_values)

    target_lang_support = False
    if _call_accepts_parameter(transcribe, "target_lang"):
        kwargs["target_lang"] = target_lang
        target_lang_support = True
    else:
        warnings.append(
            "Installed model.transcribe() API does not expose target_lang; continuing "
            "without target-language conditioning."
        )

    return _TranscribeCall(args=args, kwargs=kwargs), target_lang_support


def _call_accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _call_has_named_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


if __name__ == "__main__":
    raise SystemExit(main())
