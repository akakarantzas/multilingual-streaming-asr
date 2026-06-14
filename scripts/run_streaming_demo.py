from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.model_loader import DEFAULT_MODEL_ID, load_model  # noqa: E402
from src.asr.streaming_asr import ChunkedASRSession  # noqa: E402
from src.audio.microphone import DEFAULT_SAMPLE_RATE, MicrophoneAudioStream  # noqa: E402


LANGUAGE_CHOICES = ("en", "el")
TARGET_LANG_CHOICES = ("en-US", "en-GB", "el-GR", "auto")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a terminal microphone streaming ASR demo.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Microphone chunk size in milliseconds.",
    )
    parser.add_argument(
        "--inference-window-ms",
        type=int,
        default=1000,
        help="How often to run ASR over the rolling buffer, in milliseconds.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGE_CHOICES,
        required=True,
        help="Display language label.",
    )
    parser.add_argument(
        "--target-lang",
        choices=TARGET_LANG_CHOICES,
        default="en-US",
        help="Target language hint for ASR when supported.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Target device for loading, usually "cuda" or "cpu".',
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional session limit for testing.",
    )
    return parser


def run_demo(args: argparse.Namespace) -> int:
    validate_language_target(args.language, args.target_lang)
    chunk_samples = chunk_ms_to_samples(args.chunk_ms, DEFAULT_SAMPLE_RATE)
    inference_window_s = args.inference_window_ms / 1000

    model = load_model(model_id_or_path=args.model_id_or_path, device=args.device)
    session = ChunkedASRSession(
        model=model,
        sample_rate=DEFAULT_SAMPLE_RATE,
        target_lang=args.target_lang,
        inference_interval_s=inference_window_s,
        log_events=False,
    )
    print_startup_header(
        model_id_or_path=args.model_id_or_path,
        target_lang=args.target_lang,
        chunk_ms=args.chunk_ms,
        inference_window_ms=args.inference_window_ms,
        language=args.language,
        language_readiness=session.language_readiness,
        readiness_warnings=session.session_warnings,
        device=args.device,
        gpu_name=get_gpu_name(),
        mode=session.mode,
    )

    events: list[dict] = []
    start = time.perf_counter()
    try:
        with MicrophoneAudioStream(chunk_size=chunk_samples) as microphone:
            for chunk in microphone:
                if args.max_seconds is not None and time.perf_counter() - start >= args.max_seconds:
                    break
                event = session.process_chunk(chunk)
                if event is None:
                    continue
                events.append(event)
                print_partial_event(event)
    except KeyboardInterrupt:
        print("\nStreaming demo interrupted by user.")
    finally:
        total_runtime_s = time.perf_counter() - start
        print()
        print_session_summary(summarize_session(events, total_runtime_s))
    return 0


def summarize_session(events: list[dict], total_runtime_s: float) -> dict:
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms") is not None]
    rtfs = [float(event["rtf"]) for event in events if event.get("rtf") is not None]
    return {
        "total_runtime_s": total_runtime_s,
        "inference_windows": len(events),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_rtf": _mean(rtfs),
    }


def print_startup_header(
    model_id_or_path: str,
    target_lang: str,
    chunk_ms: int,
    inference_window_ms: int,
    language: str,
    language_readiness: str,
    readiness_warnings: list[str],
    device: str,
    gpu_name: str | None,
    mode: str,
) -> None:
    print("Streaming ASR demo")
    print(f"model: {model_id_or_path}")
    print(f"target_lang: {target_lang}")
    print(f"chunk_ms: {chunk_ms}")
    print(f"inference_window_ms: {inference_window_ms}")
    print(f"language: {language}")
    print(f"language_readiness: {language_readiness}")
    for warning in readiness_warnings:
        print(f"warning: {warning}")
    print(f"device: {device}")
    print(f"gpu_name: {gpu_name or 'unavailable'}")
    print(f"mode: {mode}")
    print("Press Ctrl+C to stop. Partial transcripts are exploratory, not final utterances.")


def print_partial_event(event: dict) -> None:
    transcript = str(event.get("transcript", ""))
    latency_ms = float(event.get("latency_ms", 0.0))
    rtf = float(event.get("rtf", 0.0))
    message = f"\r{transcript}  latency_ms={latency_ms:.2f} rtf={rtf:.3f}"
    print(message, end="", flush=True)


def print_session_summary(summary: dict) -> None:
    print("Session summary")
    print(f"total_runtime_s: {_format_optional_float(summary['total_runtime_s'])}")
    print(f"inference_windows: {summary['inference_windows']}")
    print(f"avg_latency_ms: {_format_optional_float(summary['avg_latency_ms'])}")
    print(f"p50_latency_ms: {_format_optional_float(summary['p50_latency_ms'])}")
    print(f"p95_latency_ms: {_format_optional_float(summary['p95_latency_ms'])}")
    print(f"avg_rtf: {_format_optional_float(summary['avg_rtf'])}")


def validate_language_target(language: str, target_lang: str) -> None:
    if language == "en" and target_lang not in {"en-US", "en-GB", "auto"}:
        raise ValueError("English display language should use --target-lang en-US, en-GB, or auto")
    if language == "el" and target_lang not in {"el-GR", "auto"}:
        raise ValueError("Greek display language should use --target-lang el-GR or auto")


def chunk_ms_to_samples(chunk_ms: int, sample_rate: int) -> int:
    if chunk_ms <= 0:
        raise ValueError("--chunk-ms must be positive")
    return max(1, int(sample_rate * (chunk_ms / 1000)))


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile_value / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def get_gpu_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_demo(args)
    except (RuntimeError, ValueError, ImportError) as exc:
        print(f"Streaming demo failed: {exc}", file=sys.stderr)
        return 1


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
