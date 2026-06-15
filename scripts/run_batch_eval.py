from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.infer import (
    _build_transcribe_call,
    extract_transcript_text,
)
from src.asr.model_loader import DEFAULT_MODEL_ID, get_gpu_memory_allocated_mb, load_model
from src.eval.cer import compute_cer
from src.eval.manifest import load_jsonl_manifest
from src.eval.wer import compute_wer


LANGUAGE_CHOICES = ("en", "el")
TARGET_LANG_CHOICES = ("en-US", "en-GB", "el-GR")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch ASR inference and evaluation over a JSONL manifest.",
    )
    parser.add_argument("--manifest", required=True, help="Path to a JSONL manifest.")
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=LANGUAGE_CHOICES,
        help="Evaluation language. Cantonese (Yue) is intentionally disabled for now.",
    )
    parser.add_argument(
        "--target-lang",
        default="en-US",
        choices=TARGET_LANG_CHOICES,
        help="Target language hint passed to inference when supported.",
    )
    parser.add_argument("--output", required=True, help="Output per-file CSV path.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Output summary JSON path. Defaults to experiments/baseline/{language}_summary.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Target device for loading, usually "cuda" or "cpu".',
    )
    parser.add_argument(
        "--validate-files",
        dest="validate_files",
        action="store_true",
        default=True,
        help="Validate that manifest audio files exist.",
    )
    parser.add_argument(
        "--no-validate-files",
        dest="validate_files",
        action="store_false",
        help="Do not validate that manifest audio files exist.",
    )
    return parser


def run_batch_eval(
    manifest_path: str,
    model_id_or_path: str,
    language: str,
    target_lang: str,
    output: str,
    summary_output: str | None = None,
    device: str = "cuda",
    validate_files: bool = True,
    inference_fn: Callable[[dict], dict] | None = None,
) -> dict:
    _validate_language_target(language, target_lang)
    manifest_rows = load_jsonl_manifest(manifest_path, validate_files=validate_files)
    language_readiness, readiness_warnings = language_readiness_for(language)

    if inference_fn is None:
        model = load_model(model_id_or_path=model_id_or_path, device=device)

        def inference_fn(row: dict) -> dict:
            return run_single_file_with_loaded_model(
                model=model,
                audio_filepath=row["audio_filepath"],
                target_lang=target_lang,
            )

    per_file_rows: list[dict] = []
    warnings: list[str] = list(readiness_warnings)
    for row in manifest_rows:
        inference_result = inference_fn(row)
        row_warnings = _coerce_warnings(inference_result.get("warnings", []))
        warnings.extend(row_warnings)
        per_file_rows.append(
            {
                "audio_filepath": row["audio_filepath"],
                "reference_text": row["text"],
                "hypothesis_text": str(inference_result.get("transcript", "")),
                "language": language,
                "language_readiness": language_readiness,
                "wer": None,
                "cer": None,
                "latency_ms": _coerce_float(inference_result.get("latency_ms")),
                "gpu_memory_allocated_mb": _coerce_float(
                    inference_result.get("gpu_memory_allocated_mb")
                ),
                "duration": row.get("duration"),
                "model_id_or_path": model_id_or_path,
                "device": device,
                "warnings": json.dumps(row_warnings, ensure_ascii=False),
            }
        )

    summary = build_summary(
        per_file_rows=per_file_rows,
        language=language,
        model_id_or_path=model_id_or_path,
        target_lang=target_lang,
        language_readiness=language_readiness,
        manifest_path=manifest_path,
        warnings=warnings,
    )
    _attach_metric_to_rows(per_file_rows, summary)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_file_rows).to_csv(output_path, index=False)

    summary_path = Path(summary_output or f"experiments/baseline/{language}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(summary)
    return summary


def run_single_file_with_loaded_model(
    model: Any,
    audio_filepath: str,
    target_lang: str,
) -> dict:
    transcribe = getattr(model, "transcribe", None)
    if not callable(transcribe):
        raise RuntimeError(
            "Loaded ASR model does not expose model.transcribe(). Check the installed "
            "NeMo/Nemotron 3.5 ASR inference API for the restored model class."
        )

    warnings: list[str] = []
    transcribe_call, target_lang_support = _build_transcribe_call(
        transcribe=transcribe,
        audio_file=Path(audio_filepath),
        target_lang=target_lang,
        warnings=warnings,
    )
    start = time.perf_counter()
    raw_transcript = transcribe(*transcribe_call.args, **transcribe_call.kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "transcript": extract_transcript_text(raw_transcript),
        "latency_ms": latency_ms,
        "gpu_memory_allocated_mb": get_gpu_memory_allocated_mb(),
        "target_lang_support": target_lang_support,
        "warnings": warnings,
    }


def build_summary(
    per_file_rows: list[dict],
    language: str,
    model_id_or_path: str,
    target_lang: str,
    language_readiness: str,
    manifest_path: str,
    warnings: list[str] | None = None,
) -> dict:
    references = [str(row["reference_text"]) for row in per_file_rows]
    hypotheses = [str(row["hypothesis_text"]) for row in per_file_rows]

    if language in {"en", "el"}:
        metric = compute_wer(references, hypotheses)
        metric_name = "wer"
        metric_value = metric["wer"]
    elif language == "yue":
        metric = compute_cer(references, hypotheses)
        metric_name = "cer"
        metric_value = metric["cer"]
    else:
        raise ValueError(f"Unsupported language: {language}")

    latencies = [
        float(row["latency_ms"])
        for row in per_file_rows
        if row.get("latency_ms") is not None
    ]

    return {
        "language": language,
        "num_files": len(per_file_rows),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "substitutions": int(metric["substitutions"]),
        "deletions": int(metric["deletions"]),
        "insertions": int(metric["insertions"]),
        "model_id_or_path": model_id_or_path,
        "target_lang": target_lang,
        "language_readiness": language_readiness,
        "warnings": sorted(set(warnings or [])),
        "manifest_path": manifest_path,
    }


def language_readiness_for(language: str) -> tuple[str, list[str]]:
    if language == "en":
        return "transcription_ready", []
    if language == "el":
        return (
            "adaptation_ready",
            [
                "Greek el evaluation is exploratory/adaptation-ready until a confirmed "
                "fine-tuned model is evaluated."
            ],
        )
    if language == "yue":
        return (
            "future_extension_disabled",
            ["Cantonese (Yue) CER is implemented but not enabled for the core milestone."],
        )
    raise ValueError(f"Unsupported language: {language}")


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


def print_summary(summary: dict) -> None:
    columns = [
        "language",
        "num_files",
        "metric_name",
        "metric_value",
        "avg_latency_ms",
        "substitutions",
        "deletions",
        "insertions",
    ]
    print(" | ".join(columns))
    print(" | ".join("-" * len(column) for column in columns))
    print(" | ".join(_format_summary_value(summary.get(column)) for column in columns))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_batch_eval(
            manifest_path=args.manifest,
            model_id_or_path=args.model_id_or_path,
            language=args.language,
            target_lang=args.target_lang,
            output=args.output,
            summary_output=args.summary_output,
            device=args.device,
            validate_files=args.validate_files,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"Batch evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _validate_language_target(language: str, target_lang: str) -> None:
    if language == "en" and target_lang not in {"en-US", "en-GB"}:
        raise ValueError("English evaluation requires --target-lang en-US or en-GB")
    if language == "el" and target_lang != "el-GR":
        raise ValueError("Greek evaluation requires --target-lang el-GR")
    if language not in LANGUAGE_CHOICES:
        raise ValueError(f"Unsupported language: {language}")


def _attach_metric_to_rows(per_file_rows: list[dict], summary: dict) -> None:
    metric_name = summary["metric_name"]
    for row in per_file_rows:
        reference = str(row["reference_text"])
        hypothesis = str(row["hypothesis_text"])
        if metric_name == "wer":
            row[metric_name] = compute_wer([reference], [hypothesis])["wer"]
        elif metric_name == "cer":
            row[metric_name] = compute_cer([reference], [hypothesis])["cer"]
        else:
            raise ValueError(f"Unsupported metric_name: {metric_name}")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
