from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.infer import transcribe_file
from src.asr.model_loader import (
    DEFAULT_MODEL_ID,
    get_gpu_name,
    load_model,
    resolve_device,
)
from src.eval.cer import compute_cer
from src.eval.manifest import load_jsonl_manifest
from src.eval.wer import compute_wer
from src.profiling.stats import mean as _mean
from src.profiling.stats import percentile


LANGUAGE_CHOICES = ("el",)
TARGET_LANG_CHOICES = ("el-GR",)


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
        default="el-GR",
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
        default="auto",
        help='Target device: "auto" (cuda if available, else cpu), "cuda", or "cpu".',
    )
    parser.add_argument(
        "--validate-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check that manifest audio files exist (default: on).",
    )
    return parser


def run_batch_eval(
    manifest_path: str,
    model_id_or_path: str,
    language: str,
    target_lang: str,
    output: str,
    summary_output: str | None = None,
    device: str = "auto",
    validate_files: bool = True,
    inference_fn: Callable[[dict], dict] | None = None,
) -> dict:
    _validate_language_target(language, target_lang)
    device = resolve_device(device)
    manifest_rows = load_jsonl_manifest(manifest_path, validate_files=validate_files)
    language_readiness, readiness_warnings = language_readiness_for(language)

    if inference_fn is None:
        model = load_model(model_id_or_path=model_id_or_path, device=device)

        def inference_fn(row: dict) -> dict:
            return transcribe_file(
                model=model,
                audio_file=Path(row["audio_filepath"]),
                target_lang=target_lang,
            )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_file_rows: list[dict] = []
    warnings: list[str] = list(readiness_warnings)
    try:
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
                    "wer_no_accents": None,
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
    except BaseException:
        # Keep finished transcripts (also on Ctrl+C); no metrics or summary for a partial run.
        if per_file_rows:
            _write_csv(output_path, per_file_rows)
            print(
                f"Partial results ({len(per_file_rows)} of {len(manifest_rows)} files) "
                f"written to {output_path}",
                file=sys.stderr,
            )
        raise

    summary = build_summary(
        per_file_rows=per_file_rows,
        language=language,
        model_id_or_path=model_id_or_path,
        target_lang=target_lang,
        language_readiness=language_readiness,
        manifest_path=manifest_path,
        warnings=warnings,
    )
    summary["device"] = device
    summary["gpu_name"] = get_gpu_name()
    _attach_metric_to_rows(per_file_rows, summary)
    _write_csv(output_path, per_file_rows)

    summary_path = Path(summary_output or f"experiments/baseline/{language}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(summary)
    return summary


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

    if language == "el":
        metric = compute_wer(references, hypotheses)
        metric_name = "wer"
        metric_value = metric["wer"]
        metric_value_no_accents = compute_wer(
            references, hypotheses, strip_accents=True
        )["wer"]
    elif language == "yue":
        # Planned Cantonese hook: not reachable from the CLI while LANGUAGE_CHOICES is Greek-only.
        metric = compute_cer(references, hypotheses)
        metric_name = "cer"
        metric_value = metric["cer"]
        metric_value_no_accents = None
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
        "metric_value_no_accents": (
            None if metric_value_no_accents is None else float(metric_value_no_accents)
        ),
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


def print_summary(summary: dict) -> None:
    columns = [
        "language",
        "num_files",
        "metric_name",
        "metric_value",
        "metric_value_no_accents",
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
            row["wer_no_accents"] = compute_wer(
                [reference], [hypothesis], strip_accents=True
            )["wer"]
        elif metric_name == "cer":
            row[metric_name] = compute_cer([reference], [hypothesis])["cer"]
        else:
            raise ValueError(f"Unsupported metric_name: {metric_name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
