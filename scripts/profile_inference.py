from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.model_loader import DEFAULT_MODEL_ID, load_model  # noqa: E402
from src.eval.manifest import load_jsonl_manifest  # noqa: E402
from src.profiling.gpu_metrics import concurrency_test, profile_inference, snapshot  # noqa: E402


LANGUAGE_CHOICES = ("en", "el")
TARGET_LANG_CHOICES = ("en-US", "en-GB", "el-GR")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile local ASR inference over a manifest.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--language", required=True, choices=LANGUAGE_CHOICES)
    parser.add_argument(
        "--target-lang",
        default="en-US",
        choices=TARGET_LANG_CHOICES,
        help="Target language hint. Use en-US/en-GB for English and el-GR for Greek.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark JSON files.")
    parser.add_argument("--device", default="cuda", help='Target device, usually "cuda" or "cpu".')
    parser.add_argument(
        "--stream-counts",
        default="1,2,4,8",
        help="Comma-separated local concurrency stream counts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_language_target(args.language, args.target_lang)
        stream_counts = parse_stream_counts(args.stream_counts)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = load_jsonl_manifest(args.manifest, validate_files=True)
        audio_inputs = [row["audio_filepath"] for row in manifest_rows]

        gpu_snapshots = [snapshot()]
        model = load_model(model_id_or_path=args.model_id_or_path, device=args.device)
        profile_summary = profile_inference(
            model=model,
            audio_inputs=audio_inputs,
            label=f"{args.language}:{args.target_lang}",
            target_lang=args.target_lang,
        )
        gpu_snapshots.append(snapshot())
        concurrency_summary = concurrency_test(
            model=model,
            audio_inputs=audio_inputs,
            stream_counts=stream_counts,
            target_lang=args.target_lang,
        )
        gpu_snapshots.append(snapshot())

        _write_json(output_dir / "profile_summary.json", profile_summary)
        _write_json(output_dir / "concurrency_summary.json", concurrency_summary)
        _write_json(output_dir / "gpu_snapshots.json", gpu_snapshots)
        print_profile_summary(profile_summary, concurrency_summary, output_dir)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"Profiling failed: {exc}", file=sys.stderr)
        return 1
    return 0


def parse_stream_counts(value: str) -> list[int]:
    try:
        stream_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--stream-counts must be a comma-separated list of integers") from exc
    if not stream_counts or any(stream_count <= 0 for stream_count in stream_counts):
        raise ValueError("--stream-counts must contain positive integers")
    return stream_counts


def validate_language_target(language: str, target_lang: str) -> None:
    if language == "en" and target_lang not in {"en-US", "en-GB"}:
        raise ValueError("English profiling requires --target-lang en-US or en-GB")
    if language == "el" and target_lang != "el-GR":
        raise ValueError("Greek profiling requires --target-lang el-GR")


def print_profile_summary(profile_summary: dict, concurrency_summary: list[dict], output_dir: Path) -> None:
    print("Inference profile complete")
    print(f"output_dir: {output_dir}")
    print(f"num_inputs: {profile_summary.get('num_inputs')}")
    print(f"avg_latency_ms: {_format_optional_float(profile_summary.get('avg_latency_ms'))}")
    print(f"p95_latency_ms: {_format_optional_float(profile_summary.get('p95_latency_ms'))}")
    if concurrency_summary:
        last = concurrency_summary[-1]
        print(
            "max_stream_count: "
            f"{last.get('stream_count')} p95_latency_ms={_format_optional_float(last.get('p95_latency_ms'))}"
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
