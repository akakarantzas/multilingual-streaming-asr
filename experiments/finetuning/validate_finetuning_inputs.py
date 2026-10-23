from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.manifest import load_jsonl_manifest  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Greek fine-tuning manifests without starting training.",
    )
    parser.add_argument("--train-manifest", required=True, help="Path to the train JSONL manifest.")
    parser.add_argument("--val-manifest", required=True, help="Path to the validation JSONL manifest.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = validate_finetuning_inputs(args.train_manifest, args.val_manifest)
    except ValueError as exc:
        print(f"Fine-tuning input validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"train utterances: {result['train_utterances']}")
    print(f"validation utterances: {result['validation_utterances']}")
    print(f"approx train duration seconds: {_format_optional(result['train_duration_s'])}")
    print(f"approx validation duration seconds: {_format_optional(result['validation_duration_s'])}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    print("Validation complete. Training was not started.")
    return 0


def validate_finetuning_inputs(train_manifest: str, val_manifest: str) -> dict:
    train_rows = load_jsonl_manifest(train_manifest, validate_files=True)
    val_rows = load_jsonl_manifest(val_manifest, validate_files=True)

    warnings: list[str] = []
    if len(train_rows) < 100:
        warnings.append("Train manifest has fewer than 100 utterances; fine-tuning may be unstable.")
    if len(val_rows) < 20:
        warnings.append("Validation manifest has fewer than 20 utterances; validation may be noisy.")

    return {
        "train_utterances": len(train_rows),
        "validation_utterances": len(val_rows),
        "train_duration_s": total_duration(train_rows),
        "validation_duration_s": total_duration(val_rows),
        "warnings": warnings,
    }


def total_duration(rows: list[dict]) -> float | None:
    durations = [row.get("duration") for row in rows]
    numeric_durations = [duration for duration in durations if isinstance(duration, (int, float))]
    if len(numeric_durations) != len(rows):
        return None
    return float(sum(numeric_durations))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
