from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.model_loader import DEFAULT_MODEL_ID  # noqa: E402
from src.eval.manifest import load_jsonl_manifest  # noqa: E402


REQUIRED_CONFIG_FIELDS = {
    "base_model_id_or_path",
    "train_manifest",
    "val_manifest",
    "output_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "notes",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Greek fine-tuning preparation inputs without starting training.",
    )
    parser.add_argument("--config", required=True, help="Path to fine-tuning config JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = validate_finetuning_inputs(args.config)
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


def validate_finetuning_inputs(config_path: str) -> dict:
    config = load_config(config_path)
    missing_fields = sorted(REQUIRED_CONFIG_FIELDS.difference(config))
    if missing_fields:
        raise ValueError(f"Config is missing required fields: {', '.join(missing_fields)}")

    validate_base_model_id_or_path(config["base_model_id_or_path"])
    train_rows = load_jsonl_manifest(config["train_manifest"], validate_files=True)
    val_rows = load_jsonl_manifest(config["val_manifest"], validate_files=True)

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


def load_config(config_path: str) -> dict:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise ValueError(f"Config path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Config path must be a file: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config is not valid JSON: {exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError("Config JSON must be an object")
    return config


def validate_base_model_id_or_path(base_model_id_or_path: Any) -> None:
    if not isinstance(base_model_id_or_path, str) or not base_model_id_or_path.strip():
        raise ValueError("base_model_id_or_path must be a non-empty string")
    if base_model_id_or_path == DEFAULT_MODEL_ID:
        return

    path = Path(base_model_id_or_path).expanduser()
    if not path.exists():
        raise ValueError(
            "base_model_id_or_path must be the Nemotron model ID or an existing local .nemo file"
        )
    if path.suffix != ".nemo":
        raise ValueError("Local base_model_id_or_path must end with .nemo")
    if not path.is_file():
        raise ValueError("Local base_model_id_or_path must be a .nemo file")


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
