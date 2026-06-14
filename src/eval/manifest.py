from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_manifest(path: str, validate_files: bool = True) -> list[dict]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise ValueError(f"Manifest path does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Manifest path must be a file: {manifest_path}")

    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                raw_row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in manifest {manifest_path} on line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(raw_row, dict):
                raise ValueError(
                    f"Invalid manifest row on line {line_number}: expected a JSON object"
                )
            rows.append(
                validate_manifest_row(
                    raw_row,
                    line_number=line_number,
                    validate_files=validate_files,
                )
            )
    return rows


def validate_manifest_row(
    row: dict,
    line_number: int | None = None,
    validate_files: bool = True,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError(_line_error("expected row to be a dict", line_number))

    audio_filepath = row.get("audio_filepath")
    if not isinstance(audio_filepath, str) or not audio_filepath.strip():
        raise ValueError(
            _line_error("audio_filepath is required and must be a non-empty string", line_number)
        )

    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(_line_error("text is required and must be a non-empty string", line_number))

    duration = row.get("duration")
    if duration is not None and not _is_number(duration):
        raise ValueError(_line_error("duration must be numeric when present", line_number))

    normalized_audio_filepath = str(Path(audio_filepath).expanduser())
    if validate_files and not Path(normalized_audio_filepath).exists():
        raise ValueError(
            _line_error(f"audio_filepath does not exist: {normalized_audio_filepath}", line_number)
        )

    return {
        "audio_filepath": normalized_audio_filepath,
        "text": text,
        "duration": duration,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _line_error(message: str, line_number: int | None) -> str:
    if line_number is None:
        return f"Invalid manifest row: {message}"
    return f"Invalid manifest row on line {line_number}: {message}"
