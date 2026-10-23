from __future__ import annotations

import json
from pathlib import Path

from experiments.finetuning.validate_finetuning_inputs import (
    total_duration,
    validate_finetuning_inputs,
)


def test_validate_finetuning_inputs_counts_utterances_and_duration(tmp_path) -> None:
    train_manifest = _write_manifest(tmp_path, "train.jsonl", count=2)
    val_manifest = _write_manifest(tmp_path, "val.jsonl", count=1)

    result = validate_finetuning_inputs(str(train_manifest), str(val_manifest))

    assert result["train_utterances"] == 2
    assert result["validation_utterances"] == 1
    assert result["train_duration_s"] == 3.0
    assert result["validation_duration_s"] == 1.5
    assert any("fewer than 100" in warning for warning in result["warnings"])
    assert any("fewer than 20" in warning for warning in result["warnings"])


def test_total_duration_is_unknown_when_any_row_lacks_duration() -> None:
    assert total_duration([{"duration": 1.5}, {"duration": None}]) is None


def _write_manifest(temp_dir: Path, filename: str, count: int) -> Path:
    manifest_path = temp_dir / filename
    rows = []
    for index in range(count):
        audio_path = temp_dir / f"{filename}_{index}.wav"
        audio_path.write_bytes(b"placeholder")
        rows.append(
            {
                "audio_filepath": str(audio_path),
                "text": "γεια σου",
                "duration": 1.5,
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path
