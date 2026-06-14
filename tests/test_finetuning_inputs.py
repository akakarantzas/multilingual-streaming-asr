from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from experiments.finetuning.validate_finetuning_inputs import (
    load_config,
    validate_base_model_id_or_path,
    validate_finetuning_inputs,
)
from src.asr.model_loader import DEFAULT_MODEL_ID


def test_validate_finetuning_inputs_accepts_valid_config_with_model_id() -> None:
    with _workspace_temp_dir() as temp_dir:
        train_manifest = _write_manifest(temp_dir, "train.jsonl", count=2)
        val_manifest = _write_manifest(temp_dir, "val.jsonl", count=1)
        config_path = _write_config(
            temp_dir,
            {
                "base_model_id_or_path": DEFAULT_MODEL_ID,
                "train_manifest": str(train_manifest),
                "val_manifest": str(val_manifest),
                "output_dir": str(temp_dir / "output"),
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 1e-5,
                "seed": 42,
                "notes": "test",
            },
        )

        result = validate_finetuning_inputs(str(config_path))

    assert result["train_utterances"] == 2
    assert result["validation_utterances"] == 1
    assert result["train_duration_s"] == 3.0
    assert result["validation_duration_s"] == 1.5
    assert any("fewer than 100" in warning for warning in result["warnings"])
    assert any("fewer than 20" in warning for warning in result["warnings"])


def test_validate_base_model_accepts_existing_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.nemo"
        checkpoint.write_text("placeholder", encoding="utf-8")

        validate_base_model_id_or_path(str(checkpoint))


def test_validate_base_model_rejects_missing_local_file() -> None:
    with pytest.raises(ValueError, match="existing local .nemo"):
        validate_base_model_id_or_path("missing-model.nemo")


def test_validate_base_model_rejects_non_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.write_text("placeholder", encoding="utf-8")

        with pytest.raises(ValueError, match=".nemo"):
            validate_base_model_id_or_path(str(checkpoint))


def test_load_config_rejects_invalid_json() -> None:
    with _workspace_temp_dir() as temp_dir:
        config_path = temp_dir / "config.json"
        config_path.write_text("{", encoding="utf-8")

        with pytest.raises(ValueError, match="valid JSON"):
            load_config(str(config_path))


def test_validate_finetuning_inputs_rejects_missing_required_field() -> None:
    with _workspace_temp_dir() as temp_dir:
        config_path = _write_config(temp_dir, {"base_model_id_or_path": DEFAULT_MODEL_ID})

        with pytest.raises(ValueError, match="missing required fields"):
            validate_finetuning_inputs(str(config_path))


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


def _write_config(temp_dir: Path, content: dict) -> Path:
    config_path = temp_dir / "config.json"
    config_path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return config_path


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="finetuning_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)
