from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.eval.manifest import load_jsonl_manifest


def test_valid_manifest() -> None:
    with _workspace_temp_dir() as temp_dir:
        audio_file = temp_dir / "sample.wav"
        audio_file.write_bytes(b"placeholder")
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(
            manifest_file,
            [{"audio_filepath": str(audio_file), "text": "ground truth", "duration": 3.2}],
        )

        rows = load_jsonl_manifest(str(manifest_file))

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": 3.2,
        }
    ]


def test_relative_audio_path_resolves_from_manifest_directory() -> None:
    with _workspace_temp_dir() as temp_dir:
        manifest_dir = temp_dir / "manifests"
        sample_dir = temp_dir / "samples"
        manifest_dir.mkdir()
        sample_dir.mkdir()
        audio_file = sample_dir / "sample.wav"
        audio_file.write_bytes(b"placeholder")
        manifest_file = manifest_dir / "manifest.jsonl"
        _write_jsonl(
            manifest_file,
            [{"audio_filepath": "../samples/sample.wav", "text": "ground truth"}],
        )

        rows = load_jsonl_manifest(str(manifest_file))

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": None,
        }
    ]


def test_missing_audio_filepath() -> None:
    with _workspace_temp_dir() as temp_dir:
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(manifest_file, [{"text": "ground truth"}])

        with pytest.raises(ValueError, match="line 1.*audio_filepath"):
            load_jsonl_manifest(str(manifest_file), validate_files=False)


def test_missing_text() -> None:
    with _workspace_temp_dir() as temp_dir:
        audio_file = temp_dir / "sample.wav"
        audio_file.write_bytes(b"placeholder")
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(manifest_file, [{"audio_filepath": str(audio_file)}])

        with pytest.raises(ValueError, match="line 1.*text"):
            load_jsonl_manifest(str(manifest_file))


def test_invalid_duration() -> None:
    with _workspace_temp_dir() as temp_dir:
        audio_file = temp_dir / "sample.wav"
        audio_file.write_bytes(b"placeholder")
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(
            manifest_file,
            [{"audio_filepath": str(audio_file), "text": "ground truth", "duration": "3.2"}],
        )

        with pytest.raises(ValueError, match="line 1.*duration"):
            load_jsonl_manifest(str(manifest_file))


def test_invalid_json_line() -> None:
    with _workspace_temp_dir() as temp_dir:
        manifest_file = temp_dir / "manifest.jsonl"
        manifest_file.write_text('{"audio_filepath": "sample.wav", "text": ', encoding="utf-8")

        with pytest.raises(ValueError, match="line 1"):
            load_jsonl_manifest(str(manifest_file), validate_files=False)


def test_missing_audio_file_when_validate_files_true() -> None:
    with _workspace_temp_dir() as temp_dir:
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(
            manifest_file,
            [{"audio_filepath": str(temp_dir / "missing.wav"), "text": "ground truth"}],
        )

        with pytest.raises(ValueError, match="line 1.*does not exist"):
            load_jsonl_manifest(str(manifest_file), validate_files=True)


def test_allows_missing_audio_file_when_validate_files_false() -> None:
    with _workspace_temp_dir() as temp_dir:
        audio_file = temp_dir / "missing.wav"
        manifest_file = temp_dir / "manifest.jsonl"
        _write_jsonl(
            manifest_file,
            [{"audio_filepath": str(audio_file), "text": "ground truth"}],
        )

        rows = load_jsonl_manifest(str(manifest_file), validate_files=False)

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": None,
        }
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="manifest_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)
