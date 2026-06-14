from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.asr.model_loader import (
    DEFAULT_MODEL_ID,
    _looks_like_local_nemo_path,
    _validate_local_nemo_path,
    format_bytes_as_gb,
)


def test_default_model_id_is_treated_as_hugging_face_id() -> None:
    assert not _looks_like_local_nemo_path(DEFAULT_MODEL_ID)


def test_nemo_suffix_is_treated_as_local_path() -> None:
    assert _looks_like_local_nemo_path("model.nemo")


def test_windows_path_is_treated_as_local_path() -> None:
    assert _looks_like_local_nemo_path(r"checkpoints\model.nemo")


def test_validate_local_nemo_path_accepts_existing_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.nemo"
        checkpoint.write_text("placeholder")

        assert _validate_local_nemo_path(str(checkpoint)) == checkpoint


def test_validate_local_nemo_path_rejects_missing_checkpoint() -> None:
    with _workspace_temp_dir() as temp_dir:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validate_local_nemo_path(str(temp_dir / "missing.nemo"))


def test_validate_local_nemo_path_rejects_non_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.write_text("placeholder")

        with pytest.raises(ValueError, match=".nemo suffix"):
            _validate_local_nemo_path(str(checkpoint))


def test_validate_local_nemo_path_rejects_directory() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.nemo"
        checkpoint.mkdir()

        with pytest.raises(ValueError, match="must be a file"):
            _validate_local_nemo_path(str(checkpoint))


def test_format_bytes_as_gb() -> None:
    assert format_bytes_as_gb(0) == "0.00 GB"
    assert format_bytes_as_gb(1073741824) == "1.00 GB"


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="model_loader_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)
