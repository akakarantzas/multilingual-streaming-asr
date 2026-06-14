from __future__ import annotations

import pytest

from src.asr.infer import (
    _language_readiness,
    extract_transcript_text,
)


class TranscriptObject:
    def __init__(self, text: str) -> None:
        self.text = text


def test_extract_transcript_from_string() -> None:
    assert extract_transcript_text("hello world") == "hello world"


def test_extract_transcript_from_object_with_text() -> None:
    assert extract_transcript_text(TranscriptObject("object text")) == "object text"


def test_extract_transcript_from_list_of_strings() -> None:
    assert extract_transcript_text(["first", "second"]) == "first\nsecond"


def test_extract_transcript_from_list_of_objects_with_text() -> None:
    result = [TranscriptObject("first"), TranscriptObject("second")]

    assert extract_transcript_text(result) == "first\nsecond"


def test_extract_transcript_rejects_unknown_result_type() -> None:
    with pytest.raises(RuntimeError, match="Could not extract transcript text"):
        extract_transcript_text(object())


def test_language_readiness_marks_english_as_transcription_ready() -> None:
    warnings: list[str] = []

    assert _language_readiness("en-US", warnings) == "transcription_ready"
    assert warnings == []


def test_language_readiness_warns_for_greek_exploratory_use() -> None:
    warnings: list[str] = []

    assert _language_readiness("el-GR", warnings) == "adaptation_ready"
    assert "Greek el-GR is adaptation-ready" in warnings[0]
