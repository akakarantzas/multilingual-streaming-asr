from __future__ import annotations

import pytest

from src.eval.cer import compute_cer
from src.eval.wer import compute_wer


def test_wer_exact_match() -> None:
    result = compute_wer(["hello world"], ["hello world"])

    assert result["wer"] == 0.0
    assert result["total_words"] == 2


def test_wer_insertion() -> None:
    result = compute_wer(["hello"], ["hello world"])

    assert result["insertions"] == 1
    assert result["wer"] == 1.0


def test_wer_deletion() -> None:
    result = compute_wer(["hello world"], ["hello"])

    assert result["deletions"] == 1
    assert result["wer"] == 0.5


def test_wer_substitution() -> None:
    result = compute_wer(["hello world"], ["hello there"])

    assert result["substitutions"] == 1
    assert result["wer"] == 0.5


def test_wer_empty_hypothesis() -> None:
    result = compute_wer(["hello world"], [""])

    assert result["deletions"] == 2
    assert result["wer"] == 1.0


def test_wer_greek_punctuation_normalization() -> None:
    result = compute_wer(["Γεια σου, κόσμε!"], ["γεια σου κόσμε"])

    assert result["wer"] == 0.0
    assert result["total_words"] == 3


def test_wer_result_includes_normalization_description() -> None:
    result = compute_wer(["hello"], ["hello"])

    assert "normalization" in result
    assert "lowercase" in result["normalization"]
    assert "punctuation" in result["normalization"]


def test_wer_mismatched_list_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_wer(["hello"], ["hello", "world"])


def test_cer_exact_match() -> None:
    result = compute_cer(["你好"], ["你好"])

    assert result["cer"] == 0.0
    assert result["total_chars"] == 2


def test_cer_insertion() -> None:
    result = compute_cer(["你好"], ["你好呀"])

    assert result["insertions"] == 1
    assert result["cer"] == 0.5


def test_cer_deletion() -> None:
    result = compute_cer(["你好"], ["你"])

    assert result["deletions"] == 1
    assert result["cer"] == 0.5


def test_cer_substitution() -> None:
    result = compute_cer(["你好"], ["你壞"])

    assert result["substitutions"] == 1
    assert result["cer"] == 0.5


def test_cer_empty_hypothesis() -> None:
    result = compute_cer(["你好"], [""])

    assert result["deletions"] == 2
    assert result["cer"] == 1.0


def test_cer_cantonese_spaces_stripped() -> None:
    result = compute_cer(["你 好 世 界"], ["你好世界"])

    assert result["cer"] == 0.0
    assert result["total_chars"] == 4


def test_cer_mismatched_list_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_cer(["你好"], ["你好", "世界"])
