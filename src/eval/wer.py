from __future__ import annotations

import json
import string
import unicodedata

import jiwer


NORMALIZATION_DESCRIPTION = (
    "unicode NFC, lowercase, strip punctuation by Unicode category, normalize whitespace"
)


def compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    normalized_references = [_normalize_for_wer(reference) for reference in references]
    normalized_hypotheses = [_normalize_for_wer(hypothesis) for hypothesis in hypotheses]

    # TODO: Add raw/cased/punctuated WER alongside normalized WER for Nemotron
    # punctuation and capitalization analysis.
    word_output = jiwer.process_words(normalized_references, normalized_hypotheses)
    total_words = word_output.hits + word_output.substitutions + word_output.deletions

    return {
        "wer": float(word_output.wer),
        "substitutions": int(word_output.substitutions),
        "deletions": int(word_output.deletions),
        "insertions": int(word_output.insertions),
        "total_words": int(total_words),
        "normalization": NORMALIZATION_DESCRIPTION,
    }


def _normalize_for_wer(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower()
    without_punctuation = "".join(
        " " if _is_punctuation(char) else char for char in normalized
    )
    return " ".join(without_punctuation.split())


def _is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


if __name__ == "__main__":
    result = compute_wer(["Hello, world!"], ["hello world"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
