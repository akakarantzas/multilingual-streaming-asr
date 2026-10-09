from __future__ import annotations

import json
import string
import unicodedata

import jiwer


NORMALIZATION_DESCRIPTION = (
    "unicode NFC, lowercase, final sigma folded to sigma, strip punctuation by "
    "Unicode category, normalize whitespace"
)
ACCENT_INSENSITIVE_DESCRIPTION = ", strip Greek accents (tonos, dialytika)"


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    strip_accents: bool = False,
) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    normalized_references = [_normalize_for_wer(reference, strip_accents) for reference in references]
    normalized_hypotheses = [_normalize_for_wer(hypothesis, strip_accents) for hypothesis in hypotheses]

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
        "normalization": NORMALIZATION_DESCRIPTION
        + (ACCENT_INSENSITIVE_DESCRIPTION if strip_accents else ""),
    }


def _normalize_for_wer(text: str, strip_accents: bool = False) -> str:
    normalized = unicodedata.normalize("NFC", text).lower().replace("ς", "σ")
    if strip_accents:
        decomposed = unicodedata.normalize("NFD", normalized)
        normalized = unicodedata.normalize(
            "NFC", "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        )
    without_punctuation = "".join(
        " " if _is_punctuation(char) else char for char in normalized
    )
    return " ".join(without_punctuation.split())


def _is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


if __name__ == "__main__":
    result = compute_wer(["Hello, world!"], ["hello world"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
