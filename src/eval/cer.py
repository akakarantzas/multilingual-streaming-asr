from __future__ import annotations

import jiwer


def compute_cer(references: list[str], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    normalized_references = [_strip_spaces(reference) for reference in references]
    normalized_hypotheses = [_strip_spaces(hypothesis) for hypothesis in hypotheses]
    char_output = jiwer.process_characters(normalized_references, normalized_hypotheses)
    return {
        "cer": float(char_output.cer),
        "substitutions": int(char_output.substitutions),
        "deletions": int(char_output.deletions),
        "insertions": int(char_output.insertions),
        "total_chars": sum(len(reference) for reference in normalized_references),
    }


def _strip_spaces(text: str) -> str:
    return "".join(char for char in text if not char.isspace())
