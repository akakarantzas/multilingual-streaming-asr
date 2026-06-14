from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def compute_cer(references: list[str], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    total_counts = EditCounts(substitutions=0, deletions=0, insertions=0)
    total_chars = 0

    for reference, hypothesis in zip(references, hypotheses):
        normalized_reference = _strip_spaces(reference)
        normalized_hypothesis = _strip_spaces(hypothesis)
        counts = _edit_counts(normalized_reference, normalized_hypothesis)
        total_counts = EditCounts(
            substitutions=total_counts.substitutions + counts.substitutions,
            deletions=total_counts.deletions + counts.deletions,
            insertions=total_counts.insertions + counts.insertions,
        )
        total_chars += len(normalized_reference)

    cer = total_counts.total_errors / total_chars if total_chars else 0.0
    return {
        "cer": float(cer),
        "substitutions": total_counts.substitutions,
        "deletions": total_counts.deletions,
        "insertions": total_counts.insertions,
        "total_chars": total_chars,
    }


def _strip_spaces(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def _edit_counts(reference: str, hypothesis: str) -> EditCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs: list[list[tuple[int, EditCounts]]] = [
        [(0, EditCounts(0, 0, 0)) for _ in range(cols)] for _ in range(rows)
    ]

    for row in range(1, rows):
        counts = EditCounts(0, row, 0)
        costs[row][0] = (row, counts)
    for col in range(1, cols):
        counts = EditCounts(0, 0, col)
        costs[0][col] = (col, counts)

    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                costs[row][col] = costs[row - 1][col - 1]
                continue

            substitute = _increment(costs[row - 1][col - 1], substitutions=1)
            delete = _increment(costs[row - 1][col], deletions=1)
            insert = _increment(costs[row][col - 1], insertions=1)
            costs[row][col] = min(
                (substitute, delete, insert),
                key=lambda item: (
                    item[0],
                    item[1].substitutions,
                    item[1].deletions,
                    item[1].insertions,
                ),
            )

    return costs[-1][-1][1]


def _increment(
    current: tuple[int, EditCounts],
    substitutions: int = 0,
    deletions: int = 0,
    insertions: int = 0,
) -> tuple[int, EditCounts]:
    distance, counts = current
    return (
        distance + substitutions + deletions + insertions,
        EditCounts(
            substitutions=counts.substitutions + substitutions,
            deletions=counts.deletions + deletions,
            insertions=counts.insertions + insertions,
        ),
    )


if __name__ == "__main__":
    result = compute_cer(["你好 世界"], ["你好世 界"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
