"""Dependency-light classification metrics for reviewed labels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMetrics:
    """One-vs-rest metrics for a label."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    """Confusion matrix and macro metrics for an evaluated sample."""

    labels: tuple[str, ...]
    sample_size: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_labels(
    expected: list[str], predicted: list[str], *, labels: list[str] | None = None
) -> EvaluationReport:
    """Calculate precision, recall, F1, and a fixed-order confusion matrix."""

    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected and predicted labels must have equal, non-zero length.")
    selected = tuple(labels or sorted(set(expected) | set(predicted)))
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("Evaluation labels must be non-empty and unique.")
    unknown = (set(expected) | set(predicted)) - set(selected)
    if unknown:
        raise ValueError(
            "Observed labels are missing from the configured label set: "
            f"{sorted(unknown)}"
        )

    positions = {label: index for index, label in enumerate(selected)}
    matrix = [[0 for _ in selected] for _ in selected]
    for truth, guess in zip(expected, predicted, strict=True):
        matrix[positions[truth]][positions[guess]] += 1

    metrics: dict[str, ClassMetrics] = {}
    for index, label in enumerate(selected):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        precision = _divide(true_positive, true_positive + false_positive)
        recall = _divide(true_positive, true_positive + false_negative)
        f1 = _divide(2 * precision * recall, precision + recall) if precision + recall else 0.0
        metrics[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=sum(matrix[index]),
        )

    count = len(selected)
    return EvaluationReport(
        labels=selected,
        sample_size=len(expected),
        confusion_matrix=tuple(tuple(row) for row in matrix),
        per_class=metrics,
        macro_precision=sum(item.precision for item in metrics.values()) / count,
        macro_recall=sum(item.recall for item in metrics.values()) / count,
        macro_f1=sum(item.f1 for item in metrics.values()) / count,
    )
