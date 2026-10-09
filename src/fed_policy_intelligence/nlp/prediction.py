"""Validated prediction records shared by NLP classifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PredictionError(ValueError):
    """Raised when a classifier returns an invalid prediction."""


@dataclass(frozen=True)
class ClassificationPrediction:
    """One auditable document-level classification result."""

    label: str
    probabilities: dict[str, float]
    chunk_count: int
    token_count: int
    aggregation: str
    model_id: str
    tokenizer_id: str
    device: str

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise PredictionError("A prediction must contain at least one probability.")
        if self.label not in self.probabilities:
            raise PredictionError("The winning label must be present in probabilities.")
        if self.chunk_count < 1 or self.token_count < 1:
            raise PredictionError("Chunk and token counts must be positive.")
        values = list(self.probabilities.values())
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise PredictionError("Probabilities must be finite values between zero and one.")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise PredictionError("Probabilities must sum to one.")

    def to_record(self, *, document_id: str) -> dict[str, Any]:
        """Flatten the result for a tabular feature store."""

        record: dict[str, Any] = {
            "document_id": document_id,
            "label": self.label,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "aggregation": self.aggregation,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "device": self.device,
        }
        record.update({f"probability_{key}": value for key, value in self.probabilities.items()})
        return record
