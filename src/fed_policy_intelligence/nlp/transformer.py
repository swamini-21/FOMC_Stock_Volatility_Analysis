"""Testable transformer classification with explicit tokenizer/model pairing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from fed_policy_intelligence.nlp.chunking import (
    TokenChunk,
    TokenizerLike,
    chunk_text,
    split_paragraphs,
)
from fed_policy_intelligence.nlp.prediction import ClassificationPrediction, PredictionError


class ModelCompatibilityError(ValueError):
    """Raised when a model and tokenizer were loaded from different checkpoints."""


class LogitBackend(Protocol):
    """Backend contract that keeps core tests independent of PyTorch."""

    model_id: str
    tokenizer_id: str
    labels: tuple[str, ...]
    device: str

    def predict_logits(
        self, input_ids: Sequence[Sequence[int]], *, batch_size: int
    ) -> Sequence[Sequence[float]]: ...


def _normalized_identifier(identifier: str) -> str:
    return identifier.strip().replace("\\", "/").rstrip("/").casefold()


def validate_model_pair(*, tokenizer_id: str, model_id: str) -> None:
    """Reject the cross-checkpoint tokenizer bug found in the original notebook."""

    if not tokenizer_id.strip() or not model_id.strip():
        raise ModelCompatibilityError("Model and tokenizer identifiers must not be empty.")
    if _normalized_identifier(tokenizer_id) != _normalized_identifier(model_id):
        raise ModelCompatibilityError(
            f"Tokenizer '{tokenizer_id}' does not match model '{model_id}'."
        )


def softmax(logits: Sequence[float]) -> np.ndarray:
    """Return a numerically stable probability vector."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise PredictionError("Logits must be a finite one-dimensional vector.")
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def aggregate_probabilities(
    probabilities: Sequence[Sequence[float]],
    *,
    weights: Sequence[int] | None = None,
) -> np.ndarray:
    """Average probabilities, optionally weighted by non-special-token counts."""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise PredictionError("At least one probability vector is required.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise PredictionError("Probability vectors must be finite and non-negative.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise PredictionError("Every probability vector must sum to one.")
    if weights is None:
        result = matrix.mean(axis=0)
    else:
        numeric_weights = np.asarray(weights, dtype=np.float64)
        if numeric_weights.shape != (matrix.shape[0],) or np.any(numeric_weights <= 0):
            raise PredictionError("Aggregation weights must be positive and match the rows.")
        result = np.average(matrix, axis=0, weights=numeric_weights)
    return result / result.sum()


class TransformerTextClassifier:
    """Run long-document classification through an injected inference backend."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        backend: LogitBackend,
        max_length: int,
        stride: int,
        batch_size: int,
    ) -> None:
        validate_model_pair(tokenizer_id=backend.tokenizer_id, model_id=backend.model_id)
        tokenizer_name = str(getattr(tokenizer, "name_or_path", ""))
        validate_model_pair(tokenizer_id=tokenizer_name, model_id=backend.model_id)
        if not backend.labels or len(set(backend.labels)) != len(backend.labels):
            raise ModelCompatibilityError("Model labels must be present and unique.")
        self.tokenizer = tokenizer
        self.backend = backend
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size

    def _chunk_groups(self, text: str, aggregation: str) -> list[list[TokenChunk]]:
        if aggregation not in {"document_mean", "document_length_weighted", "paragraph_mean"}:
            raise PredictionError(f"Unsupported aggregation strategy: {aggregation}")
        units = split_paragraphs(text) if aggregation == "paragraph_mean" else [text]
        return [
            chunk_text(
                    self.tokenizer,
                    unit,
                    max_length=self.max_length,
                    stride=self.stride,
                )
            for unit in units
        ]

    def predict(
        self, text: str, *, aggregation: str = "document_length_weighted"
    ) -> ClassificationPrediction:
        """Return a normalized full distribution for one document."""

        chunk_groups = self._chunk_groups(text, aggregation)
        chunks = [chunk for group in chunk_groups for chunk in group]
        logits = self.backend.predict_logits(
            [chunk.input_ids for chunk in chunks], batch_size=self.batch_size
        )
        if len(logits) != len(chunks):
            raise PredictionError("The backend returned a different number of rows than chunks.")
        probabilities = [softmax(row) for row in logits]
        if any(len(row) != len(self.backend.labels) for row in probabilities):
            raise PredictionError("The backend output dimension does not match its labels.")
        if aggregation == "paragraph_mean":
            paragraph_probabilities: list[np.ndarray] = []
            offset = 0
            for group in chunk_groups:
                group_probabilities = probabilities[offset : offset + len(group)]
                paragraph_probabilities.append(
                    aggregate_probabilities(
                        group_probabilities,
                        weights=[chunk.content_token_count for chunk in group],
                    )
                )
                offset += len(group)
            combined = aggregate_probabilities(paragraph_probabilities)
        else:
            weights = (
                [chunk.content_token_count for chunk in chunks]
                if aggregation == "document_length_weighted"
                else None
            )
            combined = aggregate_probabilities(probabilities, weights=weights)
        distribution = {
            label.casefold(): float(combined[index])
            for index, label in enumerate(self.backend.labels)
        }
        winning_label = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        return ClassificationPrediction(
            label=winning_label,
            probabilities=distribution,
            chunk_count=len(chunks),
            token_count=sum(chunk.content_token_count for chunk in chunks),
            aggregation=aggregation,
            model_id=self.backend.model_id,
            tokenizer_id=self.backend.tokenizer_id,
            device=self.backend.device,
        )


class HuggingFaceLogitBackend:
    """Lazy PyTorch backend; importing core NLP modules does not require transformers."""

    def __init__(self, *, model: Any, tokenizer: Any, model_id: str, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.tokenizer_id = model_id
        self.device = device
        id2label = model.config.id2label
        self.labels = tuple(str(id2label[index]).casefold() for index in range(len(id2label)))

    def predict_logits(
        self, input_ids: Sequence[Sequence[int]], *, batch_size: int
    ) -> list[list[float]]:
        """Run batched inference in evaluation and inference modes."""

        import torch

        rows: list[list[float]] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(input_ids), batch_size):
                features = [
                    {"input_ids": list(row)}
                    for row in input_ids[start : start + batch_size]
                ]
                batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch).logits.detach().cpu().tolist()
                rows.extend(output)
        return rows


def load_huggingface_classifier(
    model_id: str,
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: str = "cpu",
    revision: str | None = None,
) -> TransformerTextClassifier:
    """Load a tokenizer and model from the exact same Hugging Face checkpoint."""

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "NLP dependencies are missing. Install the project with: pip install -e '.[nlp]'"
        ) from exc

    selected_device = device
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    model.to(selected_device)
    backend = HuggingFaceLogitBackend(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=selected_device,
    )
    return TransformerTextClassifier(
        tokenizer=tokenizer,
        backend=backend,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
    )
