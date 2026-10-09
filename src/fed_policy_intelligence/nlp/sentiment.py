"""Financial-sentiment classification built on the shared transformer runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fed_policy_intelligence.nlp.prediction import ClassificationPrediction
from fed_policy_intelligence.nlp.transformer import TransformerTextClassifier

FINBERT_LABELS = frozenset({"positive", "negative", "neutral"})


@dataclass(frozen=True)
class FinancialSentimentPrediction:
    """FinBERT distribution plus the documented positive-minus-negative score."""

    classification: ClassificationPrediction
    sentiment_score: float

    def to_record(self, *, document_id: str) -> dict[str, Any]:
        record = self.classification.to_record(document_id=document_id)
        record["sentiment_score"] = self.sentiment_score
        return record


class FinancialSentimentAnalyzer:
    """Validate and expose FinBERT as financial sentiment, not policy stance."""

    def __init__(self, classifier: TransformerTextClassifier) -> None:
        labels = frozenset(label.casefold() for label in classifier.backend.labels)
        if labels != FINBERT_LABELS:
            raise ValueError(
                f"FinBERT must expose labels {sorted(FINBERT_LABELS)}; "
                f"got {sorted(labels)}"
            )
        self.classifier = classifier

    def predict(
        self, text: str, *, aggregation: str = "document_length_weighted"
    ) -> FinancialSentimentPrediction:
        result = self.classifier.predict(text, aggregation=aggregation)
        score = result.probabilities["positive"] - result.probabilities["negative"]
        return FinancialSentimentPrediction(classification=result, sentiment_score=score)
