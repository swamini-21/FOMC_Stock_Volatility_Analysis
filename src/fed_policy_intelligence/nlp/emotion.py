"""Emotion classification with strict label and tokenizer compatibility checks."""

from __future__ import annotations

from fed_policy_intelligence.nlp.prediction import ClassificationPrediction
from fed_policy_intelligence.nlp.transformer import TransformerTextClassifier

EMOTION_LABELS = frozenset(
    {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
)


class EmotionAnalyzer:
    """Return the complete seven-label distribution documented by the model card."""

    def __init__(self, classifier: TransformerTextClassifier) -> None:
        labels = frozenset(label.casefold() for label in classifier.backend.labels)
        if labels != EMOTION_LABELS:
            raise ValueError(
                f"Emotion model must expose labels {sorted(EMOTION_LABELS)}; got {sorted(labels)}"
            )
        self.classifier = classifier

    def predict(
        self, text: str, *, aggregation: str = "document_length_weighted"
    ) -> ClassificationPrediction:
        return self.classifier.predict(text, aggregation=aggregation)
