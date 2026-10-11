"""Domain policy-stance classification using the published FOMC-RoBERTa model."""

from __future__ import annotations

from fed_policy_intelligence.nlp.prediction import ClassificationPrediction, PredictionError
from fed_policy_intelligence.nlp.transformer import TransformerTextClassifier

POLICY_MODEL_ID = "gtfintechlab/FOMC-RoBERTa"
POLICY_MODEL_REVISION = "aa3bc4281fb1fe73c8872e09ad5c64b898f90d83"
POLICY_LABELS = ("dovish", "hawkish", "neutral")
POLICY_LABEL_ALIASES = {
    "label_0": "dovish",
    "label_1": "hawkish",
    "label_2": "neutral",
    "dovish": "dovish",
    "hawkish": "hawkish",
    "neutral": "neutral",
}


def normalize_policy_prediction(
    prediction: ClassificationPrediction,
) -> ClassificationPrediction:
    """Map checkpoint label IDs to stable domain labels and preserve provenance."""

    unknown = sorted(set(prediction.probabilities) - set(POLICY_LABEL_ALIASES))
    if unknown:
        raise PredictionError(f"FOMC policy model returned unsupported labels: {unknown}")
    probabilities = {label: 0.0 for label in POLICY_LABELS}
    for source, probability in prediction.probabilities.items():
        probabilities[POLICY_LABEL_ALIASES[source]] += probability
    label = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    return ClassificationPrediction(
        label=label,
        probabilities=probabilities,
        chunk_count=prediction.chunk_count,
        token_count=prediction.token_count,
        aggregation=prediction.aggregation,
        model_id=prediction.model_id,
        tokenizer_id=prediction.tokenizer_id,
        device=prediction.device,
    )


class DomainPolicyAnalyzer:
    """Expose FOMC-RoBERTa with stable dovish/hawkish/neutral labels."""

    def __init__(self, classifier: TransformerTextClassifier) -> None:
        self.classifier = classifier

    def predict(self, text: str) -> ClassificationPrediction:
        return normalize_policy_prediction(self.classifier.predict(text))

    def predict_many(self, texts: list[str]) -> list[ClassificationPrediction]:
        return [
            normalize_policy_prediction(prediction)
            for prediction in self.classifier.predict_many(texts)
        ]
