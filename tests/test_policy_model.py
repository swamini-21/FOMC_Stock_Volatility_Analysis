
import pytest

from fed_policy_intelligence.evaluation.expert_policy import evaluate_expert_policy
from fed_policy_intelligence.nlp.policy_model import (
    DomainPolicyAnalyzer,
    normalize_policy_prediction,
)
from fed_policy_intelligence.nlp.prediction import ClassificationPrediction, PredictionError


def _prediction(probabilities: dict[str, float]) -> ClassificationPrediction:
    return ClassificationPrediction(
        label=max(probabilities, key=probabilities.get),  # type: ignore[arg-type]
        probabilities=probabilities,
        chunk_count=1,
        token_count=5,
        aggregation="document_length_weighted",
        model_id="gtfintechlab/FOMC-RoBERTa",
        tokenizer_id="gtfintechlab/FOMC-RoBERTa",
        device="cpu",
    )


def test_checkpoint_labels_are_mapped_to_policy_stance() -> None:
    normalized = normalize_policy_prediction(
        _prediction({"label_0": 0.1, "label_1": 0.8, "label_2": 0.1})
    )

    assert normalized.label == "hawkish"
    assert normalized.probabilities == pytest.approx(
        {"dovish": 0.1, "hawkish": 0.8, "neutral": 0.1}
    )


def test_unknown_checkpoint_label_is_rejected() -> None:
    with pytest.raises(PredictionError, match="unsupported labels"):
        normalize_policy_prediction(_prediction({"unexpected": 1.0}))


class StubClassifier:
    def predict(self, text: str) -> ClassificationPrediction:
        if "tighten" in text:
            return _prediction({"label_0": 0.0, "label_1": 1.0, "label_2": 0.0})
        return _prediction({"label_0": 1.0, "label_1": 0.0, "label_2": 0.0})


def test_expert_policy_evaluation_returns_metrics_and_rows() -> None:
    import pandas as pd

    test_data = pd.DataFrame(
        {
            "sentence": ["Policy may tighten.", "Policy may ease."],
            "policy_stance": ["hawkish", "dovish"],
        }
    )

    report, predictions = evaluate_expert_policy(
        test_data, analyzer=DomainPolicyAnalyzer(StubClassifier())
    )

    assert report["sample_size"] == 2
    assert report["macro_f1"] == pytest.approx(2 / 3)
    assert predictions["predicted_policy_stance"].tolist() == ["hawkish", "dovish"]
