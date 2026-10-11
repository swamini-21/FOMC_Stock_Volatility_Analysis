import json

import pandas as pd
import pytest

from fed_policy_intelligence.nlp.document_policy import (
    aggregate_sentence_predictions,
    analyze_document_policy,
    split_policy_sentences,
)
from fed_policy_intelligence.nlp.prediction import ClassificationPrediction


def _prediction(label: str) -> ClassificationPrediction:
    probabilities = {"dovish": 0.1, "hawkish": 0.1, "neutral": 0.1}
    probabilities[label] = 0.8
    return ClassificationPrediction(
        label=label,
        probabilities=probabilities,
        chunk_count=1,
        token_count=4,
        aggregation="document_length_weighted",
        model_id="local-policy-model",
        tokenizer_id="local-policy-model",
        device="cpu",
    )


class StubPolicyAnalyzer:
    def predict_many(self, texts: list[str]) -> list[ClassificationPrediction]:
        return [
            _prediction("hawkish" if "inflation" in text.casefold() else "neutral")
            for text in texts
        ]


def test_sentence_splitter_keeps_policy_sentences() -> None:
    result = split_policy_sentences(
        "Inflation remains elevated. Policy may remain restrictive.\nGrowth is steady."
    )

    assert result == [
        "Inflation remains elevated.",
        "Policy may remain restrictive.",
        "Growth is steady.",
    ]


def test_sentence_predictions_are_aggregated_transparently() -> None:
    result = aggregate_sentence_predictions(
        [_prediction("hawkish"), _prediction("hawkish"), _prediction("neutral")]
    )

    assert result["domain_policy_share_hawkish"] == pytest.approx(2 / 3)
    assert result["domain_policy_share_neutral"] == pytest.approx(1 / 3)
    assert result["domain_policy_net_stance"] == pytest.approx(2 / 3)
    assert json.loads(str(result["domain_policy_probabilities"]))["hawkish"] > 0.5


def test_document_policy_preserves_document_identity() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["doc-1"],
            "cleaned_text": ["Inflation is elevated. Conditions are otherwise stable."],
            "processing_version": ["1.0"],
        }
    )

    result = analyze_document_policy(documents, analyzer=StubPolicyAnalyzer())

    assert result.loc[0, "document_id"] == "doc-1"
    assert result.loc[0, "domain_policy_sentence_count"] == 2
    assert result.loc[0, "domain_policy_status"] == "success"
