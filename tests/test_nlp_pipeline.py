import json

import pandas as pd

from fed_policy_intelligence.nlp.pipeline import analyze_documents
from fed_policy_intelligence.nlp.prediction import ClassificationPrediction
from fed_policy_intelligence.nlp.sentiment import FinancialSentimentPrediction


def _prediction(labels: list[str]) -> ClassificationPrediction:
    probability = 1.0 / len(labels)
    return ClassificationPrediction(
        label=labels[0],
        probabilities={label: probability for label in labels},
        chunk_count=2,
        token_count=12,
        aggregation="document_length_weighted",
        model_id="test/model",
        tokenizer_id="test/model",
        device="cpu",
    )


class StubSentiment:
    def predict(self, text: str, *, aggregation: str):
        if text == "fail":
            raise RuntimeError("deliberate test failure")
        result = _prediction(["positive", "negative", "neutral"])
        return FinancialSentimentPrediction(result, sentiment_score=0.0)


class StubEmotion:
    def predict(self, text: str, *, aggregation: str):
        del text, aggregation
        return _prediction(
            ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        )


def test_document_pipeline_retains_successes_and_failures() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["doc-1", "doc-2"],
            "cleaned_text": ["The Committee raised the target range.", "fail"],
            "processing_version": ["test", "test"],
        }
    )

    output = analyze_documents(
        documents,
        sentiment=StubSentiment(),
        emotion=StubEmotion(),
        sentiment_aggregation="document_length_weighted",
        emotion_aggregation="document_length_weighted",
    )

    assert output["document_id"].tolist() == ["doc-1", "doc-2"]
    assert output["nlp_status"].tolist() == ["success", "error"]
    assert output.loc[0, "policy_stance"] == "hawkish"
    assert output.loc[0, "sentiment_device"] == "cpu"
    assert output.loc[0, "emotion_token_count"] == 12
    assert set(json.loads(output.loc[0, "emotion_probabilities"])) == {
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    }
    assert "deliberate test failure" in output.loc[1, "nlp_error"]
