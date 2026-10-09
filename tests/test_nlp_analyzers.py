import pytest
from test_nlp_transformer import FakeBackend, FakeTokenizer

from fed_policy_intelligence.nlp.emotion import EMOTION_LABELS, EmotionAnalyzer
from fed_policy_intelligence.nlp.sentiment import FinancialSentimentAnalyzer
from fed_policy_intelligence.nlp.transformer import TransformerTextClassifier


def test_financial_sentiment_score_is_positive_minus_negative() -> None:
    analyzer = FinancialSentimentAnalyzer(
        TransformerTextClassifier(
            tokenizer=FakeTokenizer("test/model"),
            backend=FakeBackend(),
            max_length=8,
            stride=1,
            batch_size=2,
        )
    )

    result = analyzer.predict("positive outlook")

    expected = (
        result.classification.probabilities["positive"]
        - result.classification.probabilities["negative"]
    )
    assert result.sentiment_score == pytest.approx(expected)


class EmotionBackend(FakeBackend):
    labels = tuple(sorted(EMOTION_LABELS))

    def predict_logits(self, input_ids, *, batch_size):
        del batch_size
        return [[float(index) for index in range(len(self.labels))] for _ in input_ids]


def test_emotion_analyzer_returns_all_supported_labels() -> None:
    analyzer = EmotionAnalyzer(
        TransformerTextClassifier(
            tokenizer=FakeTokenizer("test/model"),
            backend=EmotionBackend(),
            max_length=8,
            stride=1,
            batch_size=2,
        )
    )

    result = analyzer.predict("ordinary policy language")

    assert set(result.probabilities) == EMOTION_LABELS
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_emotion_analyzer_rejects_incomplete_label_space() -> None:
    classifier = TransformerTextClassifier(
        tokenizer=FakeTokenizer("test/model"),
        backend=FakeBackend(),
        max_length=8,
        stride=1,
        batch_size=2,
    )

    with pytest.raises(ValueError, match="Emotion model"):
        EmotionAnalyzer(classifier)
