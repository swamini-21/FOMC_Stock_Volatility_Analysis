from collections.abc import Sequence

import numpy as np
import pytest

from fed_policy_intelligence.nlp.chunking import chunk_text
from fed_policy_intelligence.nlp.prediction import PredictionError
from fed_policy_intelligence.nlp.transformer import (
    ModelCompatibilityError,
    TransformerTextClassifier,
    aggregate_probabilities,
    validate_model_pair,
)


class FakeTokenizer:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        vocabulary = {"positive": 1, "negative": 2}
        return [vocabulary.get(word, 3) for word in text.split()]

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int:
        del pair
        return 2

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [101, *token_ids, 102]


class FakeBackend:
    model_id = "test/model"
    tokenizer_id = "test/model"
    labels = ("positive", "negative", "neutral")
    device = "cpu"

    def predict_logits(
        self, input_ids: Sequence[Sequence[int]], *, batch_size: int
    ) -> list[list[float]]:
        assert batch_size > 0
        rows = []
        for row in input_ids:
            if 1 in row:
                rows.append([4.0, 0.0, 0.0])
            elif 2 in row:
                rows.append([0.0, 4.0, 0.0])
            else:
                rows.append([0.0, 0.0, 4.0])
        return rows


def _classifier() -> TransformerTextClassifier:
    return TransformerTextClassifier(
        tokenizer=FakeTokenizer("test/model"),
        backend=FakeBackend(),
        max_length=6,
        stride=1,
        batch_size=2,
    )


def test_mismatched_finbert_tokenizer_and_emotion_model_is_rejected() -> None:
    with pytest.raises(ModelCompatibilityError, match="does not match"):
        validate_model_pair(
            tokenizer_id="ProsusAI/finbert",
            model_id="j-hartmann/emotion-english-distilroberta-base",
        )


def test_long_document_chunking_reserves_special_tokens_and_overlaps() -> None:
    chunks = chunk_text(
        FakeTokenizer("test/model"),
        "one two three four five six seven eight nine",
        max_length=6,
        stride=1,
    )

    assert [chunk.content_token_count for chunk in chunks] == [4, 4, 3]
    assert all(len(chunk.input_ids) <= 6 for chunk in chunks)
    assert chunks[0].input_ids[-2] == chunks[1].input_ids[1]


def test_classifier_returns_finite_normalized_full_distribution() -> None:
    result = _classifier().predict(
        "positive words continue for enough tokens to require another chunk"
    )

    assert set(result.probabilities) == {"positive", "negative", "neutral"}
    assert np.isfinite(list(result.probabilities.values())).all()
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.chunk_count > 1


def test_probability_aggregation_does_not_average_logits() -> None:
    result = aggregate_probabilities([[0.8, 0.2], [0.2, 0.8]], weights=[3, 1])

    assert result.tolist() == pytest.approx([0.65, 0.35])


class WrongDimensionBackend(FakeBackend):
    labels = ("positive", "negative", "neutral")

    def predict_logits(
        self, input_ids: Sequence[Sequence[int]], *, batch_size: int
    ) -> list[list[float]]:
        del batch_size
        return [[1.0, 0.0] for _ in input_ids]


def test_invalid_output_dimension_is_rejected() -> None:
    classifier = TransformerTextClassifier(
        tokenizer=FakeTokenizer("test/model"),
        backend=WrongDimensionBackend(),
        max_length=6,
        stride=1,
        batch_size=2,
    )

    with pytest.raises(PredictionError, match="dimension"):
        classifier.predict("some text")


def test_paragraph_and_document_aggregation_are_both_available() -> None:
    classifier = _classifier()
    text = "positive filler filler filler filler filler filler\n\nnegative"

    paragraph = classifier.predict(text, aggregation="paragraph_mean")
    document = classifier.predict(text, aggregation="document_length_weighted")

    assert paragraph.aggregation == "paragraph_mean"
    assert document.aggregation == "document_length_weighted"
    assert paragraph.probabilities != document.probabilities
