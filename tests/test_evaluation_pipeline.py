import pandas as pd
import pytest

from fed_policy_intelligence.evaluation.pipeline import (
    evaluate_prediction_frame,
    load_reviewed_annotations,
)


def _annotation_row(document_id: str = "doc-1") -> dict[str, str]:
    return {
        "document_id": document_id,
        "split": "evaluation",
        "financial_sentiment": "positive",
        "emotion": "neutral",
        "policy_stance": "dovish",
        "policy_evidence": "supporting words",
        "annotator_id": "1",
        "annotated_at": "2026-10-10",
        "annotator_notes": "reviewed",
    }


def test_load_reviewed_annotations_selects_and_validates_split() -> None:
    selected = load_reviewed_annotations("tests/fixtures/evaluation_annotations.csv")

    assert selected["document_id"].tolist() == ["doc-1"]


def test_evaluate_prediction_frame_reports_mismatches() -> None:
    annotations = pd.DataFrame([_annotation_row()])
    predictions = pd.DataFrame(
        [
            {
                "document_id": "doc-1",
                "nlp_status": "success",
                "sentiment_label": "negative",
                "emotion_label": "neutral",
                "policy_stance": "dovish",
            }
        ]
    )

    report, errors = evaluate_prediction_frame(annotations, predictions)

    assert report["sample_size"] == 1
    assert report["tasks"]["emotion"]["per_class"]["neutral"]["f1"] == 1.0
    assert errors.to_dict("records") == [
        {
            "document_id": "doc-1",
            "task": "financial_sentiment",
            "expected": "positive",
            "predicted": "negative",
        }
    ]


def test_evaluate_prediction_frame_rejects_missing_predictions() -> None:
    annotations = pd.DataFrame([_annotation_row()])
    predictions = pd.DataFrame(
        columns=("document_id", "nlp_status", "sentiment_label", "emotion_label", "policy_stance")
    )

    with pytest.raises(ValueError, match="missing predictions"):
        evaluate_prediction_frame(annotations, predictions)
