import json

import pandas as pd
import pytest

from fed_policy_intelligence.analysis.events import build_event_dataset


def _probabilities(values: dict[str, float]) -> str:
    return json.dumps(values)


def test_build_event_dataset_aggregates_documents_and_uses_next_session() -> None:
    documents = pd.DataFrame(
        [
            {
                "document_id": "a",
                "publication_date": "2024-01-05",
                "document_type": "statement",
                "timing_precision": "date",
            },
            {
                "document_id": "b",
                "publication_date": "2024-01-05",
                "document_type": "statement",
                "timing_precision": "date",
            },
        ]
    )
    sentiment = _probabilities({"negative": 0.2, "neutral": 0.3, "positive": 0.5})
    emotion = _probabilities(
        {
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "neutral": 1.0,
            "sadness": 0.0,
            "surprise": 0.0,
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "document_id": "a",
                "nlp_status": "success",
                "sentiment_score": 0.3,
                "sentiment_probabilities": sentiment,
                "emotion_probabilities": emotion,
                "policy_stance": "hawkish",
            },
            {
                "document_id": "b",
                "nlp_status": "success",
                "sentiment_score": -0.1,
                "sentiment_probabilities": sentiment,
                "emotion_probabilities": emotion,
                "policy_stance": "dovish",
            },
        ]
    )
    domain_policy = pd.DataFrame(
        [
            {
                "document_id": "a",
                "domain_policy_status": "success",
                "domain_policy_share_hawkish": 0.75,
                "domain_policy_share_dovish": 0.10,
                "domain_policy_share_neutral": 0.15,
                "domain_policy_net_stance": 0.65,
                "domain_policy_mean_confidence": 0.80,
            },
            {
                "document_id": "b",
                "domain_policy_status": "success",
                "domain_policy_share_hawkish": 0.25,
                "domain_policy_share_dovish": 0.30,
                "domain_policy_share_neutral": 0.45,
                "domain_policy_net_stance": -0.05,
                "domain_policy_mean_confidence": 0.70,
            },
        ]
    )
    vix = pd.DataFrame(
        [
            {"trading_date": "2024-01-04", "value": 12.0, "is_usable": True},
            {"trading_date": "2024-01-05", "value": 13.0, "is_usable": True},
            {"trading_date": "2024-01-08", "value": 15.0, "is_usable": True},
            {"trading_date": "2024-01-09", "value": 14.0, "is_usable": True},
            {"trading_date": "2024-01-10", "value": 16.0, "is_usable": True},
            {"trading_date": "2024-01-11", "value": 17.0, "is_usable": True},
            {"trading_date": "2024-01-12", "value": 18.0, "is_usable": True},
        ]
    )

    events, report = build_event_dataset(
        documents, predictions, vix, domain_policy=domain_policy
    )

    event = events.iloc[0]
    assert event["document_count"] == 2
    assert event["pre_event_trading_date"] == pd.Timestamp("2024-01-04")
    assert event["post_1d_trading_date"] == pd.Timestamp("2024-01-08")
    assert event["vix_change_1d"] == pytest.approx(3.0)
    assert event["post_5d_trading_date"] == pd.Timestamp("2024-01-12")
    assert event["vix_change_5d"] == pytest.approx(6.0)
    assert event["sentiment_score"] == pytest.approx(0.1)
    assert event["policy_share_hawkish"] == pytest.approx(0.5)
    assert event["policy_share_dovish"] == pytest.approx(0.5)
    assert event["domain_policy_share_hawkish"] == pytest.approx(0.5)
    assert event["domain_policy_share_dovish"] == pytest.approx(0.2)
    assert event["domain_policy_net_stance"] == pytest.approx(0.3)
    assert report["domain_policy_included"] is True
    assert report["multi_document_events"] == 1


def test_build_event_dataset_excludes_transcripts_by_default() -> None:
    documents = pd.DataFrame(
        [
            {
                "document_id": "statement",
                "publication_date": "2024-01-05",
                "document_type": "statement",
                "timing_precision": "date",
            },
            {
                "document_id": "transcript",
                "publication_date": "2024-01-05",
                "document_type": "transcript",
                "timing_precision": "date",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "document_id": document_id,
                "nlp_status": "success",
                "sentiment_score": 0.0,
                "sentiment_probabilities": _probabilities(
                    {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
                ),
                "emotion_probabilities": _probabilities(
                    {
                        "anger": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "joy": 0.0,
                        "neutral": 1.0,
                        "sadness": 0.0,
                        "surprise": 0.0,
                    }
                ),
                "policy_stance": "neutral_or_mixed",
            }
            for document_id in ("statement", "transcript")
        ]
    )
    vix = pd.DataFrame(
        [
            {"trading_date": "2024-01-04", "value": 12.0, "is_usable": True},
            {"trading_date": "2024-01-08", "value": 13.0, "is_usable": True},
            {"trading_date": "2024-01-09", "value": 14.0, "is_usable": True},
            {"trading_date": "2024-01-10", "value": 15.0, "is_usable": True},
            {"trading_date": "2024-01-11", "value": 16.0, "is_usable": True},
            {"trading_date": "2024-01-12", "value": 17.0, "is_usable": True},
        ]
    )

    events, report = build_event_dataset(documents, predictions, vix)

    assert events.loc[0, "document_count"] == 1
    assert json.loads(events.loc[0, "document_ids"]) == ["statement"]
    assert report["eligible_documents"] == 1
    assert report["excluded_documents"] == 1


def test_build_event_dataset_retains_events_without_market_coverage() -> None:
    documents = pd.DataFrame(
        [
            {
                "document_id": "early",
                "publication_date": "1989-01-01",
                "document_type": "statement",
                "timing_precision": "date",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "document_id": "early",
                "nlp_status": "success",
                "sentiment_score": 0.0,
                "sentiment_probabilities": _probabilities(
                    {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
                ),
                "emotion_probabilities": _probabilities(
                    {
                        "anger": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "joy": 0.0,
                        "neutral": 1.0,
                        "sadness": 0.0,
                        "surprise": 0.0,
                    }
                ),
                "policy_stance": "insufficient_evidence",
            }
        ]
    )
    vix = pd.DataFrame(
        [
            {"trading_date": "1990-01-02", "value": 17.0, "is_usable": True},
            {"trading_date": "1990-01-03", "value": 18.0, "is_usable": True},
        ]
    )

    events, _ = build_event_dataset(documents, predictions, vix, horizons=(1,))

    assert len(events) == 1
    assert not bool(events.loc[0, "is_usable_1d"])
    assert events.loc[0, "exclusion_reason"] == "no_prior_usable_vix"
