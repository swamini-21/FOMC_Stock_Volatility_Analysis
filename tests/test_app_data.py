import json
from unittest.mock import patch

import pandas as pd

from fed_policy_intelligence.app.data import filter_events, load_dashboard_data


def test_dashboard_loads_cached_outputs_without_api_access() -> None:
    frames = {
        "fed_documents.csv": pd.DataFrame(
            [
                {
                    "document_id": "doc-1",
                    "publication_date": "2024-01-31",
                    "document_type": "statement",
                    "raw_text": "Original",
                    "cleaned_text": "Cleaned",
                }
            ]
        ),
        "nlp_predictions.csv": pd.DataFrame(
            [
                {
                    "document_id": "doc-1",
                    "nlp_status": "success",
                    "sentiment_label": "neutral",
                    "sentiment_score": 0.0,
                    "sentiment_probabilities": '{"neutral": 1.0}',
                    "emotion_label": "neutral",
                    "emotion_probabilities": '{"neutral": 1.0}',
                }
            ]
        ),
        "domain_policy_predictions.csv": pd.DataFrame(
            [
                {
                    "document_id": "doc-1",
                    "domain_policy_status": "success",
                    "domain_policy_stance": "neutral",
                    "domain_policy_share_hawkish": 0.1,
                    "domain_policy_share_dovish": 0.2,
                    "domain_policy_share_neutral": 0.7,
                    "domain_policy_mean_confidence": 0.8,
                }
            ]
        ),
        "fed_vix_events.csv": pd.DataFrame(
            [
                {
                    "event_date": "2024-01-31",
                    "event_era": "2015-2025",
                    "vix_change_1d": 1.2,
                }
            ]
        ),
        "research_coefficients.csv": pd.DataFrame(
            [
                {
                    "outcome_key": "absolute_1d",
                    "model": "stance",
                    "term": "domain_policy_share_hawkish",
                    "coefficient": 1.0,
                    "ci_lower": -1.0,
                    "ci_upper": 3.0,
                }
            ]
        ),
        "research_correlations.csv": pd.DataFrame(
            [{"outcome": "vix_change_1d", "feature": "sentiment_score"}]
        ),
    }
    with (
        patch("fed_policy_intelligence.app.data.Path.is_file", return_value=True),
        patch(
            "fed_policy_intelligence.app.data.pd.read_csv",
            side_effect=lambda path: frames[path.name].copy(),
        ),
        patch(
            "fed_policy_intelligence.app.data.Path.read_text",
            return_value=json.dumps({"design": {}, "outcomes": {}}),
        ),
    ):
        data = load_dashboard_data("ignored-cache")

    filtered = filter_events(
        data.events,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-31"),
        ["2015-2025"],
    )
    assert len(data.documents) == 1
    assert data.documents.loc[0, "domain_policy_stance"] == "neutral"
    assert len(filtered) == 1
