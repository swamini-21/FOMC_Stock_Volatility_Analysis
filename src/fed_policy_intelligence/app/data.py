"""Validated data loading and presentation helpers for the research dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FILES = {
    "documents": "fed_documents.csv",
    "nlp": "nlp_predictions.csv",
    "domain_policy": "domain_policy_predictions.csv",
    "events": "fed_vix_events.csv",
    "coefficients": "research_coefficients.csv",
    "correlations": "research_correlations.csv",
    "research": "research_analysis.json",
}

OUTCOME_LABELS = {
    "absolute_1d": "VIX point change — 1 day",
    "absolute_5d": "VIX point change — 5 days",
    "percentage_1d": "VIX percentage change — 1 day",
    "percentage_5d": "VIX percentage change — 5 days",
    "sp500_1d": "S&P 500 return — 1 day",
    "sp500_5d": "S&P 500 return — 5 days",
    "dgs2_1d": "2-year yield change — 1 day",
    "dgs2_5d": "2-year yield change — 5 days",
    "dgs10_1d": "10-year yield change — 1 day",
    "dgs10_5d": "10-year yield change — 5 days",
}

EVENT_OUTCOMES = {
    "VIX point change — 1 day": ("vix_change_1d", "VIX points"),
    "VIX point change — 5 days": ("vix_change_5d", "VIX points"),
    "S&P 500 return — 1 day": ("sp500_return_1d", "return"),
    "S&P 500 return — 5 days": ("sp500_return_5d", "return"),
    "2-year yield change — 1 day": ("dgs2_basis_points_1d", "basis points"),
    "2-year yield change — 5 days": ("dgs2_basis_points_5d", "basis points"),
    "10-year yield change — 1 day": ("dgs10_basis_points_1d", "basis points"),
    "10-year yield change — 5 days": ("dgs10_basis_points_5d", "basis points"),
}


@dataclass(frozen=True)
class DashboardData:
    documents: pd.DataFrame
    events: pd.DataFrame
    coefficients: pd.DataFrame
    correlations: pd.DataFrame
    research: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Cached {name} data is missing columns: {missing}")


def load_dashboard_data(processed_dir: str | Path) -> DashboardData:
    """Load cached outputs without invoking models, downloads, or external APIs."""

    directory = Path(processed_dir)
    paths = {name: directory / filename for name, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Dashboard inputs are missing: " + ", ".join(missing))

    documents = pd.read_csv(paths["documents"])
    nlp = pd.read_csv(paths["nlp"])
    domain = pd.read_csv(paths["domain_policy"])
    events = pd.read_csv(paths["events"])
    coefficients = pd.read_csv(paths["coefficients"])
    correlations = pd.read_csv(paths["correlations"])
    research = json.loads(paths["research"].read_text(encoding="utf-8"))

    _require_columns(
        documents,
        {"document_id", "publication_date", "document_type", "raw_text", "cleaned_text"},
        "document",
    )
    _require_columns(
        nlp,
        {"document_id", "nlp_status", "sentiment_label", "sentiment_probabilities"},
        "NLP",
    )
    _require_columns(
        domain,
        {
            "document_id",
            "domain_policy_status",
            "domain_policy_stance",
            "domain_policy_share_hawkish",
            "domain_policy_share_dovish",
            "domain_policy_share_neutral",
        },
        "domain policy",
    )
    _require_columns(events, {"event_date", "event_era"}, "event")
    _require_columns(
        coefficients,
        {"outcome_key", "model", "term", "coefficient", "ci_lower", "ci_upper"},
        "coefficient",
    )

    prediction_fields = [
        "document_id",
        "nlp_status",
        "sentiment_label",
        "sentiment_score",
        "sentiment_probabilities",
        "emotion_label",
        "emotion_probabilities",
    ]
    domain_fields = [
        "document_id",
        "domain_policy_status",
        "domain_policy_stance",
        "domain_policy_share_hawkish",
        "domain_policy_share_dovish",
        "domain_policy_share_neutral",
        "domain_policy_mean_confidence",
    ]
    documents = documents.merge(
        nlp[prediction_fields], on="document_id", how="left", validate="one_to_one"
    ).merge(domain[domain_fields], on="document_id", how="left", validate="one_to_one")
    documents["publication_date"] = pd.to_datetime(
        documents["publication_date"], errors="coerce"
    )
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    documents = documents.sort_values("publication_date", ascending=False).reset_index(drop=True)
    events = events.sort_values("event_date").reset_index(drop=True)
    return DashboardData(documents, events, coefficients, correlations, research)


def filter_events(
    events: pd.DataFrame,
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    eras: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Apply dashboard date and era filters with inclusive endpoints."""

    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    return events.loc[
        events["event_date"].between(start_timestamp, end_timestamp)
        & events["event_era"].isin(eras)
    ].copy()


def parse_probabilities(value: object) -> dict[str, float]:
    """Parse one cached probability object for display."""

    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Cached probabilities must be a JSON object.")
    return {str(label): float(probability) for label, probability in parsed.items()}
