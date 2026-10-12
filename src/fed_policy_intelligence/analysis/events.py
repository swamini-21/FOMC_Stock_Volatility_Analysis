"""Construct auditable Fed-document events and forward VIX outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fed_policy_intelligence.settings import ProjectSettings, load_settings

SENTIMENT_LABELS = ("negative", "neutral", "positive")
EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
POLICY_LABELS = ("hawkish", "dovish", "neutral_or_mixed", "insufficient_evidence")
DOMAIN_POLICY_FEATURES = (
    "domain_policy_share_hawkish",
    "domain_policy_share_dovish",
    "domain_policy_share_neutral",
    "domain_policy_net_stance",
    "domain_policy_mean_confidence",
)
EXTERNAL_MARKET_SERIES = {
    "sp500": ("sp500_close", "return"),
    "dgs2": ("dgs2", "basis_points"),
    "dgs10": ("dgs10", "basis_points"),
}


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _parse_bool_series(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"true", "false"})
    if invalid:
        raise ValueError(f"{field} contains invalid Boolean values: {invalid}")
    return normalized.eq("true")


def _expand_probabilities(
    frame: pd.DataFrame, source_column: str, labels: tuple[str, ...], prefix: str
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for document_id, raw in frame[["document_id", source_column]].itertuples(index=False):
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid {source_column} JSON for document {document_id}."
            ) from exc
        missing = sorted(set(labels) - set(parsed))
        if missing:
            raise ValueError(
                f"{source_column} for document {document_id} is missing labels: {missing}"
            )
        records.append({f"{prefix}_{label}": float(parsed[label]) for label in labels})
    return pd.DataFrame.from_records(records, index=frame.index)


def _prepare_documents(documents: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        documents,
        {"document_id", "publication_date", "document_type", "timing_precision"},
        "Document data",
    )
    _require_columns(
        predictions,
        {
            "document_id",
            "nlp_status",
            "sentiment_score",
            "sentiment_probabilities",
            "emotion_probabilities",
            "policy_stance",
        },
        "Prediction data",
    )
    if documents["document_id"].duplicated().any():
        raise ValueError("Document data contains duplicate document_id values.")
    if predictions["document_id"].duplicated().any():
        raise ValueError("Prediction data contains duplicate document_id values.")

    prediction_fields = [
        "document_id",
        "nlp_status",
        "sentiment_score",
        "sentiment_probabilities",
        "emotion_probabilities",
        "policy_stance",
    ]
    merged = documents.merge(
        predictions[prediction_fields],
        on="document_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] != "both", "document_id"].tolist()
    if missing:
        raise ValueError(f"Documents are missing NLP predictions: {missing[:10]}")
    merged = merged.drop(columns="_merge")
    merged["publication_date"] = pd.to_datetime(
        merged["publication_date"], errors="coerce"
    ).dt.normalize()
    invalid_dates = merged.loc[merged["publication_date"].isna(), "document_id"].tolist()
    if invalid_dates:
        raise ValueError(f"Documents have invalid publication dates: {invalid_dates[:10]}")

    success = merged["nlp_status"].eq("success")
    successful = merged.loc[success].copy()
    sentiment = _expand_probabilities(
        successful,
        "sentiment_probabilities",
        SENTIMENT_LABELS,
        "sentiment_probability",
    )
    emotion = _expand_probabilities(
        successful,
        "emotion_probabilities",
        EMOTION_LABELS,
        "emotion_probability",
    )
    for column in sentiment.columns:
        merged[column] = sentiment[column]
    for column in emotion.columns:
        merged[column] = emotion[column]
    merged["sentiment_score"] = pd.to_numeric(merged["sentiment_score"], errors="coerce")
    for label in POLICY_LABELS:
        merged[f"policy_share_{label}"] = merged["policy_stance"].eq(label).astype(float)
    return merged


def _merge_domain_policy(
    documents: pd.DataFrame, domain_policy: pd.DataFrame
) -> pd.DataFrame:
    required = {"document_id", "domain_policy_status", *DOMAIN_POLICY_FEATURES}
    _require_columns(domain_policy, required, "Domain policy data")
    if domain_policy["document_id"].duplicated().any():
        raise ValueError("Domain policy data contains duplicate document_id values.")
    fields = ["document_id", "domain_policy_status", *DOMAIN_POLICY_FEATURES]
    merged = documents.merge(
        domain_policy[fields], on="document_id", how="left", validate="one_to_one"
    )
    for column in DOMAIN_POLICY_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def _aggregate_events(documents: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "sentiment_score",
        *(f"sentiment_probability_{label}" for label in SENTIMENT_LABELS),
        *(f"emotion_probability_{label}" for label in EMOTION_LABELS),
        *(f"policy_share_{label}" for label in POLICY_LABELS),
    ]
    feature_columns.extend(
        feature for feature in DOMAIN_POLICY_FEATURES if feature in documents.columns
    )
    records: list[dict[str, object]] = []
    for event_date, group in documents.groupby("publication_date", sort=True):
        successful = group.loc[group["nlp_status"].eq("success")]
        year = int(pd.Timestamp(event_date).year)
        if year <= 2004:
            era = "1990-2004"
        elif year <= 2014:
            era = "2005-2014"
        else:
            era = "2015-2025"
        record: dict[str, object] = {
            "event_date": event_date,
            "event_year": year,
            "event_month": int(pd.Timestamp(event_date).month),
            "event_era": era,
            "timing_rule": "date_only_next_usable_session",
            "document_count": len(group),
            "statement_count": int(group["document_type"].eq("statement").sum()),
            "transcript_count": int(group["document_type"].eq("transcript").sum()),
            "nlp_success_count": len(successful),
            "nlp_error_count": int(group["nlp_status"].ne("success").sum()),
            "document_ids": json.dumps(sorted(group["document_id"].astype(str).tolist())),
        }
        for column in feature_columns:
            record[column] = successful[column].mean() if not successful.empty else float("nan")
        records.append(record)
    return pd.DataFrame.from_records(records)


def _prepare_market(vix: pd.DataFrame) -> pd.DataFrame:
    _require_columns(vix, {"trading_date", "value", "is_usable"}, "VIX data")
    market = vix.copy()
    market["trading_date"] = pd.to_datetime(market["trading_date"], errors="coerce").dt.normalize()
    market["value"] = pd.to_numeric(market["value"], errors="coerce")
    market["is_usable"] = _parse_bool_series(market["is_usable"], "is_usable")
    usable = market.loc[
        market["is_usable"] & market["trading_date"].notna() & market["value"].notna(),
        ["trading_date", "value"],
    ].sort_values("trading_date")
    if usable.empty:
        raise ValueError("VIX data contains no usable observations.")
    if usable["trading_date"].duplicated().any():
        raise ValueError("Usable VIX data contains duplicate trading dates.")
    return usable.reset_index(drop=True)


def _align_market_outcomes(
    events: pd.DataFrame, market: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    dates = market["trading_date"]
    values = market["value"]
    records: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        event_date = pd.Timestamp(event["event_date"])
        prior_position = int(dates.searchsorted(event_date, side="left")) - 1
        first_after = int(dates.searchsorted(event_date, side="right"))
        reasons: list[str] = []
        if prior_position < 0:
            reasons.append("no_prior_usable_vix")
            pre_value = float("nan")
            event["pre_event_trading_date"] = pd.NaT
        else:
            pre_value = float(values.iloc[prior_position])
            event["pre_event_trading_date"] = dates.iloc[prior_position]
        event["pre_event_vix"] = pre_value
        for lookback in (1, 5):
            history_position = prior_position - lookback
            if prior_position < 0 or history_position < 0:
                event[f"pre_event_vix_change_{lookback}d"] = float("nan")
            else:
                event[f"pre_event_vix_change_{lookback}d"] = (
                    pre_value - float(values.iloc[history_position])
                )

        for horizon in horizons:
            outcome_position = first_after + horizon - 1
            usable = prior_position >= 0 and outcome_position < len(market)
            event[f"is_usable_{horizon}d"] = usable
            if not usable:
                if outcome_position >= len(market):
                    reasons.append(f"no_post_{horizon}d_usable_vix")
                event[f"post_{horizon}d_trading_date"] = pd.NaT
                event[f"post_{horizon}d_vix"] = float("nan")
                event[f"vix_change_{horizon}d"] = float("nan")
                event[f"vix_pct_change_{horizon}d"] = float("nan")
                continue
            post_value = float(values.iloc[outcome_position])
            event[f"post_{horizon}d_trading_date"] = dates.iloc[outcome_position]
            event[f"post_{horizon}d_vix"] = post_value
            event[f"vix_change_{horizon}d"] = post_value - pre_value
            event[f"vix_pct_change_{horizon}d"] = (post_value / pre_value) - 1.0
        event["alignment_status"] = "complete" if not reasons else "excluded"
        event["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
        records.append(event)
    return pd.DataFrame.from_records(records)


def _prepare_external_markets(market_controls: pd.DataFrame) -> pd.DataFrame:
    required = {"trading_date", *(spec[0] for spec in EXTERNAL_MARKET_SERIES.values())}
    _require_columns(market_controls, required, "External market data")
    prepared = market_controls.copy()
    prepared["trading_date"] = pd.to_datetime(
        prepared["trading_date"], errors="coerce"
    ).dt.normalize()
    for value_column, _ in EXTERNAL_MARKET_SERIES.values():
        prepared[value_column] = pd.to_numeric(prepared[value_column], errors="coerce")
    prepared = prepared.dropna(subset=["trading_date"]).sort_values("trading_date")
    if prepared["trading_date"].duplicated().any():
        raise ValueError("External market data contains duplicate dates.")
    return prepared.reset_index(drop=True)


def _change(current: float, reference: float, change_type: str) -> float:
    if change_type == "return":
        return (current / reference) - 1.0
    return (current - reference) * 100.0


def _align_external_markets(
    events: pd.DataFrame, market_controls: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    aligned = events.copy()
    for prefix, (value_column, change_type) in EXTERNAL_MARKET_SERIES.items():
        series = market_controls[["trading_date", value_column]].dropna().reset_index(drop=True)
        if series.empty:
            raise ValueError(f"External market series {value_column} contains no observations.")
        dates = series["trading_date"]
        values = series[value_column]
        for row_index, event_date in aligned["event_date"].items():
            event_date = pd.Timestamp(event_date)
            prior_position = int(dates.searchsorted(event_date, side="left")) - 1
            first_after = int(dates.searchsorted(event_date, side="right"))
            if prior_position < 0:
                continue
            pre_value = float(values.iloc[prior_position])
            aligned.loc[row_index, f"pre_event_{prefix}_date"] = dates.iloc[prior_position]
            aligned.loc[row_index, f"pre_event_{prefix}"] = pre_value
            for lookback in (1, 5):
                history_position = prior_position - lookback
                if history_position >= 0:
                    aligned.loc[row_index, f"pre_event_{prefix}_{change_type}_{lookback}d"] = (
                        _change(pre_value, float(values.iloc[history_position]), change_type)
                    )
            for horizon in horizons:
                outcome_position = first_after + horizon - 1
                usable = outcome_position < len(series)
                aligned.loc[row_index, f"is_usable_{prefix}_{horizon}d"] = usable
                if not usable:
                    continue
                post_value = float(values.iloc[outcome_position])
                aligned.loc[row_index, f"post_{prefix}_{horizon}d_date"] = dates.iloc[
                    outcome_position
                ]
                aligned.loc[row_index, f"post_{prefix}_{horizon}d"] = post_value
                aligned.loc[row_index, f"{prefix}_{change_type}_{horizon}d"] = _change(
                    post_value, pre_value, change_type
                )
        for horizon in horizons:
            usable_column = f"is_usable_{prefix}_{horizon}d"
            aligned[usable_column] = aligned[usable_column].eq(True)
    return aligned


def build_event_dataset(
    documents: pd.DataFrame,
    predictions: pd.DataFrame,
    vix: pd.DataFrame,
    *,
    domain_policy: pd.DataFrame | None = None,
    market_controls: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (1, 5),
    eligible_document_types: tuple[str, ...] = ("statement",),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate public-at-event documents and align forward VIX outcomes.

    All source publication times are date-only. To avoid using an unknown same-day
    close as post-announcement data, the first post-event observation is the first
    usable VIX session strictly after the publication date. Transcripts are excluded
    by default because their text was released after the meeting date represented in
    the corpus and would introduce look-ahead leakage.
    """

    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("Event horizons must be positive integers.")
    if not eligible_document_types:
        raise ValueError("At least one eligible document type is required.")
    selected_horizons = tuple(sorted(set(horizons)))
    prepared_documents = _prepare_documents(documents, predictions)
    if domain_policy is not None:
        prepared_documents = _merge_domain_policy(prepared_documents, domain_policy)
    eligible = prepared_documents.loc[
        prepared_documents["document_type"].isin(eligible_document_types)
    ].copy()
    if eligible.empty:
        raise ValueError("No documents match the eligible event document types.")
    if domain_policy is not None:
        missing_domain = eligible.loc[
            eligible["domain_policy_status"].ne("success"), "document_id"
        ].tolist()
        if missing_domain:
            raise ValueError(
                "Eligible documents are missing successful domain policy predictions: "
                f"{missing_domain[:10]}"
            )
    events = _aggregate_events(eligible)
    market = _prepare_market(vix)
    aligned = _align_market_outcomes(events, market, selected_horizons)
    prepared_external = None
    if market_controls is not None:
        prepared_external = _prepare_external_markets(market_controls)
        aligned = _align_external_markets(aligned, prepared_external, selected_horizons)
    report = {
        "input_documents": len(documents),
        "input_predictions": len(predictions),
        "eligible_document_types": list(eligible_document_types),
        "eligible_documents": len(eligible),
        "excluded_documents": len(prepared_documents) - len(eligible),
        "domain_policy_included": domain_policy is not None,
        "external_market_controls_included": market_controls is not None,
        "event_dates": len(aligned),
        "multi_document_events": int(aligned["document_count"].gt(1).sum()),
        "usable_vix_observations": len(market),
        "timing_precision": sorted(eligible["timing_precision"].astype(str).unique()),
        "timing_rule": "first usable VIX session strictly after publication date",
        "horizons": {
            f"{horizon}d": {
                "usable_events": int(aligned[f"is_usable_{horizon}d"].sum()),
                "excluded_events": int((~aligned[f"is_usable_{horizon}d"]).sum()),
            }
            for horizon in selected_horizons
        },
    }
    if prepared_external is not None:
        report["external_market_observations"] = {
            prefix: int(prepared_external[value_column].notna().sum())
            for prefix, (value_column, _) in EXTERNAL_MARKET_SERIES.items()
        }
        report["external_market_horizons"] = {
            f"{prefix}_{horizon}d": {
                "usable_events": int(aligned[f"is_usable_{prefix}_{horizon}d"].sum()),
                "excluded_events": int((~aligned[f"is_usable_{prefix}_{horizon}d"]).sum()),
            }
            for prefix in EXTERNAL_MARKET_SERIES
            for horizon in selected_horizons
        }
    return aligned, report


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def _write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def run_event_pipeline(settings: ProjectSettings) -> dict[str, Path]:
    """Load Phase 3/4 outputs and write the Phase 5 event dataset."""

    processed = settings.data.processed_dir
    document_path = processed / "fed_documents.csv"
    prediction_path = processed / "nlp_predictions.csv"
    domain_policy_path = processed / "domain_policy_predictions.csv"
    vix_path = processed / "vix_observations.csv"
    for required in (document_path, prediction_path, domain_policy_path, vix_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required processed input not found: {required}")

    market_controls_path = processed / "market_controls.csv"
    events, report = build_event_dataset(
        pd.read_csv(document_path),
        pd.read_csv(prediction_path),
        pd.read_csv(vix_path),
        domain_policy=pd.read_csv(domain_policy_path),
        market_controls=(
            pd.read_csv(market_controls_path) if market_controls_path.is_file() else None
        ),
    )
    event_path = processed / "fed_vix_events.csv"
    report_path = processed / "event_construction_report.json"
    _write_csv_atomic(events, event_path)
    _write_json_atomic(report, report_path)
    return {"events": event_path, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    """Build event-level NLP features and VIX outcomes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    paths = run_event_pipeline(load_settings(args.config))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
