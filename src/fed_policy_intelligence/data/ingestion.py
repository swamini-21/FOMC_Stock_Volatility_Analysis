"""CSV ingestion that preserves source rows and adds auditable derived fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fed_policy_intelligence.data.identity import (
    canonicalize_url,
    document_id,
    sha256_file,
    sha256_text,
)
from fed_policy_intelligence.data.preprocessing import clean_document_text

FED_REQUIRED_COLUMNS = {"url", "doc_type", "date", "text", "year", "meeting"}
VIX_REQUIRED_COLUMNS = {"observation_date", "VIXCLS"}
KNOWN_DOCUMENT_TYPES = {
    "implementation_note",
    "minutes",
    "other",
    "speech",
    "statement",
    "transcript",
}


class DataValidationError(ValueError):
    """Raised when an input cannot satisfy the minimum pipeline schema."""


def _require_columns(frame: pd.DataFrame, required: set[str], dataset: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{dataset} is missing required columns: {', '.join(missing)}")


def read_fed_corpus(path: str | Path) -> pd.DataFrame:
    """Read the source corpus without deduplicating or cleaning in place."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Federal Reserve corpus not found: {source}")
    frame = pd.read_csv(source, encoding="utf-8-sig")
    _require_columns(frame, FED_REQUIRED_COLUMNS, "Federal Reserve corpus")
    return frame


def prepare_fed_documents(
    source: pd.DataFrame,
    *,
    source_path: str | Path,
    source_name: str,
    processing_version: str,
) -> pd.DataFrame:
    """Create validated document records while retaining every source row."""

    _require_columns(source, FED_REQUIRED_COLUMNS, "Federal Reserve corpus")
    raw = source.reset_index(drop=True).copy(deep=False)
    result = pd.DataFrame(index=raw.index)
    result["source_row"] = range(len(raw))

    source_urls = raw["url"].astype("string")
    raw_text = raw["text"].fillna("").astype(str)
    document_types = raw["doc_type"].astype("string").str.strip().str.lower()
    parsed_dates = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    canonical_urls = source_urls.map(canonicalize_url)
    content_hashes = raw_text.map(sha256_text)

    result["document_id"] = [
        document_id(url, content_hash)
        for url, content_hash in zip(canonical_urls, content_hashes, strict=True)
    ]
    result["canonical_url"] = canonical_urls
    result["source_url"] = source_urls
    result["document_type"] = document_types
    result["publication_date"] = parsed_dates.dt.strftime("%Y-%m-%d").astype("string")
    result["publication_timestamp"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    result["timing_precision"] = "date"
    result["source_year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    result["meeting"] = raw["meeting"].astype("string")
    result["raw_text"] = raw_text
    result["content_sha256"] = content_hashes

    cleaning_results = raw_text.map(clean_document_text)
    result["cleaned_text"] = cleaning_results.map(lambda item: item.cleaned_text)
    result["raw_text_length"] = cleaning_results.map(lambda item: item.original_length)
    result["cleaned_text_length"] = cleaning_results.map(lambda item: item.cleaned_length)
    result["characters_removed"] = cleaning_results.map(lambda item: item.characters_removed)
    result["removal_ratio"] = cleaning_results.map(lambda item: item.removal_ratio)
    result["cleaning_rules"] = cleaning_results.map(lambda item: ",".join(item.matched_rules))

    result["is_exact_duplicate"] = raw.duplicated(keep=False)
    result["is_duplicate_content"] = content_hashes.duplicated(keep=False)
    result["is_duplicate_document_id"] = result["document_id"].duplicated(keep=False)

    errors: list[str] = []
    valid: list[bool] = []
    for url, text, doc_type, publication_date in zip(
        canonical_urls,
        raw_text,
        document_types,
        parsed_dates,
        strict=True,
    ):
        row_errors: list[str] = []
        if not url:
            row_errors.append("invalid_or_missing_url")
        if not text.strip():
            row_errors.append("empty_text")
        if doc_type not in KNOWN_DOCUMENT_TYPES:
            row_errors.append("unknown_document_type")
        if pd.isna(publication_date):
            row_errors.append("invalid_publication_date")
        errors.append(",".join(row_errors))
        valid.append(not row_errors)

    result["record_valid"] = valid
    result["validation_errors"] = errors
    result["source_name"] = source_name
    result["source_retrieved_at"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    result["source_dataset_sha256"] = sha256_file(source_path)
    result["processing_version"] = processing_version
    return result.reset_index(drop=True)


def read_vix_data(path: str | Path) -> pd.DataFrame:
    """Read local VIX observations without dropping missing values."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"VIX data not found: {source}")
    frame = pd.read_csv(source)
    _require_columns(frame, VIX_REQUIRED_COLUMNS, "VIX data")
    return frame


def prepare_vix_observations(
    source: pd.DataFrame,
    *,
    source_path: str | Path,
    source_name: str,
    series_id: str,
    processing_version: str,
) -> pd.DataFrame:
    """Normalize VIX rows and retain explicit reasons for unusable observations."""

    _require_columns(source, VIX_REQUIRED_COLUMNS, "VIX data")
    raw = source.reset_index(drop=True).copy(deep=False)
    parsed_dates = pd.to_datetime(raw["observation_date"], format="mixed", errors="coerce")
    values = pd.to_numeric(raw["VIXCLS"], errors="coerce")

    reasons: list[str] = []
    for date, value in zip(parsed_dates, values, strict=True):
        row_reasons: list[str] = []
        if pd.isna(date):
            row_reasons.append("invalid_trading_date")
        if pd.isna(value):
            row_reasons.append("missing_or_non_numeric_value")
        elif value < 0:
            row_reasons.append("negative_value")
        reasons.append(",".join(row_reasons))

    result = pd.DataFrame(
        {
            "source_row": range(len(raw)),
            "series_id": series_id,
            "trading_date": parsed_dates.dt.strftime("%Y-%m-%d").astype("string"),
            "value": values,
            "is_usable": [not reason for reason in reasons],
            "exclusion_reason": reasons,
            "source_name": source_name,
            "source_retrieved_at": pd.Series(pd.NA, index=raw.index, dtype="string"),
            "source_dataset_sha256": sha256_file(source_path),
            "processing_version": processing_version,
        }
    )
    return result.reset_index(drop=True)

