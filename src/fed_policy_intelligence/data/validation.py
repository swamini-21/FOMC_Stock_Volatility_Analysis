"""Dataset-level validation reports with explicit counts and examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding and the number of affected rows."""

    code: str
    severity: str
    count: int
    message: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Auditable record counts and validation issues for one dataset."""

    dataset: str
    input_rows: int
    output_rows: int
    usable_rows: int
    metrics: dict[str, Any]
    issues: tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _examples(series: pd.Series, mask: pd.Series, limit: int = 5) -> tuple[str, ...]:
    return tuple(series.loc[mask].dropna().astype(str).head(limit).tolist())


def validate_fed_documents(frame: pd.DataFrame, *, input_rows: int) -> ValidationReport:
    """Validate prepared document records without treating shared dates as duplicates."""

    issues: list[ValidationIssue] = []
    invalid_mask = ~frame["record_valid"]
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        issues.append(
            ValidationIssue(
                code="invalid_document_records",
                severity="error",
                count=invalid_count,
                message="Documents failed one or more required-field validations.",
                examples=_examples(frame["validation_errors"], invalid_mask),
            )
        )

    id_conflicts = frame.groupby("document_id")["content_sha256"].nunique().gt(1)
    conflict_ids = id_conflicts[id_conflicts].index
    if len(conflict_ids):
        issues.append(
            ValidationIssue(
                code="document_id_content_conflicts",
                severity="error",
                count=int(len(conflict_ids)),
                message="A canonical document identity maps to multiple content hashes.",
                examples=tuple(str(item) for item in conflict_ids[:5]),
            )
        )

    exact_duplicates = int(frame["is_exact_duplicate"].sum())
    if exact_duplicates:
        issues.append(
            ValidationIssue(
                code="exact_duplicate_rows",
                severity="warning",
                count=exact_duplicates,
                message="Exact source duplicates were retained and flagged.",
            )
        )

    duplicate_content = int(frame["is_duplicate_content"].sum())
    if duplicate_content:
        issues.append(
            ValidationIssue(
                code="duplicate_content_rows",
                severity="warning",
                count=duplicate_content,
                message="Repeated content was retained for review.",
            )
        )

    high_removal_mask = frame["removal_ratio"].gt(0.25)
    high_removal_count = int(high_removal_mask.sum())
    if high_removal_count:
        issues.append(
            ValidationIssue(
                code="large_text_reductions",
                severity="warning",
                count=high_removal_count,
                message="Cleaning removed more than 25% of source characters.",
                examples=_examples(frame["document_id"], high_removal_mask),
            )
        )

    mojibake_mask = frame["raw_text"].str.contains("\u00e2", regex=False, na=False)
    mojibake_count = int(mojibake_mask.sum())
    if mojibake_count:
        issues.append(
            ValidationIssue(
                code="possible_mojibake",
                severity="warning",
                count=mojibake_count,
                message="Documents contain characters associated with encoding artifacts.",
                examples=_examples(frame["document_id"], mojibake_mask),
            )
        )

    date_counts = frame.groupby("publication_date", dropna=True).size()
    multi_date_groups = int(date_counts.gt(1).sum())
    cleaning_rule_mask = frame["cleaning_rules"].ne("")
    metrics: dict[str, Any] = {
        "unique_document_ids": int(frame["document_id"].nunique()),
        "unique_canonical_urls": int(frame["canonical_url"].nunique()),
        "unique_content_hashes": int(frame["content_sha256"].nunique()),
        "unique_publication_dates": int(frame["publication_date"].nunique()),
        "multi_document_date_groups": multi_date_groups,
        "documents_on_multi_document_dates": int(date_counts[date_counts.gt(1)].sum()),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_content_rows": duplicate_content,
        "date_only_timing_rows": int(frame["publication_timestamp"].isna().sum()),
        "large_text_reduction_rows": high_removal_count,
        "documents_with_cleaning_rules": int(cleaning_rule_mask.sum()),
        "maximum_removal_ratio": float(frame["removal_ratio"].max()),
        "total_characters_removed": int(frame["characters_removed"].sum()),
        "cleaning_rule_counts": frame.loc[cleaning_rule_mask, "cleaning_rules"]
        .value_counts()
        .to_dict(),
        "possible_mojibake_rows": mojibake_count,
    }
    return ValidationReport(
        dataset="federal_reserve_documents",
        input_rows=input_rows,
        output_rows=len(frame),
        usable_rows=int(frame["record_valid"].sum()),
        metrics=metrics,
        issues=tuple(issues),
    )


def validate_vix_observations(frame: pd.DataFrame, *, input_rows: int) -> ValidationReport:
    """Validate prepared market observations and report every exclusion reason."""

    issues: list[ValidationIssue] = []
    invalid_date_mask = frame["trading_date"].isna()
    if invalid_date_mask.any():
        issues.append(
            ValidationIssue(
                code="invalid_trading_dates",
                severity="error",
                count=int(invalid_date_mask.sum()),
                message="Market rows contain unparseable trading dates.",
            )
        )

    duplicate_date_mask = frame["trading_date"].duplicated(keep=False) & frame[
        "trading_date"
    ].notna()
    if duplicate_date_mask.any():
        issues.append(
            ValidationIssue(
                code="duplicate_trading_dates",
                severity="error",
                count=int(duplicate_date_mask.sum()),
                message="A market series contains more than one row for a trading date.",
                examples=_examples(frame["trading_date"], duplicate_date_mask),
            )
        )

    missing_value_mask = frame["value"].isna()
    if missing_value_mask.any():
        issues.append(
            ValidationIssue(
                code="missing_market_values",
                severity="warning",
                count=int(missing_value_mask.sum()),
                message="Rows with missing values were retained and marked unusable.",
                examples=_examples(frame["trading_date"], missing_value_mask),
            )
        )

    negative_mask = frame["value"].lt(0).fillna(False)
    if negative_mask.any():
        issues.append(
            ValidationIssue(
                code="negative_market_values",
                severity="error",
                count=int(negative_mask.sum()),
                message="VIX values must not be negative.",
                examples=_examples(frame["trading_date"], negative_mask),
            )
        )

    usable = frame.loc[frame["is_usable"], "value"]
    metrics: dict[str, Any] = {
        "unique_trading_dates": int(frame["trading_date"].nunique()),
        "missing_value_rows": int(missing_value_mask.sum()),
        "excluded_rows": int((~frame["is_usable"]).sum()),
        "minimum_usable_value": float(usable.min()) if not usable.empty else None,
        "maximum_usable_value": float(usable.max()) if not usable.empty else None,
    }
    return ValidationReport(
        dataset="vix_observations",
        input_rows=input_rows,
        output_rows=len(frame),
        usable_rows=int(frame["is_usable"].sum()),
        metrics=metrics,
        issues=tuple(issues),
    )

