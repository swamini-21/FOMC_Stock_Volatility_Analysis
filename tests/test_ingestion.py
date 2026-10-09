from pathlib import Path

import pytest

from fed_policy_intelligence.data.identity import sha256_file
from fed_policy_intelligence.data.ingestion import (
    DataValidationError,
    prepare_fed_documents,
    prepare_vix_observations,
    read_fed_corpus,
    read_vix_data,
)
from fed_policy_intelligence.data.validation import (
    validate_fed_documents,
    validate_vix_observations,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_document_pipeline_retains_same_date_records() -> None:
    path = FIXTURES / "fed_sample.csv"
    source = read_fed_corpus(path)

    documents = prepare_fed_documents(
        source,
        source_path=path,
        source_name="test",
        processing_version="test",
    )

    assert len(documents) == 3
    assert int(documents["publication_date"].eq("2020-03-15").sum()) == 2
    assert documents["document_id"].nunique() == 3


def test_document_pipeline_preserves_raw_text_and_flags_duplicate_content() -> None:
    path = FIXTURES / "fed_sample.csv"
    source = read_fed_corpus(path)
    documents = prepare_fed_documents(
        source,
        source_path=path,
        source_name="test",
        processing_version="test",
    )

    assert documents.loc[0, "raw_text"].startswith("Conference call")
    assert "We do not support" in documents.loc[0, "cleaned_text"]
    assert int(documents["is_duplicate_content"].sum()) == 2
    assert documents["source_dataset_sha256"].nunique() == 1
    assert documents.loc[0, "source_dataset_sha256"] == sha256_file(path)


def test_document_validation_reports_shared_dates_without_error() -> None:
    path = FIXTURES / "fed_sample.csv"
    source = read_fed_corpus(path)
    documents = prepare_fed_documents(
        source,
        source_path=path,
        source_name="test",
        processing_version="test",
    )

    report = validate_fed_documents(documents, input_rows=len(source))

    assert report.has_errors is False
    assert report.metrics["multi_document_date_groups"] == 1
    assert report.metrics["documents_on_multi_document_dates"] == 2


def test_vix_pipeline_retains_missing_rows_with_reason() -> None:
    path = FIXTURES / "vix_sample.csv"
    source = read_vix_data(path)
    market = prepare_vix_observations(
        source,
        source_path=path,
        source_name="test",
        series_id="VIXCLS",
        processing_version="test",
    )

    assert len(market) == 3
    assert int(market["is_usable"].sum()) == 2
    assert market.loc[2, "exclusion_reason"] == "missing_or_non_numeric_value"

    report = validate_vix_observations(market, input_rows=len(source))
    assert report.has_errors is False
    assert report.metrics["missing_value_rows"] == 1


def test_missing_required_column_is_rejected() -> None:
    path = FIXTURES / "fed_sample.csv"
    source = read_fed_corpus(path).drop(columns="url")

    with pytest.raises(DataValidationError, match="url"):
        prepare_fed_documents(
            source,
            source_path=path,
            source_name="test",
            processing_version="test",
        )


def test_invalid_document_is_retained_and_reported_as_error() -> None:
    path = FIXTURES / "fed_sample.csv"
    source = read_fed_corpus(path)
    source.loc[0, "url"] = "not-a-url"
    documents = prepare_fed_documents(
        source,
        source_path=path,
        source_name="test",
        processing_version="test",
    )

    assert len(documents) == len(source)
    assert bool(documents.loc[0, "record_valid"]) is False
    report = validate_fed_documents(documents, input_rows=len(source))
    assert report.has_errors is True


def test_negative_vix_value_is_validation_error() -> None:
    path = FIXTURES / "vix_sample.csv"
    source = read_vix_data(path)
    source.loc[0, "VIXCLS"] = -1.0
    market = prepare_vix_observations(
        source,
        source_path=path,
        source_name="test",
        series_id="VIXCLS",
        processing_version="test",
    )

    report = validate_vix_observations(market, input_rows=len(source))
    assert report.has_errors is True
    assert "negative_value" in market.loc[0, "exclusion_reason"]

