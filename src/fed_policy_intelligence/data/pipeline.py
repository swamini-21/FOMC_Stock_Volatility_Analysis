"""Command-line and Python entry points for the Phase 3 data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fed_policy_intelligence.data.identity import sha256_file
from fed_policy_intelligence.data.ingestion import (
    DataValidationError,
    prepare_fed_documents,
    prepare_vix_observations,
    read_fed_corpus,
    read_vix_data,
)
from fed_policy_intelligence.data.validation import (
    ValidationReport,
    validate_fed_documents,
    validate_vix_observations,
)
from fed_policy_intelligence.logging_config import configure_logging
from fed_policy_intelligence.settings import ProjectSettings, load_settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """In-memory outputs and optional saved paths from one pipeline run."""

    documents: pd.DataFrame
    market: pd.DataFrame
    document_report: ValidationReport
    market_report: ValidationReport
    output_paths: dict[str, Path]


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


def run_data_pipeline(
    settings: ProjectSettings,
    *,
    write_outputs: bool = True,
    fail_on_errors: bool = True,
) -> PipelineResult:
    """Load, preserve, validate, and optionally persist current project data."""

    fed_source = read_fed_corpus(settings.data.corpus_path)
    vix_source = read_vix_data(settings.data.vix_path)
    documents = prepare_fed_documents(
        fed_source,
        source_path=settings.data.corpus_path,
        source_name=settings.data.corpus_source,
        processing_version=settings.data.processing_version,
    )
    market = prepare_vix_observations(
        vix_source,
        source_path=settings.data.vix_path,
        source_name=settings.data.market_source,
        series_id=settings.data.market_series_id,
        processing_version=settings.data.processing_version,
    )
    document_report = validate_fed_documents(documents, input_rows=len(fed_source))
    market_report = validate_vix_observations(market, input_rows=len(vix_source))

    if fail_on_errors and (document_report.has_errors or market_report.has_errors):
        raise DataValidationError(
            "Data validation failed. Inspect the validation reports before creating outputs."
        )

    LOGGER.info(
        "document validation complete",
        extra={
            "input_rows": document_report.input_rows,
            "usable_rows": document_report.usable_rows,
        },
    )
    LOGGER.info(
        "market validation complete",
        extra={
            "input_rows": market_report.input_rows,
            "usable_rows": market_report.usable_rows,
        },
    )

    output_paths: dict[str, Path] = {}
    if write_outputs:
        output_dir = settings.data.processed_dir
        documents_path = output_dir / "fed_documents.csv"
        market_path = output_dir / "vix_observations.csv"
        report_path = output_dir / "data_quality_report.json"
        _write_csv_atomic(documents, documents_path)
        _write_csv_atomic(market, market_path)
        _write_json_atomic(
            {
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "processing_version": settings.data.processing_version,
                "inputs": {
                    "corpus": {
                        "path": str(settings.data.corpus_path),
                        "source_name": settings.data.corpus_source,
                        "sha256": sha256_file(settings.data.corpus_path),
                    },
                    "market": {
                        "path": str(settings.data.vix_path),
                        "source_name": settings.data.market_source,
                        "series_id": settings.data.market_series_id,
                        "sha256": sha256_file(settings.data.vix_path),
                    },
                },
                "documents": document_report.to_dict(),
                "market": market_report.to_dict(),
            },
            report_path,
        )
        output_paths = {
            "documents": documents_path,
            "market": market_path,
            "quality_report": report_path,
        }

    return PipelineResult(
        documents=documents,
        market=market,
        document_report=document_report,
        market_report=market_report,
        output_paths=output_paths,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 3 data pipeline from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation without writing processed outputs.",
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    configure_logging(settings.runtime.log_level)
    result = run_data_pipeline(settings, write_outputs=not args.validate_only)
    print(json.dumps(
        {
            "documents": result.document_report.to_dict(),
            "market": result.market_report.to_dict(),
            "outputs": {key: str(path) for key, path in result.output_paths.items()},
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

