"""Evaluate saved NLP predictions against reviewed annotation labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from fed_policy_intelligence.evaluation.classification import evaluate_labels
from fed_policy_intelligence.settings import load_settings

ANNOTATION_COLUMNS = (
    "document_id",
    "split",
    "financial_sentiment",
    "emotion",
    "policy_stance",
    "policy_evidence",
    "annotator_id",
    "annotated_at",
    "annotator_notes",
)

TASKS = {
    "financial_sentiment": {
        "prediction": "sentiment_label",
        "labels": ("negative", "neutral", "positive"),
    },
    "emotion": {
        "prediction": "emotion_label",
        "labels": ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"),
    },
    "policy_stance": {
        "prediction": "policy_stance",
        "labels": ("hawkish", "dovish", "neutral_or_mixed", "insufficient_evidence"),
    },
}


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_reviewed_annotations(path: str | Path, *, split: str = "evaluation") -> pd.DataFrame:
    """Load and validate the reviewed annotation subset used as gold labels."""

    annotations = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(annotations, set(ANNOTATION_COLUMNS), "Annotation data")
    annotations = annotations.loc[:, ANNOTATION_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        annotations[column] = annotations[column].str.strip()

    selected = annotations.loc[annotations["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Annotation data contains no rows for split {split!r}.")
    duplicates = selected.loc[selected["document_id"].duplicated(), "document_id"].tolist()
    if duplicates:
        raise ValueError(f"Annotation split contains duplicate document_id values: {duplicates}")

    for task, specification in TASKS.items():
        allowed = set(specification["labels"])
        invalid = sorted(set(selected[task]) - allowed)
        if invalid:
            raise ValueError(f"Annotation column {task!r} contains invalid labels: {invalid}")
    return selected


def evaluate_prediction_frame(
    annotations: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    """Join predictions to gold labels and return reports plus row-level mismatches."""

    prediction_columns = {
        "document_id",
        "nlp_status",
        *(specification["prediction"] for specification in TASKS.values()),
    }
    _require_columns(predictions, prediction_columns, "Prediction data")
    if predictions["document_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["document_id"].duplicated(), "document_id"
        ].tolist()
        raise ValueError(f"Prediction data contains duplicate document_id values: {duplicates}")

    renamed = predictions.rename(
        columns={
            "sentiment_label": "predicted_financial_sentiment",
            "emotion_label": "predicted_emotion",
            "policy_stance": "predicted_policy_stance",
        }
    )
    joined = annotations.merge(
        renamed,
        on="document_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", "document_id"].tolist()
    if missing:
        raise ValueError(f"Reviewed documents are missing predictions: {missing}")
    failed = joined.loc[joined["nlp_status"] != "success", "document_id"].tolist()
    if failed:
        raise ValueError(f"Reviewed documents have failed NLP predictions: {failed}")

    task_reports: dict[str, object] = {}
    mismatch_records: list[dict[str, object]] = []
    for task, specification in TASKS.items():
        predicted_column = f"predicted_{task}"
        missing_labels = joined.loc[joined[predicted_column].isna(), "document_id"].tolist()
        if missing_labels:
            raise ValueError(f"Task {task!r} has missing predictions: {missing_labels}")
        report = evaluate_labels(
            joined[task].tolist(),
            joined[predicted_column].astype(str).tolist(),
            labels=list(specification["labels"]),
        )
        task_reports[task] = asdict(report)
        mismatches = joined.loc[joined[task] != joined[predicted_column]]
        for row in mismatches.itertuples(index=False):
            mismatch_records.append(
                {
                    "document_id": row.document_id,
                    "task": task,
                    "expected": getattr(row, task),
                    "predicted": getattr(row, predicted_column),
                }
            )

    metadata_fields = (
        "processing_version",
        "sentiment_model_id",
        "sentiment_tokenizer_id",
        "sentiment_aggregation",
        "sentiment_device",
        "emotion_model_id",
        "emotion_tokenizer_id",
        "emotion_aggregation",
        "emotion_device",
        "policy_method",
    )
    metadata = {
        field: sorted(joined[field].dropna().astype(str).unique().tolist())
        for field in metadata_fields
        if field in joined.columns
    }
    result: dict[str, object] = {
        "evaluation_date": date.today().isoformat(),
        "split": annotations["split"].iloc[0],
        "sample_size": len(joined),
        "document_ids": joined["document_id"].tolist(),
        "run_metadata": metadata,
        "tasks": task_reports,
    }
    errors = pd.DataFrame.from_records(
        mismatch_records,
        columns=("document_id", "task", "expected", "predicted"),
    )
    return result, errors


def _write_text_atomic(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    """Run reviewed-label evaluation for saved Phase 4 predictions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--annotations",
        default="data/annotations/fomc_evaluation_annotations.csv",
    )
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--errors-output", default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    annotations_path = (settings.root_dir / args.annotations).resolve()
    predictions_path = (
        Path(args.predictions).resolve()
        if args.predictions
        else settings.data.processed_dir / "nlp_predictions.csv"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else settings.data.processed_dir / "model_evaluation.json"
    )
    errors_path = (
        Path(args.errors_output).resolve()
        if args.errors_output
        else settings.data.processed_dir / "model_evaluation_errors.csv"
    )

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Reviewed annotations not found: {annotations_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"NLP predictions not found: {predictions_path}")

    annotations = load_reviewed_annotations(annotations_path, split=args.split)
    predictions = pd.read_csv(predictions_path)
    report, errors = evaluate_prediction_frame(annotations, predictions)
    _write_text_atomic(json.dumps(report, indent=2), output_path)
    _write_csv_atomic(errors, errors_path)
    summary = {
        "sample_size": report["sample_size"],
        "split": report["split"],
        "macro_f1": {
            task: task_report["macro_f1"]
            for task, task_report in report["tasks"].items()
        },
        "mismatches": len(errors),
        "output": str(output_path),
        "errors_output": str(errors_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
