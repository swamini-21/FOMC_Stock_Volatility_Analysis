"""Evaluate the pinned FOMC-RoBERTa checkpoint on the untouched expert test split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fed_policy_intelligence.data.expert_labels import load_expert_dataset
from fed_policy_intelligence.evaluation.classification import evaluate_labels
from fed_policy_intelligence.nlp.policy_model import (
    POLICY_LABELS,
    DomainPolicyAnalyzer,
)
from fed_policy_intelligence.nlp.transformer import load_huggingface_classifier


def evaluate_expert_policy(
    test_data: pd.DataFrame, *, analyzer: DomainPolicyAnalyzer
) -> tuple[dict[str, object], pd.DataFrame]:
    """Score one analyzer while retaining row-level predictions and probabilities."""

    required = {"sentence", "policy_stance"}
    missing = sorted(required - set(test_data.columns))
    if missing:
        raise ValueError(f"Expert test data is missing columns: {missing}")
    if test_data.empty:
        raise ValueError("Expert test data must not be empty.")

    records: list[dict[str, object]] = []
    for row in test_data.itertuples(index=False):
        prediction = analyzer.predict(str(row.sentence))
        records.append(
            {
                "sentence": str(row.sentence),
                "expected_policy_stance": str(row.policy_stance),
                "predicted_policy_stance": prediction.label,
                **{
                    f"probability_{label}": prediction.probabilities[label]
                    for label in POLICY_LABELS
                },
            }
        )
    predictions = pd.DataFrame.from_records(records)
    report = evaluate_labels(
        predictions["expected_policy_stance"].tolist(),
        predictions["predicted_policy_stance"].tolist(),
        labels=list(POLICY_LABELS),
    )
    payload: dict[str, object] = {
        "sample_size": report.sample_size,
        "labels": report.labels,
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "macro_f1": report.macro_f1,
        "confusion_matrix": report.confusion_matrix,
        "per_class": {label: asdict(metrics) for label, metrics in report.per_class.items()},
    }
    return payload, predictions


def _write_outputs(
    report: dict[str, object], predictions: pd.DataFrame, *, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "expert_policy_evaluation.json"
    predictions_path = output_dir / "expert_policy_predictions.csv"
    report_temporary = report_path.with_suffix(".json.tmp")
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    report_temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    predictions.to_csv(predictions_temporary, index=False)
    report_temporary.replace(report_path)
    predictions_temporary.replace(predictions_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="artifacts/models/fomc-roberta-base"
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default="cache/huggingface/datasets")
    parser.add_argument("--model-cache-dir", default="cache/huggingface/models")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    dataset = load_expert_dataset(cache_dir=args.cache_dir)
    test_data = dataset.test if args.limit is None else dataset.test.head(args.limit)
    classifier = load_huggingface_classifier(
        args.model_id,
        revision=args.model_revision,
        max_length=512,
        stride=64,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.model_cache_dir,
    )
    report, predictions = evaluate_expert_policy(
        test_data,
        analyzer=DomainPolicyAnalyzer(classifier),
    )
    report.update(
        {
            "dataset_id": dataset.dataset_id,
            "dataset_revision": dataset.revision,
            "model_id": args.model_id,
            "model_revision": args.model_revision or "local",
        }
    )
    output_dir = Path(args.output_dir).resolve()
    _write_outputs(report, predictions, output_dir=output_dir)
    print(json.dumps({**report, "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
