"""Run a bounded Gemini evaluation on expert sentences and local statements."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from fed_policy_intelligence.data.expert_labels import (
    DATASET_ID,
    DATASET_REVISION,
    load_expert_dataset,
)
from fed_policy_intelligence.evaluation.classification import evaluate_labels
from fed_policy_intelligence.llm.extractor import PolicyIntelligenceExtractor
from fed_policy_intelligence.llm.gemini import GeminiPolicyProvider
from fed_policy_intelligence.llm.pipeline import run_extraction
from fed_policy_intelligence.settings import load_settings

EXPERT_LABELS = ("dovish", "hawkish", "neutral")


def stratified_expert_sample(
    test_data: pd.DataFrame, *, size: int, seed: int
) -> pd.DataFrame:
    """Select a deterministic near-balanced sample from the untouched test split."""

    if size < len(EXPERT_LABELS):
        raise ValueError(f"Expert sample size must be at least {len(EXPERT_LABELS)}.")
    required = {"sentence", "policy_stance"}
    missing = sorted(required - set(test_data.columns))
    if missing:
        raise ValueError(f"Expert test data is missing columns: {missing}")
    indexed = test_data.reset_index(names="expert_row_id")
    base, remainder = divmod(size, len(EXPERT_LABELS))
    parts: list[pd.DataFrame] = []
    for offset, label in enumerate(EXPERT_LABELS):
        group = indexed.loc[indexed["policy_stance"].eq(label)]
        count = base + (1 if offset < remainder else 0)
        if len(group) < count:
            raise ValueError(f"Not enough {label!r} rows for a sample of {count}.")
        parts.append(group.sample(n=count, random_state=seed + offset))
    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    sample = sample.reset_index(drop=True)
    sample["document_id"] = sample["expert_row_id"].map(
        lambda value: f"expert-test-{int(value):04d}"
    )
    sample["cleaned_text"] = sample["sentence"]
    return sample


def evenly_spaced_statement_sample(documents: pd.DataFrame, *, size: int) -> pd.DataFrame:
    """Cover the statement timeline without introducing a content-based selection."""

    statements = documents.loc[documents["document_type"].eq("statement")].copy()
    statements = statements.sort_values(["publication_date", "document_id"]).reset_index(drop=True)
    if size < 1 or size > len(statements):
        raise ValueError("Document sample size must be between 1 and the statement count.")
    if size == 1:
        positions = [0]
    else:
        positions = [round(index * (len(statements) - 1) / (size - 1)) for index in range(size)]
    return statements.iloc[positions].reset_index(drop=True)


def _normalized_stance(record: dict[str, Any]) -> str | None:
    result = record.get("result")
    if record.get("status") != "success" or not isinstance(result, dict):
        return None
    if result.get("abstained") or result.get("policy_stance") == "insufficient_evidence":
        return None
    stance = str(result.get("policy_stance"))
    return "neutral" if stance == "neutral_or_mixed" else stance


def evaluate_expert_sample(
    sample: pd.DataFrame, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Report all-row accuracy plus selective metrics when Gemini does not abstain."""

    by_id = {str(record["document_id"]): record for record in records}
    rows: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        record = by_id[str(row.document_id)]
        prediction = _normalized_stance(record)
        result = record.get("result") or {}
        rows.append(
            {
                "document_id": str(row.document_id),
                "sentence": str(row.sentence),
                "expected_policy_stance": str(row.policy_stance),
                "raw_policy_stance": result.get("policy_stance"),
                "predicted_policy_stance": prediction,
                "status": record.get("status"),
                "evidence_supported": bool(record.get("evidence_supported")),
                "abstained": bool(result.get("abstained", True)),
            }
        )
    predictions = pd.DataFrame.from_records(rows)
    covered = predictions["predicted_policy_stance"].notna()
    correct = predictions["expected_policy_stance"].eq(
        predictions["predicted_policy_stance"]
    )
    insufficient = predictions["raw_policy_stance"].eq("insufficient_evidence")
    inconsistent_abstention = insufficient.ne(predictions["abstained"])
    payload: dict[str, Any] = {
        "sample_size": len(predictions),
        "expected_distribution": predictions["expected_policy_stance"].value_counts().to_dict(),
        "raw_prediction_distribution": predictions["raw_policy_stance"].value_counts(
            dropna=False
        ).to_dict(),
        "coverage": float(covered.mean()),
        "all_row_accuracy": float(correct.mean()),
        "schema_validity_rate": float(
            predictions["status"].isin(["success", "invalid_evidence"]).mean()
        ),
        "evidence_support_rate": float(predictions["evidence_supported"].mean()),
        "abstention_semantic_consistency_rate": float((~inconsistent_abstention).mean()),
        "abstention_semantic_inconsistency_count": int(inconsistent_abstention.sum()),
    }
    if covered.any():
        report = evaluate_labels(
            predictions.loc[covered, "expected_policy_stance"].tolist(),
            predictions.loc[covered, "predicted_policy_stance"].tolist(),
            labels=list(EXPERT_LABELS),
        )
        payload["covered_accuracy"] = float(correct.loc[covered].mean())
        payload["covered_macro_f1"] = report.macro_f1
        payload["covered_confusion_matrix"] = report.confusion_matrix
        payload["covered_per_class"] = {
            label: asdict(metrics) for label, metrics in report.per_class.items()
        }
    else:
        payload["covered_accuracy"] = None
        payload["covered_macro_f1"] = None
        payload["covered_confusion_matrix"] = None
        payload["covered_per_class"] = None
    return payload, predictions


def evaluate_local_reference(
    sample: pd.DataFrame, reference_predictions: pd.DataFrame
) -> dict[str, Any]:
    """Score saved local RoBERTa predictions on the identical sampled rows."""

    required = {"expected_policy_stance", "predicted_policy_stance"}
    missing = sorted(required - set(reference_predictions.columns))
    if missing:
        raise ValueError(f"Reference predictions are missing columns: {missing}")
    row_ids = sample["expert_row_id"].astype(int).tolist()
    if max(row_ids) >= len(reference_predictions):
        raise ValueError("Reference predictions do not cover every sampled expert row.")
    selected = reference_predictions.iloc[row_ids].reset_index(drop=True)
    expected = sample["policy_stance"].astype(str).tolist()
    if selected["expected_policy_stance"].astype(str).tolist() != expected:
        raise ValueError("Reference prediction rows do not match the pinned expert split.")
    predicted = selected["predicted_policy_stance"].astype(str).tolist()
    report = evaluate_labels(expected, predicted, labels=list(EXPERT_LABELS))
    return {
        "sample_size": len(expected),
        "accuracy": sum(truth == guess for truth, guess in zip(expected, predicted, strict=True))
        / len(expected),
        "macro_f1": report.macro_f1,
        "confusion_matrix": report.confusion_matrix,
        "per_class": {
            label: asdict(metrics) for label, metrics in report.per_class.items()
        },
    }


def summarize_document_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize contract reliability, grounding, abstention, stance, and usage."""

    total = len(records)
    results = [record["result"] for record in records if isinstance(record.get("result"), dict)]
    stance_counts = Counter(str(result["policy_stance"]) for result in results)
    usage_keys = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
    )
    usage_totals = {
        key: sum(int(record.get("usage", {}).get(key, 0) or 0) for record in records)
        for key in usage_keys
    }
    return {
        "sample_size": total,
        "status_counts": dict(Counter(str(record["status"]) for record in records)),
        "schema_validity_rate": sum(record.get("result") is not None for record in records)
        / total,
        "evidence_support_rate": sum(
            bool(record.get("evidence_supported")) for record in records
        )
        / total,
        "unsupported_quote_count": sum(
            len(record.get("unsupported_evidence", [])) for record in records
        ),
        "abstention_rate": sum(bool(result.get("abstained")) for result in results) / total,
        "stance_distribution": dict(stance_counts),
        "usage_totals": usage_totals,
    }


def _progress(label: str, *, request_delay_seconds: float):
    def report(position: int, total: int, record: dict[str, Any], reused: bool) -> None:
        source = "cache" if reused else "api"
        print(
            f"{label} {position}/{total}: {record['status']} ({source})",
            flush=True,
        )
        if not reused and position < total and request_delay_seconds:
            time.sleep(request_delay_seconds)

    return report


def _write_json_atomic(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--expert-size", type=int, default=30)
    parser.add_argument("--document-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="cache/huggingface/datasets")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--request-delay-seconds", type=float, default=6.0)
    args = parser.parse_args(argv)
    if args.expert_size < 3:
        parser.error("--expert-size must be at least 3.")
    if args.document_size < 1:
        parser.error("--document-size must be positive.")
    if args.request_delay_seconds < 0:
        parser.error("--request-delay-seconds must not be negative.")

    settings = load_settings(args.config)
    if settings.llm.provider != "gemini":
        raise RuntimeError("Set FPI_LLM_PROVIDER=gemini for the bounded live evaluation.")
    extractor = PolicyIntelligenceExtractor(GeminiPolicyProvider(settings.llm.model))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_dataset = load_expert_dataset(cache_dir=args.cache_dir)
    expert_sample = stratified_expert_sample(
        expert_dataset.test, size=args.expert_size, seed=args.seed
    )
    expert_cache = output_dir / "gemini_expert_evaluation.jsonl"
    expert_records, expert_reused = run_extraction(
        expert_sample[["document_id", "cleaned_text"]],
        extractor=extractor,
        cache_path=expert_cache,
        checkpoint_every=1,
        progress=_progress("expert", request_delay_seconds=args.request_delay_seconds),
    )
    expert_report, expert_predictions = evaluate_expert_sample(
        expert_sample, expert_records
    )
    expert_predictions.to_csv(output_dir / "gemini_expert_predictions.csv", index=False)
    reference_path = output_dir / "expert_policy_predictions.csv"
    local_reference = (
        evaluate_local_reference(expert_sample, pd.read_csv(reference_path))
        if reference_path.is_file()
        else None
    )

    document_path = settings.data.processed_dir / "fed_documents.csv"
    documents = pd.read_csv(document_path)
    document_sample = evenly_spaced_statement_sample(documents, size=args.document_size)
    document_cache = output_dir / "gemini_document_evaluation.jsonl"
    document_records, document_reused = run_extraction(
        document_sample[["document_id", "cleaned_text"]],
        extractor=extractor,
        cache_path=document_cache,
        checkpoint_every=1,
        progress=_progress("document", request_delay_seconds=args.request_delay_seconds),
    )
    document_report = summarize_document_records(document_records)

    report = {
        "evaluation_type": "bounded_gemini_policy_evaluation",
        "model_id": settings.llm.model,
        "seed": args.seed,
        "expert_dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "reused_cache_rows": expert_reused,
            "local_roberta_same_sample": local_reference,
            **expert_report,
        },
        "statement_documents": {
            "sampling": "evenly spaced over publication date after statement filtering",
            "reused_cache_rows": document_reused,
            **document_report,
        },
    }
    report_path = output_dir / "gemini_bounded_evaluation.json"
    _write_json_atomic(report, report_path)
    print(json.dumps({**report, "output": str(report_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
