"""Apply the validated sentence classifier and aggregate document stance features."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from fed_policy_intelligence.nlp.policy_model import POLICY_LABELS, DomainPolicyAnalyzer
from fed_policy_intelligence.nlp.prediction import ClassificationPrediction
from fed_policy_intelligence.nlp.transformer import load_huggingface_classifier

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class PolicyAnalyzerLike(Protocol):
    def predict_many(self, texts: list[str]) -> list[ClassificationPrediction]: ...


def split_policy_sentences(text: str) -> list[str]:
    """Split cleaned FOMC text into bounded inference units without extra models."""

    normalized = re.sub(r"[\t\r ]+", " ", str(text)).strip()
    if not normalized:
        return []
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", normalized):
        sentences.extend(
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY.split(paragraph.strip())
            if sentence.strip()
        )
    return sentences


def aggregate_sentence_predictions(
    predictions: Sequence[ClassificationPrediction],
) -> dict[str, object]:
    """Create transparent hard-label shares and mean probability features."""

    if not predictions:
        raise ValueError("At least one sentence prediction is required.")
    count = len(predictions)
    shares = {
        label: sum(prediction.label == label for prediction in predictions) / count
        for label in POLICY_LABELS
    }
    mean_probabilities = {
        label: sum(prediction.probabilities[label] for prediction in predictions) / count
        for label in POLICY_LABELS
    }
    document_label = max(mean_probabilities, key=mean_probabilities.get)  # type: ignore[arg-type]
    return {
        "domain_policy_stance": document_label,
        "domain_policy_probabilities": json.dumps(mean_probabilities, sort_keys=True),
        "domain_policy_sentence_count": count,
        **{f"domain_policy_share_{label}": shares[label] for label in POLICY_LABELS},
        "domain_policy_net_stance": shares["hawkish"] - shares["dovish"],
        "domain_policy_mean_confidence": sum(
            max(prediction.probabilities.values()) for prediction in predictions
        )
        / count,
    }


def analyze_document_policy(
    documents: pd.DataFrame,
    *,
    analyzer: PolicyAnalyzerLike,
    sentence_batch_size: int = 256,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Generate one auditable domain-policy row for every source document."""

    required = {"document_id", "cleaned_text", "processing_version"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"Document data is missing required columns: {missing}")
    if sentence_batch_size < 1:
        raise ValueError("sentence_batch_size must be positive.")

    records: list[dict[str, object]] = []
    total = len(documents)
    for position, row in enumerate(documents.itertuples(index=False), start=1):
        base: dict[str, object] = {
            "document_id": str(row.document_id),
            "processing_version": str(row.processing_version),
            "domain_policy_status": "success",
            "domain_policy_error": "",
        }
        try:
            sentences = split_policy_sentences(str(row.cleaned_text))
            if not sentences:
                raise ValueError("Document has no non-empty sentences.")
            predictions: list[ClassificationPrediction] = []
            for start in range(0, len(sentences), sentence_batch_size):
                predictions.extend(
                    analyzer.predict_many(sentences[start : start + sentence_batch_size])
                )
            if len(predictions) != len(sentences):
                raise ValueError("Analyzer returned a different number of sentence predictions.")
            base.update(aggregate_sentence_predictions(predictions))
            base.update(
                {
                    "domain_policy_model_id": predictions[0].model_id,
                    "domain_policy_tokenizer_id": predictions[0].tokenizer_id,
                    "domain_policy_device": predictions[0].device,
                }
            )
        except Exception as exc:
            base["domain_policy_status"] = "error"
            base["domain_policy_error"] = f"{type(exc).__name__}: {exc}"
        records.append(base)
        if progress_callback is not None:
            progress_callback(position, total)
    return pd.DataFrame.from_records(records)


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/processed/fed_documents.csv")
    parser.add_argument("--model-id", default="artifacts/models/fomc-roberta-base")
    parser.add_argument("--model-cache-dir", default="cache/huggingface/models")
    parser.add_argument("--output", default="data/processed/domain_policy_predictions.csv")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sentence-batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive.")

    documents = pd.read_csv(args.documents)
    if args.document_type is not None:
        if "document_type" not in documents.columns:
            raise ValueError("Document data has no document_type column for filtering.")
        documents = documents.loc[
            documents["document_type"].astype(str).eq(args.document_type)
        ].copy()
    if args.limit is not None:
        documents = documents.head(args.limit)
    classifier = load_huggingface_classifier(
        args.model_id,
        max_length=512,
        stride=64,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.model_cache_dir,
    )
    predictions = analyze_document_policy(
        documents,
        analyzer=DomainPolicyAnalyzer(classifier),
        sentence_batch_size=args.sentence_batch_size,
        progress_callback=lambda current, total: (
            print(json.dumps({"processed": current, "total": total}), flush=True)
            if current % args.progress_every == 0 or current == total
            else None
        ),
    )
    destination = Path(args.output).resolve()
    _write_csv_atomic(predictions, destination)
    print(
        json.dumps(
            {
                "rows": len(predictions),
                "successful_rows": int(predictions["domain_policy_status"].eq("success").sum()),
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
