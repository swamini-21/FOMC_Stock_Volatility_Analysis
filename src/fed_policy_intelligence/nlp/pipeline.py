"""Phase 4 document-to-NLP-feature orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from fed_policy_intelligence.nlp.emotion import EmotionAnalyzer
from fed_policy_intelligence.nlp.policy import classify_policy_stance
from fed_policy_intelligence.nlp.sentiment import FinancialSentimentAnalyzer
from fed_policy_intelligence.nlp.transformer import load_huggingface_classifier
from fed_policy_intelligence.settings import ProjectSettings, load_settings


class SentimentLike(Protocol):
    def predict(self, text: str, *, aggregation: str = ...) -> object: ...


def analyze_documents(
    documents: pd.DataFrame,
    *,
    sentiment: FinancialSentimentAnalyzer,
    emotion: EmotionAnalyzer,
    sentiment_aggregation: str,
    emotion_aggregation: str,
) -> pd.DataFrame:
    """Build auditable document features without dropping failed source rows."""

    required = {"document_id", "cleaned_text", "processing_version"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"Document data is missing required columns: {missing}")

    records: list[dict[str, object]] = []
    for row in documents.itertuples(index=False):
        document_id = str(row.document_id)
        text = str(row.cleaned_text)
        base: dict[str, object] = {
            "document_id": document_id,
            "processing_version": str(row.processing_version),
            "nlp_status": "success",
            "nlp_error": "",
        }
        try:
            sentiment_result = sentiment.predict(text, aggregation=sentiment_aggregation)
            emotion_result = emotion.predict(text, aggregation=emotion_aggregation)
            stance = classify_policy_stance(text)
            sentiment_record = sentiment_result.to_record(document_id=document_id)
            emotion_record = emotion_result.to_record(document_id=document_id)
            base.update(
                {
                    "sentiment_label": sentiment_record.pop("label"),
                    "sentiment_score": sentiment_record.pop("sentiment_score"),
                    "sentiment_probabilities": json.dumps(
                        sentiment_result.classification.probabilities, sort_keys=True
                    ),
                    "sentiment_chunk_count": sentiment_result.classification.chunk_count,
                    "sentiment_token_count": sentiment_result.classification.token_count,
                    "sentiment_aggregation": sentiment_result.classification.aggregation,
                    "sentiment_model_id": sentiment_result.classification.model_id,
                    "sentiment_tokenizer_id": sentiment_result.classification.tokenizer_id,
                    "sentiment_device": sentiment_result.classification.device,
                    "emotion_label": emotion_record.pop("label"),
                    "emotion_probabilities": json.dumps(
                        emotion_result.probabilities, sort_keys=True
                    ),
                    "emotion_chunk_count": emotion_result.chunk_count,
                    "emotion_token_count": emotion_result.token_count,
                    "emotion_aggregation": emotion_result.aggregation,
                    "emotion_model_id": emotion_result.model_id,
                    "emotion_tokenizer_id": emotion_result.tokenizer_id,
                    "emotion_device": emotion_result.device,
                    "policy_stance": stance.stance.value,
                    "policy_method": stance.method,
                    "policy_evidence": json.dumps(stance.supporting_evidence),
                }
            )
        except Exception as exc:  # preserve the row and make the failure explicit
            base["nlp_status"] = "error"
            base["nlp_error"] = f"{type(exc).__name__}: {exc}"
        records.append(base)
    return pd.DataFrame.from_records(records)


def build_analyzers(
    settings: ProjectSettings,
) -> tuple[FinancialSentimentAnalyzer, EmotionAnalyzer]:
    """Load each model with its own tokenizer from the same configured checkpoint."""

    common = {
        "max_length": settings.models.max_length,
        "stride": settings.models.stride,
        "batch_size": settings.models.batch_size,
        "device": settings.runtime.device,
    }
    sentiment = FinancialSentimentAnalyzer(
        load_huggingface_classifier(settings.models.sentiment_model, **common)
    )
    emotion = EmotionAnalyzer(
        load_huggingface_classifier(settings.models.emotion_model, **common)
    )
    return sentiment, emotion


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    """Run real transformer inference for a bounded or complete processed corpus."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    parser.add_argument(
        "--sentiment-aggregation",
        choices=("document_mean", "document_length_weighted", "paragraph_mean"),
        default="document_length_weighted",
    )
    parser.add_argument(
        "--emotion-aggregation",
        choices=("document_mean", "document_length_weighted", "paragraph_mean"),
        default="document_length_weighted",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    settings = load_settings(args.config)
    source_path = settings.data.processed_dir / "fed_documents.csv"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Processed documents not found at {source_path}; run the Phase 3 pipeline first."
        )
    documents = pd.read_csv(source_path)
    if args.limit is not None:
        documents = documents.head(args.limit)
    sentiment, emotion = build_analyzers(settings)
    output = analyze_documents(
        documents,
        sentiment=sentiment,
        emotion=emotion,
        sentiment_aggregation=args.sentiment_aggregation,
        emotion_aggregation=args.emotion_aggregation,
    )
    destination = settings.data.processed_dir / "nlp_predictions.csv"
    _write_csv_atomic(output, destination)
    print(json.dumps({"rows": len(output), "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
