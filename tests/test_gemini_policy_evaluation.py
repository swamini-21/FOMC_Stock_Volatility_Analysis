from pathlib import Path

import pandas as pd

from fed_policy_intelligence.evaluation.gemini_policy import (
    evaluate_expert_sample,
    evaluate_local_reference,
    evenly_spaced_statement_sample,
    stratified_expert_sample,
    summarize_document_records,
)
from fed_policy_intelligence.llm.extractor import PolicyIntelligenceExtractor
from fed_policy_intelligence.llm.pipeline import run_extraction
from fed_policy_intelligence.llm.provider import MockPolicyProvider


def test_stratified_expert_sample_is_balanced_and_deterministic() -> None:
    rows = [
        {"sentence": f"{label} {index}", "policy_stance": label}
        for label in ("dovish", "hawkish", "neutral")
        for index in range(5)
    ]
    data = pd.DataFrame(rows)

    first = stratified_expert_sample(data, size=9, seed=42)
    second = stratified_expert_sample(data, size=9, seed=42)

    assert first["document_id"].tolist() == second["document_id"].tolist()
    assert first["policy_stance"].value_counts().to_dict() == {
        "dovish": 3,
        "hawkish": 3,
        "neutral": 3,
    }


def test_evenly_spaced_statement_sample_excludes_transcripts() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["s1", "t1", "s2", "s3"],
            "document_type": ["statement", "transcript", "statement", "statement"],
            "publication_date": ["2001-01-01", "2001-01-02", "2002-01-01", "2003-01-01"],
            "cleaned_text": ["a", "b", "c", "d"],
        }
    )

    sample = evenly_spaced_statement_sample(documents, size=2)

    assert sample["document_id"].tolist() == ["s1", "s3"]


def test_expert_metrics_treat_insufficient_evidence_as_abstention() -> None:
    sample = pd.DataFrame(
        {
            "document_id": ["one", "two"],
            "sentence": ["Inflation remains elevated.", "No policy signal."],
            "policy_stance": ["hawkish", "neutral"],
        }
    )
    records = [
        {
            "document_id": "one",
            "status": "success",
            "evidence_supported": True,
            "result": {"policy_stance": "hawkish", "abstained": False},
        },
        {
            "document_id": "two",
            "status": "success",
            "evidence_supported": True,
            "result": {"policy_stance": "insufficient_evidence", "abstained": True},
        },
    ]

    report, _ = evaluate_expert_sample(sample, records)

    assert report["coverage"] == 0.5
    assert report["all_row_accuracy"] == 0.5
    assert report["covered_accuracy"] == 1.0


def test_document_summary_counts_grounding_and_usage() -> None:
    records = [
        {
            "status": "success",
            "evidence_supported": True,
            "unsupported_evidence": [],
            "usage": {"total_token_count": 10},
            "result": {"policy_stance": "hawkish", "abstained": False},
        }
    ]

    summary = summarize_document_records(records)

    assert summary["schema_validity_rate"] == 1.0
    assert summary["evidence_support_rate"] == 1.0
    assert summary["usage_totals"]["total_token_count"] == 10


def test_local_reference_uses_identical_expert_rows() -> None:
    sample = pd.DataFrame(
        {
            "expert_row_id": [2, 0],
            "policy_stance": ["neutral", "dovish"],
        }
    )
    reference = pd.DataFrame(
        {
            "expected_policy_stance": ["dovish", "hawkish", "neutral"],
            "predicted_policy_stance": ["dovish", "neutral", "neutral"],
        }
    )

    report = evaluate_local_reference(sample, reference)

    assert report["accuracy"] == 1.0
    assert report["sample_size"] == 2


def test_checkpointed_extraction_writes_each_completed_record() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["one", "two"],
            "cleaned_text": ["Inflation remains elevated.", "Growth slowed."],
        }
    )
    destination = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / ".gemini_checkpoint_test.jsonl"
    )
    try:
        records, reused = run_extraction(
            documents,
            extractor=PolicyIntelligenceExtractor(MockPolicyProvider()),
            cache_path=destination,
            checkpoint_every=1,
        )

        assert reused == 0
        assert len(records) == 2
        assert len(destination.read_text(encoding="utf-8").splitlines()) == 2
    finally:
        destination.unlink(missing_ok=True)
        destination.with_suffix(destination.suffix + ".tmp").unlink(missing_ok=True)
