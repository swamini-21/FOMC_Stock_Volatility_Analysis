from datetime import UTC, datetime

from fed_policy_intelligence.llm.extractor import PolicyIntelligenceExtractor
from fed_policy_intelligence.llm.provider import MockPolicyProvider


def test_mock_extraction_is_structured_and_grounded() -> None:
    text = (
        "Inflation remains elevated. The labor market remains strong. "
        "The Committee judges that policy is restrictive."
    )
    extractor = PolicyIntelligenceExtractor(
        MockPolicyProvider(),
        clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
    )

    record = extractor.extract(document_id="doc-1", source_text=text)

    assert record.status == "success"
    assert record.evidence_supported
    assert record.result is not None
    assert record.result["policy_stance"] == "hawkish"
    assert record.provider == "mock"
    assert record.prompt_version == "1.0"
    assert record.generated_at == "2026-01-02T00:00:00+00:00"


def test_empty_source_produces_valid_abstention() -> None:
    record = PolicyIntelligenceExtractor(MockPolicyProvider()).extract(
        document_id="empty", source_text=""
    )

    assert record.status == "success"
    assert record.result is not None
    assert record.result["abstained"] is True
    assert record.result["policy_stance"] == "insufficient_evidence"


class UnsupportedEvidenceProvider(MockPolicyProvider):
    model_id = "unsupported-evidence-test"

    def extract(self, source_text: str) -> dict[str, object]:
        del source_text
        return {
            "schema_version": "1.0",
            "policy_stance": "hawkish",
            "inflation_outlook": None,
            "labor_market_outlook": None,
            "economic_growth_outlook": None,
            "key_risks": [],
            "policy_rationale": [],
            "supporting_evidence": [
                {
                    "field": "policy_stance",
                    "quote": "This quote was invented.",
                    "claim_type": "inference",
                    "explanation": "Test fixture.",
                }
            ],
            "uncertainties": [],
            "abstained": False,
        }


def test_unsupported_provider_quote_is_retained_and_flagged() -> None:
    record = PolicyIntelligenceExtractor(UnsupportedEvidenceProvider()).extract(
        document_id="doc-2", source_text="Rates were unchanged."
    )

    assert record.status == "invalid_evidence"
    assert record.unsupported_evidence == ("This quote was invented.",)


class MalformedProvider(MockPolicyProvider):
    model_id = "malformed-test"

    def extract(self, source_text: str) -> dict[str, object]:
        del source_text
        return {"policy_stance": "hawkish"}


def test_malformed_provider_output_becomes_auditable_error() -> None:
    record = PolicyIntelligenceExtractor(MalformedProvider()).extract(
        document_id="doc-3", source_text="Inflation remains elevated."
    )

    assert record.status == "error"
    assert "SchemaValidationError" in record.error
    assert record.result is None
