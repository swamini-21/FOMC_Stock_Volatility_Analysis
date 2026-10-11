import pytest

from fed_policy_intelligence.llm.grounding import audit_evidence
from fed_policy_intelligence.llm.schema import (
    PolicyIntelligence,
    SchemaValidationError,
    policy_intelligence_json_schema,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "policy_stance": "hawkish",
        "inflation_outlook": "Inflation remains elevated.",
        "labor_market_outlook": None,
        "economic_growth_outlook": None,
        "key_risks": ["Inflation could persist."],
        "policy_rationale": [],
        "supporting_evidence": [
            {
                "field": "policy_stance",
                "quote": "Inflation remains elevated.",
                "claim_type": "explicit",
                "explanation": "Elevated inflation supports tighter policy.",
            }
        ],
        "uncertainties": [],
        "abstained": False,
    }


def test_policy_schema_rejects_unexpected_fields() -> None:
    payload = _payload()
    payload["confidence"] = 0.99

    with pytest.raises(SchemaValidationError, match="unexpected"):
        PolicyIntelligence.from_mapping(payload)


def test_abstention_requires_insufficient_evidence_stance() -> None:
    payload = _payload()
    payload["abstained"] = True

    with pytest.raises(SchemaValidationError, match="insufficient_evidence"):
        PolicyIntelligence.from_mapping(payload)


def test_evidence_audit_flags_non_verbatim_quote() -> None:
    result = PolicyIntelligence.from_mapping(_payload())

    audit = audit_evidence(result, "Inflation has declined.")

    assert not audit.supported
    assert audit.unsupported_quotes == ("Inflation remains elevated.",)


def test_provider_json_schema_forbids_extra_properties() -> None:
    schema = policy_intelligence_json_schema()

    assert schema["additionalProperties"] is False
    assert "supporting_evidence" in schema["required"]
