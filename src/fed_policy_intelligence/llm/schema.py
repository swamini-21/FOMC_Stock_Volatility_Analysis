"""Strict dependency-light schema for structured policy intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
POLICY_STANCES = {
    "hawkish",
    "dovish",
    "neutral_or_mixed",
    "insufficient_evidence",
}
EVIDENCE_FIELDS = {
    "policy_stance",
    "inflation_outlook",
    "labor_market_outlook",
    "economic_growth_outlook",
    "key_risk",
    "policy_rationale",
}
CLAIM_TYPES = {"explicit", "inference"}


class SchemaValidationError(ValueError):
    """Raised when provider output violates the versioned contract."""


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{field} must be a JSON array of strings.")
    return tuple(_non_empty_string(item, f"{field} item") for item in value)


@dataclass(frozen=True)
class EvidenceItem:
    """One source-grounded excerpt and the claim it supports."""

    field: str
    quote: str
    claim_type: Literal["explicit", "inference"]
    explanation: str

    @classmethod
    def from_mapping(cls, value: Any) -> EvidenceItem:
        if not isinstance(value, dict):
            raise SchemaValidationError("Every supporting_evidence item must be an object.")
        expected = {"field", "quote", "claim_type", "explanation"}
        extras = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if extras or missing:
            raise SchemaValidationError(
                f"Evidence fields mismatch; missing={missing}, unexpected={extras}."
            )
        field = _non_empty_string(value["field"], "evidence.field")
        if field not in EVIDENCE_FIELDS:
            raise SchemaValidationError(f"Unsupported evidence field: {field}")
        claim_type = _non_empty_string(value["claim_type"], "evidence.claim_type")
        if claim_type not in CLAIM_TYPES:
            raise SchemaValidationError(f"Unsupported claim_type: {claim_type}")
        return cls(
            field=field,
            quote=_non_empty_string(value["quote"], "evidence.quote"),
            claim_type=claim_type,  # type: ignore[arg-type]
            explanation=_non_empty_string(value["explanation"], "evidence.explanation"),
        )


@dataclass(frozen=True)
class PolicyIntelligence:
    """Validated policy analysis; absent claims use null rather than invention."""

    schema_version: str
    policy_stance: str
    inflation_outlook: str | None
    labor_market_outlook: str | None
    economic_growth_outlook: str | None
    key_risks: tuple[str, ...]
    policy_rationale: tuple[str, ...]
    supporting_evidence: tuple[EvidenceItem, ...]
    uncertainties: tuple[str, ...]
    abstained: bool

    @classmethod
    def from_mapping(cls, value: Any) -> PolicyIntelligence:
        if not isinstance(value, dict):
            raise SchemaValidationError("Structured provider output must be a JSON object.")
        expected = {
            "schema_version",
            "policy_stance",
            "inflation_outlook",
            "labor_market_outlook",
            "economic_growth_outlook",
            "key_risks",
            "policy_rationale",
            "supporting_evidence",
            "uncertainties",
            "abstained",
        }
        extras = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if extras or missing:
            raise SchemaValidationError(
                f"Policy intelligence fields mismatch; missing={missing}, unexpected={extras}."
            )
        schema_version = _non_empty_string(value["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}."
            )
        stance = _non_empty_string(value["policy_stance"], "policy_stance")
        if stance not in POLICY_STANCES:
            raise SchemaValidationError(f"Unsupported policy_stance: {stance}")
        if not isinstance(value["abstained"], bool):
            raise SchemaValidationError("abstained must be a JSON Boolean.")
        evidence_raw = value["supporting_evidence"]
        if not isinstance(evidence_raw, list):
            raise SchemaValidationError("supporting_evidence must be a JSON array.")
        evidence = tuple(EvidenceItem.from_mapping(item) for item in evidence_raw)
        result = cls(
            schema_version=schema_version,
            policy_stance=stance,
            inflation_outlook=_optional_string(
                value["inflation_outlook"], "inflation_outlook"
            ),
            labor_market_outlook=_optional_string(
                value["labor_market_outlook"], "labor_market_outlook"
            ),
            economic_growth_outlook=_optional_string(
                value["economic_growth_outlook"], "economic_growth_outlook"
            ),
            key_risks=_string_tuple(value["key_risks"], "key_risks"),
            policy_rationale=_string_tuple(value["policy_rationale"], "policy_rationale"),
            supporting_evidence=evidence,
            uncertainties=_string_tuple(value["uncertainties"], "uncertainties"),
            abstained=value["abstained"],
        )
        if result.abstained and result.policy_stance != "insufficient_evidence":
            raise SchemaValidationError(
                "An abstained result must use policy_stance='insufficient_evidence'."
            )
        if not result.abstained and not result.supporting_evidence:
            raise SchemaValidationError("A non-abstained result requires supporting evidence.")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def policy_intelligence_json_schema() -> dict[str, Any]:
    """Return the provider-facing JSON Schema for later structured-output APIs."""

    nullable_string = {"type": ["string", "null"]}
    string_array = {"type": "array", "items": {"type": "string", "minLength": 1}}
    return {
        "title": "FederalReservePolicyIntelligence",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "policy_stance",
            "inflation_outlook",
            "labor_market_outlook",
            "economic_growth_outlook",
            "key_risks",
            "policy_rationale",
            "supporting_evidence",
            "uncertainties",
            "abstained",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
            "policy_stance": {"type": "string", "enum": sorted(POLICY_STANCES)},
            "inflation_outlook": nullable_string,
            "labor_market_outlook": nullable_string,
            "economic_growth_outlook": nullable_string,
            "key_risks": string_array,
            "policy_rationale": string_array,
            "supporting_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "quote", "claim_type", "explanation"],
                    "properties": {
                        "field": {"type": "string", "enum": sorted(EVIDENCE_FIELDS)},
                        "quote": {"type": "string", "minLength": 1},
                        "claim_type": {"type": "string", "enum": sorted(CLAIM_TYPES)},
                        "explanation": {"type": "string", "minLength": 1},
                    },
                },
            },
            "uncertainties": string_array,
            "abstained": {"type": "boolean"},
        },
    }
