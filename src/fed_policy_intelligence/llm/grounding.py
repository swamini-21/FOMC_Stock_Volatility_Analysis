"""Verify that provider evidence is present verbatim in the supplied source."""

from __future__ import annotations

from dataclasses import dataclass

from fed_policy_intelligence.llm.schema import PolicyIntelligence


@dataclass(frozen=True)
class EvidenceAudit:
    supported: bool
    evidence_count: int
    unsupported_quotes: tuple[str, ...]
    missing_required_evidence: bool


def audit_evidence(result: PolicyIntelligence, source_text: str) -> EvidenceAudit:
    """Flag unsupported excerpts and missing evidence without rewriting model output."""

    unsupported = tuple(
        item.quote for item in result.supporting_evidence if item.quote not in source_text
    )
    missing = not result.abstained and not result.supporting_evidence
    return EvidenceAudit(
        supported=not unsupported and not missing,
        evidence_count=len(result.supporting_evidence),
        unsupported_quotes=unsupported,
        missing_required_evidence=missing,
    )
