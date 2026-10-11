"""Provider protocol and deterministic offline mock implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from fed_policy_intelligence.llm.schema import SCHEMA_VERSION


class StructuredPolicyProvider(Protocol):
    provider_name: str
    model_id: str

    def extract(self, source_text: str) -> ProviderOutput | dict[str, Any]: ...


@dataclass(frozen=True)
class ProviderOutput:
    """Provider payload plus non-semantic request metadata."""

    result: dict[str, Any]
    usage: dict[str, Any]


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"[\t\r ]+", " ", text).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for paragraph in re.split(r"\n+", normalized)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph)
        if sentence.strip()
    ]


def _first_match(sentences: list[str], terms: tuple[str, ...]) -> str | None:
    return next(
        (sentence for sentence in sentences if any(term in sentence.casefold() for term in terms)),
        None,
    )


class MockPolicyProvider:
    """Offline contract fixture; deterministic rules are not an LLM benchmark."""

    provider_name = "mock"
    model_id = "deterministic-policy-mock-v1"

    def extract(self, source_text: str) -> dict[str, Any]:
        sentences = _sentences(source_text)
        inflation = _first_match(sentences, ("inflation", "price stability", "prices"))
        labor = _first_match(sentences, ("employment", "labor market", "unemployment"))
        growth = _first_match(sentences, ("economic activity", "growth", "gdp"))
        risk_sentences = [sentence for sentence in sentences if "risk" in sentence.casefold()][:3]
        rationale_sentences = [
            sentence
            for sentence in sentences
            if any(term in sentence.casefold() for term in ("because", "in light of", "to achieve"))
        ][:3]
        hawkish = _first_match(
            sentences,
            ("raise the target", "tighten", "restrictive", "inflation remains elevated"),
        )
        dovish = _first_match(
            sentences,
            ("lower the target", "easing", "accommodative", "downside risks to employment"),
        )
        if hawkish and dovish:
            stance = "neutral_or_mixed"
            stance_quotes = [hawkish, dovish]
        elif hawkish:
            stance = "hawkish"
            stance_quotes = [hawkish]
        elif dovish:
            stance = "dovish"
            stance_quotes = [dovish]
        else:
            stance = "insufficient_evidence"
            stance_quotes = []

        evidence: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_evidence(field: str, quote: str | None) -> None:
            if quote is None or (field, quote) in seen:
                return
            seen.add((field, quote))
            evidence.append(
                {
                    "field": field,
                    "quote": quote,
                    "claim_type": "explicit",
                    "explanation": f"The passage explicitly supports {field.replace('_', ' ')}.",
                }
            )

        for quote in stance_quotes:
            add_evidence("policy_stance", quote)
        add_evidence("inflation_outlook", inflation)
        add_evidence("labor_market_outlook", labor)
        add_evidence("economic_growth_outlook", growth)
        for quote in risk_sentences:
            add_evidence("key_risk", quote)
        for quote in rationale_sentences:
            add_evidence("policy_rationale", quote)

        has_content = bool(evidence)
        return {
            "schema_version": SCHEMA_VERSION,
            "policy_stance": stance,
            "inflation_outlook": inflation,
            "labor_market_outlook": labor,
            "economic_growth_outlook": growth,
            "key_risks": risk_sentences,
            "policy_rationale": rationale_sentences,
            "supporting_evidence": evidence,
            "uncertainties": (
                ["Both hawkish and dovish language is present."]
                if hawkish and dovish
                else ([] if has_content else ["No supported policy signals were identified."])
            ),
            "abstained": not has_content,
        }
