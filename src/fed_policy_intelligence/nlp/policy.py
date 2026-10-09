"""Transparent rule baseline for monetary-policy stance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class PolicyStance(StrEnum):
    """Operational policy categories used by the baseline and later LLM work."""

    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL_OR_MIXED = "neutral_or_mixed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class PolicyStancePrediction:
    """Rule decision with inspectable supporting sentences."""

    stance: PolicyStance
    hawkish_matches: tuple[str, ...]
    dovish_matches: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    method: str = "phrase_rules_v1"


_HAWKISH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\braise(?:d|s|ing)? (?:the )?(?:target )?(?:rate|range)",
        r"\btighten(?:ed|ing)? (?:monetary )?policy",
        r"\binflation(?:ary)? (?:pressure|risk)s? remain(?:s)? elevated",
        r"\bupside risks? to inflation",
        r"\brestrictive (?:stance|policy)",
    )
)
_DOVISH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\blower(?:ed|s|ing)? (?:the )?(?:target )?(?:rate|range)",
        r"\beas(?:e|ed|ing) (?:the )?(?:stance of )?(?:monetary )?policy",
        r"\bdownside risks? to (?:employment|growth|the economy)",
        r"\baccommodative (?:stance|policy)",
        r"\bsupport (?:maximum )?employment",
    )
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _non_negated_matches(patterns: tuple[re.Pattern[str], ...], sentence: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        match = pattern.search(sentence)
        if match is None:
            continue
        prefix = sentence[: match.start()]
        nearby_words = re.findall(r"\b[\w']+\b", prefix.casefold())[-4:]
        if not {"not", "no", "never", "without"}.intersection(nearby_words):
            matches.append(match.group(0))
    return matches


def classify_policy_stance(text: str) -> PolicyStancePrediction:
    """Classify explicit policy language and abstain when no rule has evidence.

    This intentionally does not map positive sentiment to dovishness or negative
    sentiment to hawkishness. Mixed evidence is retained rather than forced into a
    directional class.
    """

    hawkish: list[str] = []
    dovish: list[str] = []
    evidence: list[str] = []
    for sentence in _sentences(text):
        hawkish_hits = _non_negated_matches(_HAWKISH_PATTERNS, sentence)
        dovish_hits = _non_negated_matches(_DOVISH_PATTERNS, sentence)
        if hawkish_hits or dovish_hits:
            evidence.append(sentence)
            hawkish.extend(hawkish_hits)
            dovish.extend(dovish_hits)

    if hawkish and dovish:
        stance = PolicyStance.NEUTRAL_OR_MIXED
    elif hawkish:
        stance = PolicyStance.HAWKISH
    elif dovish:
        stance = PolicyStance.DOVISH
    else:
        stance = PolicyStance.INSUFFICIENT_EVIDENCE
    return PolicyStancePrediction(
        stance=stance,
        hawkish_matches=tuple(hawkish),
        dovish_matches=tuple(dovish),
        supporting_evidence=tuple(evidence),
    )
