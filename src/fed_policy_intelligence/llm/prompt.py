"""Versioned instructions shared by future live structured-output providers."""

from __future__ import annotations

PROMPT_VERSION = "1.0"
SYSTEM_INSTRUCTIONS = """You analyze only the supplied Federal Reserve passage.
Return an object matching the provided JSON schema. Do not introduce external facts.
Classify policy stance as hawkish, dovish, neutral_or_mixed, or insufficient_evidence.
Use null or empty arrays when the source does not support a field. Every supporting
quote must be verbatim from the passage. Mark each claim as explicit or inference,
identify conflicting language in uncertainties, and abstain when evidence is inadequate.
Do not return a confidence probability."""
