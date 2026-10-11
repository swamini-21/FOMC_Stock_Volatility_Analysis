"""Validate provider output and preserve complete generation provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from fed_policy_intelligence.llm.grounding import audit_evidence
from fed_policy_intelligence.llm.prompt import PROMPT_VERSION
from fed_policy_intelligence.llm.provider import ProviderOutput, StructuredPolicyProvider
from fed_policy_intelligence.llm.schema import SCHEMA_VERSION, PolicyIntelligence


@dataclass(frozen=True)
class ExtractionRecord:
    document_id: str
    source_sha256: str
    provider: str
    model_id: str
    prompt_version: str
    schema_version: str
    generated_at: str
    status: str
    error: str
    evidence_supported: bool
    unsupported_evidence: tuple[str, ...]
    usage: dict[str, Any]
    result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyIntelligenceExtractor:
    def __init__(
        self,
        provider: StructuredPolicyProvider,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.clock = clock or (lambda: datetime.now(UTC))

    def extract(self, *, document_id: str, source_text: str) -> ExtractionRecord:
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        generated_at = self.clock().astimezone(UTC).isoformat()
        try:
            provider_result = self.provider.extract(source_text)
            if isinstance(provider_result, ProviderOutput):
                raw_result = provider_result.result
                usage = provider_result.usage
            else:
                raw_result = provider_result
                usage = {}
            result = PolicyIntelligence.from_mapping(raw_result)
            audit = audit_evidence(result, source_text)
            status = "success" if audit.supported else "invalid_evidence"
            return ExtractionRecord(
                document_id=document_id,
                source_sha256=source_hash,
                provider=self.provider.provider_name,
                model_id=self.provider.model_id,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                generated_at=generated_at,
                status=status,
                error="",
                evidence_supported=audit.supported,
                unsupported_evidence=audit.unsupported_quotes,
                usage=usage,
                result=result.to_dict(),
            )
        except Exception as exc:
            return ExtractionRecord(
                document_id=document_id,
                source_sha256=source_hash,
                provider=self.provider.provider_name,
                model_id=self.provider.model_id,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                generated_at=generated_at,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                evidence_supported=False,
                unsupported_evidence=(),
                usage={},
                result=None,
            )
