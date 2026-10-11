import json
from pathlib import Path

import pandas as pd

from fed_policy_intelligence.llm.extractor import PolicyIntelligenceExtractor
from fed_policy_intelligence.llm.pipeline import _write_jsonl_atomic, run_extraction
from fed_policy_intelligence.llm.provider import MockPolicyProvider


class CountingMockProvider(MockPolicyProvider):
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source_text: str) -> dict[str, object]:
        self.calls += 1
        return super().extract(source_text)


def test_extraction_cache_reuses_matching_source_and_versions() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["doc-1"],
            "cleaned_text": ["Inflation remains elevated."],
        }
    )
    provider = CountingMockProvider()
    extractor = PolicyIntelligenceExtractor(provider)
    cache_path = Path(__file__).parent / "fixtures" / ".llm_cache_test.jsonl"
    try:
        first, first_reused = run_extraction(
            documents, extractor=extractor, cache_path=cache_path
        )
        _write_jsonl_atomic(first, cache_path)
        second, second_reused = run_extraction(
            documents, extractor=extractor, cache_path=cache_path
        )

        assert first_reused == 0
        assert second_reused == 1
        assert provider.calls == 1
        assert second[0]["document_id"] == "doc-1"
        assert json.loads(cache_path.read_text(encoding="utf-8"))["status"] == "success"
    finally:
        cache_path.unlink(missing_ok=True)
