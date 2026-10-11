import json
from types import SimpleNamespace

from fed_policy_intelligence.llm.gemini import GeminiPolicyProvider
from fed_policy_intelligence.llm.provider import ProviderOutput


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "policy_stance": "hawkish",
        "inflation_outlook": "Inflation remains elevated.",
        "labor_market_outlook": None,
        "economic_growth_outlook": None,
        "key_risks": [],
        "policy_rationale": [],
        "supporting_evidence": [
            {
                "field": "policy_stance",
                "quote": "Inflation remains elevated.",
                "claim_type": "explicit",
                "explanation": "The document explicitly describes elevated inflation.",
            }
        ],
        "uncertainties": [],
        "abstained": False,
    }


class FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_gemini_provider_requests_schema_and_returns_usage() -> None:
    usage = SimpleNamespace(model_dump=lambda **_: {"prompt_token_count": 12})
    models = FakeModels(
        [SimpleNamespace(text=json.dumps(_valid_payload()), usage_metadata=usage)]
    )
    provider = GeminiPolicyProvider(
        "gemini-test", client=SimpleNamespace(models=models), sleep=lambda _: None
    )

    output = provider.extract("Inflation remains elevated.")

    assert isinstance(output, ProviderOutput)
    assert output.result["policy_stance"] == "hawkish"
    assert output.usage == {"prompt_token_count": 12}
    config = models.calls[0]["config"]
    assert isinstance(config, dict)
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["additionalProperties"] is False


def test_gemini_provider_retries_bounded_failures() -> None:
    models = FakeModels(
        [
            RuntimeError("temporary"),
            SimpleNamespace(text=json.dumps(_valid_payload()), usage_metadata=None),
        ]
    )
    delays: list[float] = []
    provider = GeminiPolicyProvider(
        "gemini-test",
        client=SimpleNamespace(models=models),
        max_attempts=2,
        sleep=delays.append,
    )

    output = provider.extract("Inflation remains elevated.")

    assert output.result["policy_stance"] == "hawkish"
    assert len(models.calls) == 2
    assert delays == [1]
