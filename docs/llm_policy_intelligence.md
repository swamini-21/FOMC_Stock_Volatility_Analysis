# Mock-first structured policy intelligence

## Purpose

This phase establishes a provider-neutral, evidence-grounded contract. The deterministic
mock verifies application behavior; it is not an LLM, is not evaluated as a model, and
must not be presented as evidence that structured extraction is accurate. Gemini is the
first live provider, while the mock remains the default to prevent accidental API usage.

## Versioned schema

Schema version `1.0` requires:

- policy stance: hawkish, dovish, neutral-or-mixed, or insufficient evidence;
- inflation, labor-market, and economic-growth outlooks, each nullable;
- lists of key risks and policy rationale;
- supporting evidence containing a field, verbatim quote, explicit/inference marker,
  and explanation;
- uncertainties and an explicit abstention flag.

Unexpected fields are rejected. Generated confidence values are not accepted or
treated as calibrated probabilities. A non-abstained result requires supporting
evidence, while a fully abstained result must use `insufficient_evidence`. The same
contract is exposed as JSON Schema for a future provider's native structured output.

## Grounding and provenance

Every supporting quote is checked as an exact substring of the supplied document.
Unsupported quotes produce `invalid_evidence` rather than being silently accepted.
Malformed output and provider failures produce row-level error records.

Each record stores the document ID, source SHA-256, provider, model identifier, prompt
version, schema version, UTC generation time, validation status, evidence audit, token
usage when available, and nested structured result. Cached results are invalidated when source text, model,
prompt, or schema changes. Error records are retried rather than reused indefinitely.

## Offline command

The default configuration remains `provider: mock` and requires no API key:

```powershell
python -m fed_policy_intelligence.llm.pipeline --config config/config.yaml
```

The command processes statement documents and writes ignored JSON Lines output to
`data/processed/llm_policy_intelligence.jsonl`. A repeated run reuses matching valid
records. The verified run produced 295 successful rows, and the immediate second run
reused all 295 rows from cache.

## Gemini live provider

Install the optional SDK without changing the tested PyTorch installation:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m pip install -e ".[dev,llm]"
```

Gemini credentials are read only from `GEMINI_API_KEY` or `GOOGLE_API_KEY`. The API key
must never be written to YAML, code, generated output, or Git. The integration uses
native JSON Schema output, temperature zero, a request timeout, bounded retries, local
schema validation, exact-quote auditing, and usage metadata capture.

The versioned YAML defaults to `mock`. Enable Gemini only for an intentional run:

```powershell
$env:FPI_LLM_PROVIDER = "gemini"
$env:FPI_LLM_MODEL = "gemini-3.5-flash-lite"
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.llm.pipeline `
  --config config/config.yaml `
  --limit 1 `
  --output data/processed/gemini_policy_intelligence_smoke.jsonl
```

The live-provider evaluation must report schema-validity rate, stance macro-F1 against
the expert FOMC labels, evidence-support rate, unsupported-claim frequency, abstention
behavior, token usage, and qualitative failure cases. OpenAI and Anthropic remain
accepted configuration names but fail clearly because their adapters are not implemented.

The first bounded Gemini evaluation is reported in
[`gemini_bounded_evaluation.md`](gemini_bounded_evaluation.md). Low expert-sentence
coverage and document quote-grounding failures currently block a full-corpus run.
