# FOMC Stock Market Volatility Analysis

This project is being rebuilt from an exploratory notebook into a reproducible Federal Reserve policy-intelligence and market-reaction application. Work proceeds in reviewed phases; the current repository contains the audit, architecture scaffold, and validated local-data pipeline.

## Problem statements

1. **Reliable policy-signal extraction:** How can we build a reproducible NLP pipeline that preserves distinct Federal Reserve documents and reliably extracts sentiment, emotion, policy stance, and topics from long-form communications?

2. **Incremental market-reaction analysis:** Do language-derived features from Federal Reserve communications provide incremental explanatory or predictive value for subsequent VIX changes or equity returns beyond conventional market variables?

3. **Grounded policy intelligence:** Can an LLM extract structured, evidence-backed policy signals—such as hawkish or dovish stance, inflation outlook, labor-market outlook, and key risks—more reliably and interpretably than transparent transformer or rule-based baselines?

## Current status

- Phase 1: notebook and data audit completed in [`docs/audit_report.md`](docs/audit_report.md).
- Phase 2: architecture and foundational scaffold completed in [`docs/architecture.md`](docs/architecture.md).
- Phase 3: local document and VIX ingestion, validation, identity, and safe preprocessing implemented in [`docs/data_pipeline.md`](docs/data_pipeline.md).
- Model inference, quantitative analysis, LLM integration, and the Streamlit application are not implemented yet.

## Implemented structure

```text
config/config.yaml
docs/audit_report.md
docs/architecture.md
src/fed_policy_intelligence/
├── __init__.py
├── data/
│   ├── identity.py
│   ├── ingestion.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── validation.py
├── logging_config.py
└── settings.py
tests/
├── fixtures/
└── test_*.py
.env.example
.gitignore
pyproject.toml
```

## Development setup

Python 3.11 or 3.12 is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

The default configuration uses CPU execution and a mock LLM provider. The original notebook and local datasets remain unchanged.

## Data pipeline

Run validation and create ignored local processed outputs:

```bash
python -m fed_policy_intelligence.data.pipeline --config config/config.yaml
```

