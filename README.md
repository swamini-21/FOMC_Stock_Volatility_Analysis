# FOMC Stock Market Volatility Analysis

This project is being rebuilt from an exploratory notebook into a reproducible Federal Reserve policy-intelligence and market-reaction application. Work proceeds in reviewed phases; the current repository contains the audit, architecture scaffold, validated local-data pipeline, and testable Phase 4 NLP components.

## Problem statements

1. **Reliable policy-signal extraction:** How can we build a reproducible NLP pipeline that preserves distinct Federal Reserve documents and reliably extracts sentiment, emotion, policy stance, and topics from long-form communications?

2. **Incremental market-reaction analysis:** Do language-derived features from Federal Reserve communications provide incremental explanatory or predictive value for subsequent VIX changes or equity returns beyond conventional market variables?

3. **Grounded policy intelligence:** Can an LLM extract structured, evidence-backed policy signals—such as hawkish or dovish stance, inflation outlook, labor-market outlook, and key risks—more reliably and interpretably than transparent transformer or rule-based baselines?

## Current status

- Phase 1: notebook and data audit completed in [`docs/audit_report.md`](docs/audit_report.md).
- Phase 2: architecture and foundational scaffold completed in [`docs/architecture.md`](docs/architecture.md).
- Phase 3: local document and VIX ingestion, validation, identity, and safe preprocessing implemented in [`docs/data_pipeline.md`](docs/data_pipeline.md).
- Phase 4: tokenizer-safe inference, aggregation, policy baseline, topic interfaces, and evaluation scaffolding implemented in [`docs/nlp_pipeline.md`](docs/nlp_pipeline.md). A five-document GPU smoke test succeeded; manually reviewed model evaluation remains pending.
- Quantitative analysis, LLM integration, and the Streamlit application are not implemented yet.

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
├── evaluation/
│   └── classification.py
├── nlp/
│   ├── chunking.py
│   ├── emotion.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── prediction.py
│   ├── sentiment.py
│   ├── topics.py
│   └── transformer.py
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

Install the optional real-model dependencies only when running transformer inference:

```bash
python -m pip install -e ".[dev,nlp]"
```

### Windows NVIDIA GPU setup

The tested Windows configuration uses an NVIDIA GeForce RTX 4060 Laptop GPU,
PyTorch `2.8.0+cu128`, and Python 3.12. Use a short virtual-environment path to avoid
Windows filename-length failures in PyTorch's package files. Install the CUDA-enabled
PyTorch wheel before installing the project extras:

```powershell
$FpiGpuEnv = Join-Path $env:USERPROFILE "fpi-gpu"
& "$env:USERPROFILE\anaconda3\python.exe" -m venv $FpiGpuEnv
& "$FpiGpuEnv\Scripts\python.exe" -m pip install --upgrade pip
& "$FpiGpuEnv\Scripts\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& "$FpiGpuEnv\Scripts\python.exe" -m pip install -e ".[dev,nlp]"
```

Verify CUDA, then select it for the current PowerShell session:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
$env:FPI_DEVICE = "cuda"
```

The expected verification includes `2.8.0+cu128`, `True`, and the NVIDIA GPU name.
The PyTorch wheel contains the required CUDA runtime; a matching local CUDA Toolkit is
not required. See [`docs/windows_gpu_setup.md`](docs/windows_gpu_setup.md) for the
tested versions and troubleshooting notes.

## Data pipeline

Run validation and create ignored local processed outputs:

```bash
python -m fed_policy_intelligence.data.pipeline --config config/config.yaml
```

## NLP pipeline

After installing the optional NLP dependencies, start with a bounded run. This
downloads model weights on first use and writes predictions only to the ignored
processed-data directory:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.nlp.pipeline --config config/config.yaml --limit 5
```

