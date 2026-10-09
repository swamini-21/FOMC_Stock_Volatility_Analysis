# Phase 4 NLP pipeline

## Scope and current status

Phase 4 implements the reusable inference and evaluation code for financial
sentiment, emotion, transparent policy stance, and topic discovery. Unit tests use
injected deterministic backends, so ordinary tests do not download model weights or
require a GPU. Real model execution is available through an optional dependency group.
A five-document GPU smoke test completed successfully on the verified Windows/NVIDIA
environment; this confirms execution, not model quality.

## Transformer inference

The shared classifier in `nlp/transformer.py` loads a tokenizer and model from the
same checkpoint. It rejects mismatched checkpoint identifiers, which directly guards
against the original notebook's use of FinBERT token IDs with the DistilRoBERTa
emotion model.

Long text is tokenized without special tokens, divided into overlapping windows, and
then wrapped with the selected tokenizer's own special tokens. The configured
`max_length` includes special tokens, and `stride` means the number of content tokens
repeated between consecutive windows.

Three aggregation strategies are available:

- `document_mean`: equal weight for each chunk;
- `document_length_weighted`: chunk probabilities weighted by non-special-token
  count;
- `paragraph_mean`: length-weighted chunks within each paragraph, followed by equal
  weighting across paragraphs.

Probabilities—not logits—are aggregated. All returned distributions must be finite,
bounded between zero and one, and normalized within numerical tolerance.

## Financial sentiment

The baseline model is `ProsusAI/finbert`. The output retains positive, negative, and
neutral probabilities, plus the documented derived score:

```text
sentiment_score = P(positive) - P(negative)
```

This score represents financial sentiment. It is not treated as hawkishness,
dovishness, or expected market direction.

## Emotion

The baseline is `j-hartmann/emotion-english-distilroberta-base`. Its expected label
space is anger, disgust, fear, joy, neutral, sadness, and surprise. Initialization
fails if the loaded model exposes a different label set. The complete probability
distribution is retained.

These labels were verified against the model card, but the training domains include
social and conversational text. Their validity for central-bank communications is an
open empirical question requiring manual evaluation.

## Transparent policy-stance baseline

The rule baseline defines:

- `hawkish`: explicit tightening, restrictive-policy, or upside-inflation-risk
  language without nearby negation;
- `dovish`: explicit easing, accommodative-policy, or downside employment/growth
  risk language without nearby negation;
- `neutral_or_mixed`: both hawkish and dovish evidence appears;
- `insufficient_evidence`: no configured phrase provides evidence.

The output preserves matched phrases and supporting sentences. This is deliberately a
high-precision, low-coverage baseline rather than a general semantic classifier. It
does not infer stance from sentiment. Its phrase coverage, negation window, and lack
of contextual reasoning are documented limitations to evaluate before interpretation.

## Topic discovery

`nlp/topics.py` provides a deterministic count-vector LDA baseline and an optional
BERTopic comparator. Both retain numeric topic assignments and keywords separately
from future human-readable labels. No topic count, coherence result, stability result,
or winning method has been selected yet. Those require an experiment on representative
documents and human review.

## Running real inference

For CPU-only environments, install the NLP dependency group normally:

```bash
python -m pip install -e ".[dev,nlp]"
```

For the tested Windows/NVIDIA environment, install the CUDA 12.8 build of PyTorch
2.8.0 from the official PyTorch wheel index before installing the project extras.
PyTorch is constrained to the 2.8 release family because later builds produced a
`c10.dll` initialization failure on this host. Detailed instructions and verification
commands are in `docs/windows_gpu_setup.md`.

Set `FPI_DEVICE=cuda` and run a small GPU smoke test before processing the complete
corpus:

```powershell
$env:FPI_DEVICE = "cuda"
& "$env:USERPROFILE\fpi-gpu\Scripts\python.exe" -m fed_policy_intelligence.nlp.pipeline --config config/config.yaml --limit 5
```

The command reads `data/processed/fed_documents.csv` and atomically writes the ignored
file `data/processed/nlp_predictions.csv`. Increase the limit only after inspecting
runtime, GPU memory, and output validity. In Colab, use its supported PyTorch runtime
and set `FPI_DEVICE=cuda` only after `torch.cuda.is_available()` returns `True`.

Model weights, prediction caches, and processed outputs remain excluded from Git.

## What is not yet claimed

- The five-document smoke test produced five successful FinBERT and emotion prediction
  rows with no inference errors. The full 773-document corpus has not been processed.
- No manually reviewed evaluation sample exists yet.
- No classification score, topic coherence, model superiority, or market relationship
  is claimed.
- BERTopic is an optional comparison path and has not been installed or run locally.
