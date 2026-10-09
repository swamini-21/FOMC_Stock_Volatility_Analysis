# Model evaluation plan

## Status

The evaluation machinery and annotation schema are implemented, but no gold labels
or measured model results exist yet. This document intentionally reports no fabricated
scores.

## Sampling and splits

Create a manually reviewed sample stratified by document type and publication period.
Do not select examples based on model confidence. Mark examples used to adjust policy
rules or later LLM prompts as `development`; keep `evaluation` examples untouched until
the method is fixed. Record the final sample size and class distribution.

The versioned schema is `data/annotations/annotation_template.csv`. Source text remains
in the processed document table and is joined by stable `document_id`, avoiding a
second mutable copy in the annotation file.

## Label guidance

### Financial sentiment

Assign `positive`, `negative`, or `neutral` according to the financial condition or
outlook expressed in the passage. Do not translate policy tightening directly into
negative sentiment or policy easing into positive sentiment. Note genuinely mixed or
context-dependent passages in `annotator_notes`.

### Emotion

Choose among anger, disgust, fear, joy, neutral, sadness, and surprise, matching the
configured model's output space. Federal Reserve prose may not express a human emotion
clearly; use neutral where appropriate and document difficult cases. These subjective
labels require special caution because the model was not trained specifically on FOMC
documents.

### Policy stance

- `hawkish`: evidence favors tighter policy or emphasizes upside inflation risk.
- `dovish`: evidence favors easier policy or emphasizes downside employment/growth
  risk.
- `neutral_or_mixed`: meaningful evidence exists in both directions or explicitly
  balances those directions.
- `insufficient_evidence`: the passage does not support a defensible directional
  classification.

Copy a verbatim supporting excerpt. Do not use outside economic knowledge to fill gaps.

## Metrics

For each task, report sample size, class support, confusion matrix, per-class precision,
recall, and F1, plus macro-averaged precision, recall, and F1. The implemented metric
function uses zero—not an exception or an inflated value—when a class has an undefined
precision or recall denominator.

For multiple annotators, report raw agreement and an appropriate chance-corrected
agreement measure. Disagreements should be adjudicated without silently replacing the
original annotations.

## Aggregation comparison

Evaluate `document_mean`, `document_length_weighted`, and `paragraph_mean` on the same
reviewed examples. Compare both final labels and full distributions. Select an
aggregation method based on evidence and qualitative errors, not convenience.

## Required run metadata

Record model identifier and revision when pinned, tokenizer identifier, preprocessing
version, max length, stride, aggregation method, device, dependency versions, seed,
sample identifiers, and evaluation date. Preserve failures and exclusions with reasons.

## Failure analysis

Review at least tokenizer/chunk boundary effects, negation, mixed policy signals,
quoted language, historical references, boilerplate, long transcripts, domain mismatch,
and low-confidence distributions. Topic methods additionally require representative
documents, keyword inspection, stability across seeds/settings, and coherence only as
one diagnostic rather than a sufficient selection criterion.
