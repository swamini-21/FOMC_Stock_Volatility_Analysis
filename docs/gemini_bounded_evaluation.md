# Bounded Gemini policy evaluation

## Decision

Gemini should not replace the expert-supervised local FOMC-RoBERTa stance classifier.
The bounded evaluation supports retaining RoBERTa for sentence-level stance and treating
Gemini structured extraction as an experimental enrichment layer. A full 295-statement
Gemini run is deferred until quote grounding and abstention behavior improve.

## Design

The run used `gemini-3.5-flash-lite`, a fixed seed of 42, and two deterministic samples:

- 30 expert-labelled test sentences, balanced across dovish, hawkish, and neutral;
- 30 statement documents evenly spaced over publication date after statement filtering.

The published expert test split was not used to tune the prompt. Every response passed
through the versioned local schema and exact-substring quote audit. Raw outputs and the
summary report remain ignored under `data/processed`.

Newer Flash endpoints returned temporary HTTP 503 high-demand responses during the
initial attempt. The completed run therefore used the lower-demand pinned Flash Lite
model with six-second request pacing and bounded exponential retry backoff.

## Expert-labelled sentence results

| Measure | Gemini | Local FOMC-RoBERTa on same 30 rows |
|---|---:|---:|
| Coverage | 26.7% | 100% |
| Accuracy over all rows | 20.0% | 73.3% |
| Accuracy when Gemini did not abstain | 75.0% | — |
| Macro-F1 when Gemini did not abstain | 0.583 | 0.735 overall |
| Structural schema validity | 100% | — |
| Exact-quote support | 96.7% | — |

Gemini returned `insufficient_evidence` for 22 of 30 sentences. Four of those records
were not marked as abstained, exposing a semantic inconsistency that the structural
schema alone does not prevent. The high selective accuracy is therefore not sufficient
to use Gemini as the primary stance classifier because coverage is too low.

## Statement-document results

| Measure | Result |
|---|---:|
| Documents | 30 |
| Structural schema validity | 100% |
| Exact-quote support | 76.7% |
| Fully valid records | 23 |
| Records with unsupported quotations | 7 |
| Unsupported quotations | 11 |
| Abstention rate | 0% |
| Total tokens reported | 42,673 |

Several unsupported quotations contained ellipses, altered punctuation, normalized
hyphens, or character-encoding differences. These failures show that native structured
JSON guarantees shape, not factual or verbatim grounding. The local quote audit must
remain mandatory.

## Next gate

Before scaling Gemini extraction, revise the extraction design so quoted evidence is
copied as an exact contiguous substring and enforce the relationship between
`insufficient_evidence` and `abstained`. Re-run this same frozen sample from cache or
with a new prompt version, and require materially higher evidence support without
reducing appropriate abstention. RoBERTa remains the stance source for market analysis.
