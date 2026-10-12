# Research methodology

## Objective and research questions

This project is an observational study of Federal Reserve communication and market
reactions. It asks whether policy stance extracted from FOMC statements is associated
with subsequent changes in market volatility, equity prices, and Treasury yields after
accounting for observable pre-event market conditions. It also tests whether general
financial sentiment or a selected emotion measure adds explanatory information beyond
policy stance.

The estimates are descriptive associations. They are neither forecasts nor causal
effects, and the application is not an investment-advice product.

## Data

The local corpus contains 773 Federal Reserve documents: 295 statements and 478
transcripts. Transcripts are retained for document research but excluded from the
market study because their later publication could introduce look-ahead bias. The 295
eligible statements collapse to 242 distinct announcement dates.

Policy-stance supervision comes from the pinned Georgia Tech
`gtfintechlab/fomc_communication` dataset. Its expert-labelled sentences are divided
into 1,587 fitting rows, a seeded stratified 397-row validation split, and an untouched
496-row published test split. The labels are dovish, hawkish, and neutral. The dataset
and associated model are CC BY-NC 4.0 and are loaded through the Hugging Face cache;
no copy is committed to this repository.

Market inputs are the existing local VIX series (`VIXCLS`), the S&P 500 index
(`^GSPC`) downloaded through `yfinance`, and 2-year and 10-year constant-maturity
Treasury yields (`DGS2` and `DGS10`) from FRED. Download provenance, normalized-file
hashes, source identifiers, and retrieval times are stored with generated outputs.

## Document processing and language measurement

Ingestion preserves raw text, stable document identifiers, canonical source URLs, and
content hashes. Conservative cleaning creates a separate versioned field. Long texts
are processed in tokenizer-safe overlapping chunks; each model uses its paired
tokenizer, complete class probabilities are retained, and model/tokenizer revisions,
aggregation rules, run time, and device are recorded.

The primary language measure is a locally fine-tuned RoBERTa-base classifier trained
on the expert FOMC labels. On the untouched external test split it achieved 65.7%
accuracy and 0.639 macro-F1 (class F1: dovish 0.600, hawkish 0.590, neutral 0.726).
It is applied sentence by sentence to the 295 statements, then aggregated into
continuous hawkish, dovish, and neutral sentence shares. These shares preserve more
information than a document argmax, which is neutral for 274 statements.

FinBERT's positive-minus-negative score is a secondary financial-sentiment measure.
The Hartmann emotion model is exploratory; only its prespecified fear probability is
used, because the reviewed annotation set did not establish performance across the
full emotion label space. Topic discovery remains descriptive. The rule-based stance
baseline and the bounded Gemini experiment are retained as audit evidence but are not
used as the primary policy measurement.

## Event construction and outcomes

Because verified publication timestamps are unavailable, each statement date is
aligned to the first usable market observation strictly after that date. This avoids
calling an unknown same-day close a post-announcement observation, while accepting
that part of the true reaction may be omitted.

Outcomes are signed 1-session and 5-session VIX point and percentage changes, S&P 500
returns, and 2-year and 10-year Treasury-yield changes in basis points. Corresponding
pre-event market changes are constructed as controls. Same-date statements are
aggregated without silently discarding documents.

## Statistical design

For each outcome, nested models use the same complete-case sample:

1. Baseline: pre-event VIX, prior 1-day and 5-day VIX changes, prior 1-day and 5-day
   S&P 500 returns, prior 1-day 2-year and 10-year yield changes, statement count, and
   historical-era fixed effects.
2. Policy stance: baseline plus hawkish and dovish shares, with neutral as the
   compositional reference.
3. Sentiment: stance model plus the FinBERT sentiment score.
4. Emotion: sentiment model plus fear probability.

One-day models use HC3 heteroskedasticity-robust standard errors. Five-day models use
HAC standard errors with five lags because event windows may overlap. The workflow
reports descriptive statistics, Pearson and Spearman correlations, stance-group
comparisons, nested adjusted R-squared changes, joint block tests, confidence
intervals, VIFs, residual diagnostics, Cook's distance, and a 2007–2009 exclusion
sensitivity. Holm correction controls the family of ten market-outcome tests.

## Main findings

No language block survives both Holm correction and the crisis-exclusion sensitivity.
The primary 1-day VIX stance-block result is nominal only (raw p = 0.084; Holm p =
0.759). The 5-day 2-year-yield stance block is also nominal (raw p = 0.023; Holm p =
0.225) and weakens to p = 0.101 outside 2007–2009. Sentiment and fear add no robust
incremental explanatory value. These results support a cautious negative conclusion:
the improved language measurements show plausible directions but not stable evidence
of a general market-reaction relationship in this sample.

## Limitations

- Date-only alignment cannot isolate intraday announcement reactions.
- The domain classifier is materially better than the rule baseline but remains a
  moderate, error-prone proxy for expert judgement.
- Dominant-stance groups are imbalanced, residuals are heavy-tailed, and influential
  crisis observations remain important.
- The regressions omit a direct market-expectations or monetary-policy-surprise
  measure; unobserved confounding prevents causal interpretation.
- Yahoo Finance is an unofficial convenience source and may change.
- Multiple outcomes and model blocks reduce power after multiplicity correction.

## Reproducibility

Exact commands and dependency profiles are in the project [README](../README.md).
Detailed contracts and evidence are in
[the expert-dataset note](expert_fomc_dataset.md),
[event-study documentation](event_study.md),
[research results](research_analysis.md), and
[market-data documentation](market_data.md). Generated data, model weights, caches,
and credentials are intentionally excluded from Git.
