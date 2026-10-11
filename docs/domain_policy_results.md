# Expert-labelled domain policy results

## Outcome

The project now replaces manual project annotation for monetary-policy stance with
the public Georgia Tech FOMC Communication dataset. Because the authors' published
FOMC-RoBERTa weights are gated, the reproducible default fine-tunes public
RoBERTa-base on the pinned expert training split.

The training design uses 1,587 sentences for fitting and a seeded, stratified set of
397 sentences for validation. The published 496-sentence test split is untouched
until final evaluation.

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Training-split validation | 67.5% | 0.657 |
| Published external test split | 65.7% | 0.639 |

External-test class F1 scores are 0.600 for dovish, 0.590 for hawkish, and 0.726 for
neutral. This is a material improvement over the Phase 4 policy-rule macro-F1 of
0.299, but it remains a moderate classifier rather than an expert substitute.

## Existing statement corpus

The validated classifier was applied sentence by sentence to all 295 statement
documents used by the market study. All 295 completed successfully. Mean sentence
shares across documents are 13.5% hawkish, 19.2% dovish, and 67.2% neutral. The
pipeline preserves these continuous shares instead of relying only on the document
argmax, which is neutral for 274 of 295 statements.

The 478 transcripts were not processed in this run because the market-event pipeline
excludes them to prevent publication-date look-ahead. The same command can process all
773 documents later if transcript-level research is added.

## Updated VIX analysis

The event dataset contains 242 unique statement dates. The enhanced regressions now
use domain-model hawkish and dovish sentence shares, with neutral as the reference,
instead of the Phase 4 rule labels.

| Outcome | Baseline adjusted R² | Enhanced adjusted R² | Change | Joint NLP p-value |
|---|---:|---:|---:|---:|
| Absolute VIX, 1 day | 0.0860 | 0.1109 | +0.0249 | 0.0402 |
| Absolute VIX, 5 day | 0.1055 | 0.1010 | -0.0045 | 0.2531 |
| Percentage VIX, 1 day | 0.0180 | 0.0334 | +0.0154 | 0.0693 |
| Percentage VIX, 5 day | 0.0234 | 0.0171 | -0.0062 | 0.5587 |

The one-day absolute result is nominally significant in the full sample, but its
joint p-value rises to 0.0966 after excluding 2007–2009. The five-day models do not
improve. The appropriate conclusion is therefore that expert-supervised stance
measurement improves the NLP layer and shows a possible short-horizon association,
but does not yet establish a robust market-reaction effect. Market-expectation data,
precise timestamps, and additional asset outcomes remain the next substantive data
improvements.
