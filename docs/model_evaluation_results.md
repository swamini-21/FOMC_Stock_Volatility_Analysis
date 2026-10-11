# Phase 4 model evaluation results

## Evaluation run

The reviewed evaluation split contains 21 documents. Predictions came from the
completed 773-document NLP run using CUDA, document-length-weighted aggregation,
FinBERT for financial sentiment, DistilRoBERTa for emotion, and the version 1 phrase
rules for policy stance. The evaluation was run on October 11, 2026.

| Task | Accuracy | Macro F1 | Main result |
| --- | ---: | ---: | --- |
| Financial sentiment | 52.4% | 0.449 | The model predicted neutral for 17 of 21 documents. |
| Emotion | 95.2% | 0.139 | All 21 gold labels are neutral, so non-neutral performance is not measured. |
| Policy stance | 33.3% | 0.299 | No hawkish example was classified correctly. |

Macro F1 uses each model's fixed output label space. The emotion macro F1 is low
because six classes have no gold support, while neutral-class F1 is 0.976. The high
emotion accuracy therefore does not establish broad emotion-classification quality.

## Error patterns

Financial sentiment had 10 mismatches. Five positive and four negative documents were
classified as neutral. This suggests that full-document aggregation dilutes directional
language, especially in long transcripts containing competing views and boilerplate.

Policy stance had 14 mismatches. Five hawkish documents were classified as
`neutral_or_mixed`, and five dovish documents were classified as
`insufficient_evidence`. The phrase rules made no hawkish predictions on the reviewed
set. Their current vocabulary and negation handling provide insufficient coverage for
historical wording and transcript discussion.

Emotion had one mismatch: a neutral document was classified as sadness. Because the
gold sample contains only neutral emotion labels, this task needs a larger deliberately
diagnostic sample before model selection or strong claims.

## Interpretation and closure

These are baseline results, not production-quality estimates. The sample is small, and
emotion class coverage is incomplete. The held-out evaluation rows must not be used to
tune phrase lists, thresholds, aggregation, or prompts. Any improvements should be
developed using the seven development rows or an expanded development set, frozen, and
then assessed on a new untouched evaluation sample.

Phase 4 is closed with these results as the reproducible baseline. If model improvement
is revisited, the next controlled experiment should compare the three implemented
aggregation methods on development documents. Policy-rule changes must be tested on
development examples, with matched evidence retained for every rule decision. LLM
comparison remains a later phase and should use the same frozen label definitions.

Phase 5 may use these language features for market-reaction analysis, provided its
interpretation preserves the documented measurement limitations and does not treat the
predicted labels as error-free observations.
