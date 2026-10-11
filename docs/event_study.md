# Phase 5 event construction

## Scope

The first Phase 5 checkpoint constructs a reproducible event-level dataset from the
existing processed Federal Reserve documents, Phase 4 NLP predictions, and local VIX
observations. It does not estimate regressions or claim a causal market effect.

## Event definition and leakage controls

The corpus provides publication dates but no verified publication times. Each event is
therefore aligned conservatively:

- the pre-event VIX is the last usable observation strictly before the publication
  date;
- the 1-day outcome uses the first usable VIX session strictly after the publication
  date;
- the 5-day outcome uses the fifth usable VIX session strictly after the publication
  date;
- same-date documents are aggregated into one event so a market outcome is not copied
  across multiple document rows.

Only `statement` documents are eligible for the market-event dataset. Transcripts are
excluded because the corpus associates them with their meeting dates even though the
text was released later. Using transcript language at the meeting date would introduce
look-ahead leakage. Transcripts remain available for descriptive NLP analysis.

## Event features

Each event retains document IDs and counts, statement counts, NLP success/error counts,
mean financial-sentiment score and probabilities, mean emotion probabilities, and the
share of documents assigned to each policy-stance label. Event-level aggregation gives
each eligible document equal weight.

VIX outcomes include the pre-event value, post-event trading dates and values, absolute
changes, percentage changes, usability flags, and explicit exclusion reasons. The
source remains the undocumented local VIX file described in the Phase 3 data report.

## Running the event builder

```powershell
python -m fed_policy_intelligence.analysis.events --config config/config.yaml
```

The command writes ignored, reproducible artifacts:

- `data/processed/fed_vix_events.csv`
- `data/processed/event_construction_report.json`

## Verified local run

| Measure | Result |
| --- | ---: |
| Input documents | 773 |
| Eligible statement documents | 295 |
| Excluded transcript documents | 478 |
| Statement event dates | 242 |
| Events containing multiple statements | 47 |
| Events usable for the 1-day outcome | 242 |
| Events usable for the 5-day outcome | 241 |
| Events excluded from the 5-day outcome | 1 |

Across usable events, the mean absolute VIX change was -0.311 for the 1-day window and
-0.453 for the 5-day window. These descriptive averages are unadjusted and should not
be interpreted as effects of Federal Reserve language.

## Next checkpoint

The next Phase 5 step is to specify baseline and NLP-enhanced statistical models before
estimating them. The specification must define predictors, transformations, standard
errors, diagnostics, and how the Phase 4 measurement limitations will be represented.
