# Phase 5 regression analysis

## Scope and specification

This checkpoint estimates nested explanatory models on the statement-only event
dataset. It tests whether the available NLP features add explanatory value beyond a
VIX-only baseline. The results are exploratory associations, not causal effects or
out-of-sample predictions.

The baseline model includes the pre-event VIX level, prior 1-day and 5-day VIX changes,
and fixed effects for the project's three historical eras. The enhanced model adds the
financial-sentiment score, hawkish, dovish, and neutral-or-mixed policy shares, and the
number of statements in the event. `insufficient_evidence` is the policy reference
category. Emotion is excluded because Phase 4 did not evaluate non-neutral emotion
classes adequately.

Absolute VIX changes are the primary outcomes. Percentage changes are robustness
outcomes. The 1-day models use HC3 heteroskedasticity-robust standard errors. The 5-day
models use HAC standard errors with five lags because event windows may overlap.

## Results

| Outcome | Events | Baseline adjusted R² | Enhanced adjusted R² | Change | Joint NLP p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute VIX change, 1 day | 242 | 0.086 | 0.138 | 0.052 | 0.155 |
| Absolute VIX change, 5 days | 241 | 0.106 | 0.176 | 0.070 | 0.011 |
| Percentage VIX change, 1 day | 242 | 0.018 | 0.040 | 0.022 | 0.107 |
| Percentage VIX change, 5 days | 241 | 0.023 | 0.058 | 0.035 | <0.001 |

The 1-day models do not show statistically reliable incremental explanatory value from
the NLP block. The full-sample 5-day models produce significant joint tests, but this
result is not robust.

## Fragility and diagnostics

Only five of 242 events have a positive `neutral_or_mixed` policy share. All five occur
between October 2007 and October 2008, and the large positive coefficient on that term
drives the full-sample 5-day result. A post-hoc diagnostic excluding 2007–2009 removes
all variation in that term. In the remaining sample:

| Outcome | Events | Change in adjusted R² | Joint NLP p-value |
| --- | ---: | ---: | ---: |
| Absolute VIX change, 1 day | 213 | 0.026 | 0.205 |
| Absolute VIX change, 5 days | 212 | -0.009 | 0.621 |
| Percentage VIX change, 1 day | 213 | 0.016 | 0.107 |
| Percentage VIX change, 5 days | 212 | 0.002 | 0.259 |

Residuals are strongly non-normal and heavy-tailed. The absolute-change models also
show strong heteroskedasticity. Robust covariance estimates reduce inferential risk but
do not remove outlier sensitivity or omitted-variable bias. BIC worsens when NLP terms
are added in every specification, even where AIC improves.

## Conclusion

The existing NLP features do not provide robust incremental explanatory value for VIX
changes in this dataset. The apparent full-sample 5-day association is concentrated in
a very small set of global-financial-crisis events and should not be interpreted as a
general language effect. This is still a useful negative baseline result: it prevents
the original same-day VIX-level regression from being mistaken for evidence of market
predictability.

Future work may add verified announcement timestamps, broader conventional market
controls, equity returns, a larger policy-label evaluation set, and out-of-sample
testing. Those extensions require new data or a separately reviewed modeling phase.

## Reproduction

Install the analysis dependency and regenerate events before estimating the models:

```powershell
python -m pip install -e ".[analysis]"
python -m fed_policy_intelligence.analysis.events --config config/config.yaml
python -m fed_policy_intelligence.analysis.regression --config config/config.yaml
```

The commands write ignored, reproducible `regression_results.json` and
`regression_coefficients.csv` files under `data/processed`.
