# Explanatory statistical analysis

## Research objective

This analysis asks whether language in FOMC statements is associated with subsequent
changes in the VIX, S&P 500, and Treasury yields after accounting for observable
pre-event market conditions. It is an explanatory, observational study. The estimates
are not forecasts and should not be interpreted as causal effects.

The primary outcome is the signed one-session VIX point change. Five-session VIX
changes and percentage changes are robustness outcomes. S&P 500 returns and 2-year
Treasury-yield changes are secondary market outcomes; 10-year-yield changes are
exploratory. All event windows begin on the first usable observation strictly after
the statement date.

## Prespecified model hierarchy

The same complete-case sample is used throughout each outcome's nested hierarchy:

1. **Baseline:** pre-event VIX; prior VIX changes; prior S&P 500 returns; prior 2-year
   and 10-year yield changes; statement count; and historical-era fixed effects.
2. **Policy stance:** baseline plus FOMC-RoBERTa hawkish and dovish sentence shares;
   neutral is the reference composition.
3. **Sentiment:** stance model plus FinBERT sentiment score.
4. **Emotion:** sentiment model plus the preselected fear probability.

The stance variables are the primary language measures because they come from the
expert-supervised FOMC classifier. Sentiment and fear are secondary, exploratory
measures. One-day models use HC3 heteroskedasticity-robust inference. Five-day models
use HAC inference with five lags. Joint block tests are corrected across the four
outcomes using Holm's method. The richer complete-case specification retains all 242
one-day events and 241 of 242 five-day VIX events.

## Results

| Outcome | Events | Stance Δ adjusted R² | Stance p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| VIX point change, 1 day | 242 | 0.011 | 0.084 | 0.759 |
| VIX point change, 5 days | 241 | -0.002 | 0.289 | 1.000 |
| VIX percentage change, 1 day | 242 | 0.008 | 0.101 | 0.799 |
| VIX percentage change, 5 days | 241 | -0.008 | 0.720 | 1.000 |
| S&P 500 return, 1 day | 242 | 0.007 | 0.100 | 0.799 |
| S&P 500 return, 5 days | 242 | 0.003 | 0.131 | 0.799 |
| 2-year yield change, 1 day | 242 | 0.006 | 0.286 | 1.000 |
| 2-year yield change, 5 days | 242 | 0.022 | 0.023 | 0.225 |
| 10-year yield change, 1 day | 242 | -0.008 | 0.964 | 1.000 |
| 10-year yield change, 5 days | 242 | -0.005 | 0.413 | 1.000 |

Adding cross-market controls weakens the primary stance-block result from its earlier
unadjusted p-value of 0.034 to 0.084. In the richer model, a 0.10 increase in dovish
share corresponds to approximately 0.34 additional VIX points (HC3 coefficient 3.35;
95% CI 0.26 to 6.44; individual p = 0.034). The full stance block does not pass its
joint test at 5% and is far from significant after adjustment.

The five-day 2-year-yield model produces a nominal stance-block association. A 0.10
increase in hawkish share corresponds to approximately 1.91 basis points in the
five-day yield change (HAC coefficient 19.09; 95% CI 2.79 to 35.39; individual
p = 0.022). The block fails Holm correction and is not robust to crisis exclusion.

Neither FinBERT sentiment nor fear probability produces a Holm-significant incremental
result for any outcome. No Pearson or Spearman language correlation survives Holm
correction. Dominant-stance groups remain highly imbalanced, so group comparisons are
descriptive rather than decisive.

## Robustness and diagnostics

Excluding 2007–2009 leaves 213 one-day and 212 five-day VIX events. The primary stance
block then has p = 0.168, while the five-day 2-year-yield block has p = 0.101. Every
Holm-adjusted stance p-value is 1.000 in this sensitivity sample. The nominal findings
are therefore not stable general results.

The primary stance model has moderate overlap among prior market-movement controls;
its maximum VIF is 4.25. Residuals are strongly non-normal and heteroskedastic.
Seventeen observations exceed the conventional Cook's-distance threshold of 4/n, and
the maximum Cook's distance is 2.23. Robust standard errors help with inference under
heteroskedasticity; they do not eliminate influential-event sensitivity, omitted
variables, timing uncertainty, or composition effects.

## Interpretation

The cross-market results strengthen the negative conclusion. Expert-supervised stance
contains weak, economically plausible directions—a dovish share associated with
higher short-run VIX and a hawkish share associated with higher five-day 2-year
yields—but neither survives multiplicity correction or crisis exclusion. Sentiment
and emotion add no robust explanatory value. Verified release timestamps and a direct
measure of policy surprise are more valuable next steps than adding more language
models.

## Reproduction

Regenerate events first if their inputs have changed, then run the research analysis:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.analysis.events --config config/config.yaml
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.analysis.research --config config/config.yaml
```

The analysis writes ignored, reproducible files under `data/processed`:

- `research_analysis.json`
- `research_coefficients.csv`
- `research_correlations.csv`

