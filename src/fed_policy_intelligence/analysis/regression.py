"""Estimate frozen baseline and NLP-enhanced VIX event regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from fed_policy_intelligence.settings import ProjectSettings, load_settings

BASELINE_TERMS = (
    "pre_event_vix",
    "pre_event_vix_change_1d",
    "pre_event_vix_change_5d",
    "C(event_era)",
)
NLP_TERMS = (
    "domain_policy_share_hawkish",
    "domain_policy_share_dovish",
    "document_count",
)
NUMERIC_PREDICTORS = (
    "pre_event_vix",
    "pre_event_vix_change_1d",
    "pre_event_vix_change_5d",
    "domain_policy_share_hawkish",
    "domain_policy_share_dovish",
    "document_count",
)
OUTCOME_SCALES = {
    "absolute": "vix_change",
    "percentage": "vix_pct_change",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Event data is missing required columns: {missing}")


def prepare_regression_sample(
    events: pd.DataFrame, horizon: int, *, outcome_scale: str = "absolute"
) -> pd.DataFrame:
    """Select complete event rows for one frozen outcome specification."""

    if outcome_scale not in OUTCOME_SCALES:
        raise ValueError(f"Unsupported outcome scale: {outcome_scale}")
    outcome = f"{OUTCOME_SCALES[outcome_scale]}_{horizon}d"
    usable = f"is_usable_{horizon}d"
    required = {outcome, usable, "event_date", "event_era", *NUMERIC_PREDICTORS}
    _require_columns(events, required)
    sample = events.copy()
    sample["event_date"] = pd.to_datetime(sample["event_date"], errors="coerce")
    for column in (outcome, *NUMERIC_PREDICTORS):
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    if not pd.api.types.is_bool_dtype(sample[usable]):
        sample[usable] = sample[usable].astype(str).str.lower().eq("true")
    complete = sample.loc[
        sample[usable]
        & sample["event_date"].notna()
        & sample["event_era"].notna()
        & sample[[outcome, *NUMERIC_PREDICTORS]].notna().all(axis=1)
    ].sort_values("event_date")
    if complete.empty:
        raise ValueError(f"No complete events are available for the {horizon}-day model.")
    return complete


def _fit_formula(formula: str, sample: pd.DataFrame, horizon: int):
    fitted = smf.ols(formula, data=sample).fit()
    if horizon == 1:
        robust = fitted.get_robustcov_results(cov_type="HC3")
        covariance = {"type": "HC3"}
    else:
        robust = fitted.get_robustcov_results(cov_type="HAC", maxlags=horizon)
        covariance = {"type": "HAC", "maxlags": horizon}
    return fitted, robust, covariance


def _coefficient_records(robust) -> list[dict[str, object]]:
    names = robust.model.exog_names
    intervals = np.asarray(robust.conf_int(alpha=0.05))
    return [
        {
            "term": name,
            "coefficient": float(robust.params[index]),
            "std_error": float(robust.bse[index]),
            "statistic": float(robust.tvalues[index]),
            "p_value": float(robust.pvalues[index]),
            "ci_lower": float(intervals[index, 0]),
            "ci_upper": float(intervals[index, 1]),
        }
        for index, name in enumerate(names)
    ]


def _diagnostics(fitted) -> dict[str, float]:
    residuals = np.asarray(fitted.resid)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, fitted.model.exog)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "durbin_watson": float(durbin_watson(residuals)),
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_p_value": float(jb_pvalue),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurtosis),
        "breusch_pagan_statistic": float(bp_stat),
        "breusch_pagan_p_value": float(bp_pvalue),
        "condition_number": float(fitted.condition_number),
    }


def _model_record(fitted, robust, covariance: dict[str, object]) -> dict[str, object]:
    return {
        "nobs": int(fitted.nobs),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "covariance": covariance,
        "coefficients": _coefficient_records(robust),
        "diagnostics": _diagnostics(fitted),
    }


def _incremental_wald_test(
    robust, terms: tuple[str, ...] = NLP_TERMS
) -> dict[str, float]:
    names = robust.model.exog_names
    matrix = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        if term not in names:
            raise ValueError(f"Enhanced model is missing frozen NLP term {term!r}.")
        matrix[row, names.index(term)] = 1.0
    test = robust.wald_test(matrix, scalar=True)
    return {"statistic": float(test.statistic), "p_value": float(test.pvalue)}


def _crisis_exclusion_sensitivity(
    events: pd.DataFrame, horizons: tuple[int, ...]
) -> dict[str, object]:
    dated = events.copy()
    dated["event_date"] = pd.to_datetime(dated["event_date"], errors="coerce")
    filtered = dated.loc[~dated["event_date"].dt.year.between(2007, 2009)].copy()
    models: dict[str, object] = {}
    for outcome_scale in OUTCOME_SCALES:
        for horizon in horizons:
            sample = prepare_regression_sample(
                filtered, horizon, outcome_scale=outcome_scale
            )
            active_terms = tuple(
                term for term in NLP_TERMS if sample[term].nunique(dropna=True) > 1
            )
            dropped_terms = [term for term in NLP_TERMS if term not in active_terms]
            outcome = f"{OUTCOME_SCALES[outcome_scale]}_{horizon}d"
            baseline_formula = f"{outcome} ~ {' + '.join(BASELINE_TERMS)}"
            enhanced_formula = f"{baseline_formula} + {' + '.join(active_terms)}"
            baseline_fit, _, _ = _fit_formula(baseline_formula, sample, horizon)
            enhanced_fit, enhanced_robust, _ = _fit_formula(
                enhanced_formula, sample, horizon
            )
            models[f"{outcome_scale}_{horizon}d"] = {
                "nobs": int(enhanced_fit.nobs),
                "active_nlp_terms": list(active_terms),
                "dropped_zero_variance_terms": dropped_terms,
                "delta_adjusted_r_squared": (
                    enhanced_fit.rsquared_adj - baseline_fit.rsquared_adj
                ),
                "nlp_joint_wald_test": _incremental_wald_test(
                    enhanced_robust, active_terms
                ),
            }
    return {
        "description": "Post-hoc diagnostic excluding 2007-2009 crisis events",
        "excluded_years": [2007, 2008, 2009],
        "models": models,
    }


def estimate_nested_models(
    events: pd.DataFrame, *, horizons: tuple[int, ...] = (1, 5)
) -> tuple[dict[str, object], pd.DataFrame]:
    """Estimate baseline and enhanced models for each configured horizon."""

    results: dict[str, object] = {
        "specification": {
            "outcomes": [
                f"{prefix}_{horizon}d"
                for prefix in OUTCOME_SCALES.values()
                for horizon in horizons
            ],
            "baseline_terms": list(BASELINE_TERMS),
            "nlp_terms": list(NLP_TERMS),
            "emotion_included": False,
            "policy_reference": "domain_policy_share_neutral",
            "policy_model": "locally fine-tuned RoBERTa-base on expert FOMC labels",
            "interpretation": "exploratory association, not causal effect",
        },
        "models": {},
    }
    coefficient_rows: list[dict[str, object]] = []
    for outcome_scale, prefix in OUTCOME_SCALES.items():
        for horizon in horizons:
            sample = prepare_regression_sample(
                events, horizon, outcome_scale=outcome_scale
            )
            outcome = f"{prefix}_{horizon}d"
            baseline_formula = f"{outcome} ~ {' + '.join(BASELINE_TERMS)}"
            enhanced_formula = f"{baseline_formula} + {' + '.join(NLP_TERMS)}"
            baseline_fit, baseline_robust, baseline_covariance = _fit_formula(
                baseline_formula, sample, horizon
            )
            enhanced_fit, enhanced_robust, enhanced_covariance = _fit_formula(
                enhanced_formula, sample, horizon
            )
            baseline_record = _model_record(
                baseline_fit, baseline_robust, baseline_covariance
            )
            enhanced_record = _model_record(
                enhanced_fit, enhanced_robust, enhanced_covariance
            )
            comparison = {
                "delta_r_squared": enhanced_fit.rsquared - baseline_fit.rsquared,
                "delta_adjusted_r_squared": (
                    enhanced_fit.rsquared_adj - baseline_fit.rsquared_adj
                ),
                "delta_aic": enhanced_fit.aic - baseline_fit.aic,
                "delta_bic": enhanced_fit.bic - baseline_fit.bic,
                "nlp_joint_wald_test": _incremental_wald_test(enhanced_robust),
            }
            model_key = f"{outcome_scale}_{horizon}d"
            results["models"][model_key] = {
                "outcome": outcome,
                "outcome_scale": outcome_scale,
                "horizon": horizon,
                "sample_start": sample["event_date"].min().date().isoformat(),
                "sample_end": sample["event_date"].max().date().isoformat(),
                "baseline_formula": baseline_formula,
                "enhanced_formula": enhanced_formula,
                "baseline": baseline_record,
                "enhanced": enhanced_record,
                "comparison": comparison,
            }
            for model_name, record in (
                ("baseline", baseline_record),
                ("enhanced", enhanced_record),
            ):
                for coefficient in record["coefficients"]:
                    coefficient_rows.append(
                        {
                            "outcome_scale": outcome_scale,
                            "horizon": f"{horizon}d",
                            "model": model_name,
                            **coefficient,
                        }
                    )
    results["sensitivity"] = _crisis_exclusion_sensitivity(events, horizons)
    return results, pd.DataFrame.from_records(coefficient_rows)


def _write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def run_regression_pipeline(settings: ProjectSettings) -> dict[str, Path]:
    """Estimate the frozen models from the Phase 5 event dataset."""

    event_path = settings.data.processed_dir / "fed_vix_events.csv"
    if not event_path.is_file():
        raise FileNotFoundError(
            f"Event data not found at {event_path}; run the event pipeline first."
        )
    results, coefficients = estimate_nested_models(pd.read_csv(event_path))
    results_path = settings.data.processed_dir / "regression_results.json"
    coefficients_path = settings.data.processed_dir / "regression_coefficients.csv"
    _write_json_atomic(results, results_path)
    _write_csv_atomic(coefficients, coefficients_path)
    return {"results": results_path, "coefficients": coefficients_path}


def main(argv: list[str] | None = None) -> int:
    """Estimate frozen baseline and NLP-enhanced event regressions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    paths = run_regression_pipeline(load_settings(args.config))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
