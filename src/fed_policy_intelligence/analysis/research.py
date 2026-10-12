"""Run explanatory statistical analysis of FOMC language and VIX reactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from fed_policy_intelligence.analysis.regression import _model_record
from fed_policy_intelligence.settings import ProjectSettings, load_settings

OUTCOMES = {
    "absolute_1d": ("vix_change_1d", 1, "is_usable_1d"),
    "absolute_5d": ("vix_change_5d", 5, "is_usable_5d"),
    "percentage_1d": ("vix_pct_change_1d", 1, "is_usable_1d"),
    "percentage_5d": ("vix_pct_change_5d", 5, "is_usable_5d"),
    "sp500_1d": ("sp500_return_1d", 1, "is_usable_sp500_1d"),
    "sp500_5d": ("sp500_return_5d", 5, "is_usable_sp500_5d"),
    "dgs2_1d": ("dgs2_basis_points_1d", 1, "is_usable_dgs2_1d"),
    "dgs2_5d": ("dgs2_basis_points_5d", 5, "is_usable_dgs2_5d"),
    "dgs10_1d": ("dgs10_basis_points_1d", 1, "is_usable_dgs10_1d"),
    "dgs10_5d": ("dgs10_basis_points_5d", 5, "is_usable_dgs10_5d"),
}
BASELINE_TERMS = (
    "pre_event_vix",
    "pre_event_vix_change_1d",
    "pre_event_vix_change_5d",
    "pre_event_sp500_return_1d",
    "pre_event_sp500_return_5d",
    "pre_event_dgs2_basis_points_1d",
    "pre_event_dgs10_basis_points_1d",
    "document_count",
    "C(event_era)",
)
MODEL_BLOCKS = (
    ("baseline", ()),
    ("stance", ("domain_policy_share_hawkish", "domain_policy_share_dovish")),
    ("sentiment", ("sentiment_score",)),
    ("emotion", ("emotion_probability_fear",)),
)
LANGUAGE_FEATURES = (
    "domain_policy_share_hawkish",
    "domain_policy_share_dovish",
    "sentiment_score",
    "emotion_probability_fear",
)


def _prepare_sample(
    events: pd.DataFrame, outcome: str, usable_column: str
) -> pd.DataFrame:
    required = {
        outcome,
        usable_column,
        "event_date",
        "event_era",
        "domain_policy_share_neutral",
        *LANGUAGE_FEATURES,
        *(term for term in BASELINE_TERMS if not term.startswith("C(")),
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Research event data is missing columns: {missing}")
    sample = events.copy()
    sample["event_date"] = pd.to_datetime(sample["event_date"], errors="coerce")
    numeric = {outcome, *LANGUAGE_FEATURES, "domain_policy_share_neutral"}
    numeric.update(term for term in BASELINE_TERMS if not term.startswith("C("))
    for column in numeric:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    usable = sample[usable_column]
    if not pd.api.types.is_bool_dtype(usable):
        usable = usable.astype(str).str.lower().eq("true")
    complete = sample.loc[
        usable
        & sample["event_date"].notna()
        & sample["event_era"].notna()
        & sample[list(numeric)].notna().all(axis=1)
    ].sort_values("event_date")
    if complete.empty:
        raise ValueError(f"No complete observations are available for {outcome}.")
    return complete


def _fit(formula: str, sample: pd.DataFrame, horizon: int):
    fitted = smf.ols(formula, data=sample).fit()
    if horizon == 1:
        robust = fitted.get_robustcov_results(cov_type="HC3")
        covariance = {"type": "HC3"}
    else:
        robust = fitted.get_robustcov_results(cov_type="HAC", maxlags=horizon)
        covariance = {"type": "HAC", "maxlags": horizon}
    return fitted, robust, covariance


def _wald(robust, terms: tuple[str, ...]) -> dict[str, float]:
    names = robust.model.exog_names
    matrix = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        matrix[row, names.index(term)] = 1.0
    test = robust.wald_test(matrix, scalar=True)
    return {"statistic": float(test.statistic), "p_value": float(test.pvalue)}


def _vif(fitted) -> list[dict[str, Any]]:
    matrix = np.asarray(fitted.model.exog)
    records = []
    for index, term in enumerate(fitted.model.exog_names):
        if term == "Intercept":
            continue
        try:
            value = float(variance_inflation_factor(matrix, index))
        except (ValueError, np.linalg.LinAlgError):
            value = float("nan")
        records.append({"term": term, "vif": value})
    return records


def _model_details(fitted, robust, covariance: dict[str, Any]) -> dict[str, Any]:
    record = _model_record(fitted, robust, covariance)
    cooks = np.asarray(fitted.get_influence().cooks_distance[0])
    record["vif"] = _vif(fitted)
    record["influence"] = {
        "max_cooks_distance": float(cooks.max()),
        "count_above_4_over_n": int((cooks > 4 / len(cooks)).sum()),
    }
    return record


def _descriptive_record(sample: pd.DataFrame, outcome: str) -> dict[str, Any]:
    values = sample[outcome].to_numpy(dtype=float)
    mean = float(values.mean())
    standard_error = float(stats.sem(values))
    ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=standard_error)
    t_test = stats.ttest_1samp(values, popmean=0)
    try:
        signed_rank = stats.wilcoxon(values)
        wilcoxon = {"statistic": float(signed_rank.statistic), "p_value": float(signed_rank.pvalue)}
    except ValueError:
        wilcoxon = {"statistic": float("nan"), "p_value": float("nan")}
    return {
        "n": len(values),
        "mean": mean,
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "mean_95_ci": [float(ci[0]), float(ci[1])],
        "one_sample_t_test": {
            "statistic": float(t_test.statistic),
            "p_value": float(t_test.pvalue),
        },
        "wilcoxon_signed_rank": wilcoxon,
    }


def _dominant_stance(frame: pd.DataFrame) -> pd.Series:
    columns = {
        "hawkish": "domain_policy_share_hawkish",
        "dovish": "domain_policy_share_dovish",
        "neutral": "domain_policy_share_neutral",
    }
    values = frame[list(columns.values())]
    maximum = values.max(axis=1)
    ties = values.eq(maximum, axis=0).sum(axis=1).gt(1)
    labels = values.idxmax(axis=1).map({value: key for key, value in columns.items()})
    return labels.mask(ties, "mixed_tie")


def _group_analysis(sample: pd.DataFrame, outcome: str) -> dict[str, Any]:
    grouped = sample.assign(dominant_stance=_dominant_stance(sample))
    summaries: dict[str, Any] = {}
    arrays = []
    for label, group in grouped.groupby("dominant_stance", sort=True):
        values = group[outcome].to_numpy(dtype=float)
        summaries[str(label)] = {
            "n": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
        if len(values) > 1:
            arrays.append(values)
    if len(arrays) >= 2:
        test = stats.kruskal(*arrays)
        comparison = {"statistic": float(test.statistic), "p_value": float(test.pvalue)}
    else:
        comparison = {"statistic": None, "p_value": None}
    return {"groups": summaries, "kruskal_wallis": comparison}


def _correlations(sample: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    records = []
    for feature in LANGUAGE_FEATURES:
        pearson = stats.pearsonr(sample[feature], sample[outcome])
        spearman = stats.spearmanr(sample[feature], sample[outcome])
        records.append(
            {
                "outcome": outcome,
                "feature": feature,
                "pearson_r": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
            }
        )
    return records


def _holm_adjust(items: list[dict[str, Any]], p_key: str, output_key: str) -> None:
    if not items:
        return
    adjusted = multipletests([item[p_key] for item in items], method="holm")
    for item, reject, p_value in zip(items, adjusted[0], adjusted[1], strict=True):
        item[output_key] = float(p_value)
        item[f"{output_key}_reject_0_05"] = bool(reject)


def estimate_research_models(
    events: pd.DataFrame, *, include_sensitivity: bool = True
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Estimate prespecified nested explanatory models and descriptive tests."""

    report: dict[str, Any] = {
        "design": {
            "purpose": "explanatory association, not prediction or causal identification",
            "baseline_terms": list(BASELINE_TERMS),
            "model_blocks": {name: list(terms) for name, terms in MODEL_BLOCKS},
            "policy_reference": "domain_policy_share_neutral",
            "primary_outcome": "vix_change_1d",
        },
        "outcomes": {},
    }
    coefficients: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    block_tests: dict[str, list[dict[str, Any]]] = {
        name: [] for name, terms in MODEL_BLOCKS if terms
    }

    for outcome_key, (outcome, horizon, usable_column) in OUTCOMES.items():
        sample = _prepare_sample(events, outcome, usable_column)
        outcome_record: dict[str, Any] = {
            "outcome": outcome,
            "nobs": len(sample),
            "sample_start": sample["event_date"].min().date().isoformat(),
            "sample_end": sample["event_date"].max().date().isoformat(),
            "descriptive": _descriptive_record(sample, outcome),
            "stance_groups": _group_analysis(sample, outcome),
            "models": {},
        }
        active_terms = list(BASELINE_TERMS)
        previous_fit = None
        for model_name, block in MODEL_BLOCKS:
            active_terms.extend(block)
            formula = f"{outcome} ~ {' + '.join(active_terms)}"
            fitted, robust, covariance = _fit(formula, sample, horizon)
            model_record = _model_details(fitted, robust, covariance)
            model_record["formula"] = formula
            if previous_fit is not None:
                model_record["incremental_fit"] = {
                    "delta_adjusted_r_squared": float(
                        fitted.rsquared_adj - previous_fit.rsquared_adj
                    ),
                    "delta_aic": float(fitted.aic - previous_fit.aic),
                    "delta_bic": float(fitted.bic - previous_fit.bic),
                }
            if block:
                test = _wald(robust, block)
                model_record["block_wald_test"] = test
                block_tests[model_name].append(
                    {"outcome_key": outcome_key, "model": model_name, **test}
                )
            outcome_record["models"][model_name] = model_record
            for coefficient in model_record["coefficients"]:
                coefficients.append(
                    {"outcome_key": outcome_key, "model": model_name, **coefficient}
                )
            previous_fit = fitted
        correlations.extend(_correlations(sample, outcome))
        report["outcomes"][outcome_key] = outcome_record

    for tests in block_tests.values():
        _holm_adjust(tests, "p_value", "holm_p_value")
        for test in tests:
            report["outcomes"][test["outcome_key"]]["models"][test["model"]][
                "block_wald_test"
            ].update(
                {
                    "holm_p_value": test["holm_p_value"],
                    "holm_reject_0_05": test["holm_p_value_reject_0_05"],
                }
            )
    _holm_adjust(correlations, "pearson_p_value", "pearson_holm_p_value")
    _holm_adjust(correlations, "spearman_p_value", "spearman_holm_p_value")

    if include_sensitivity:
        dated = events.copy()
        dated["event_date"] = pd.to_datetime(dated["event_date"], errors="coerce")
        filtered = dated.loc[~dated["event_date"].dt.year.between(2007, 2009)]
        sensitivity, _, _ = estimate_research_models(filtered, include_sensitivity=False)
        report["crisis_exclusion_sensitivity"] = {
            "excluded_years": [2007, 2008, 2009],
            "outcomes": sensitivity["outcomes"],
        }
    return report, pd.DataFrame(coefficients), pd.DataFrame(correlations)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_research_pipeline(settings: ProjectSettings) -> dict[str, Path]:
    event_path = settings.data.processed_dir / "fed_vix_events.csv"
    if not event_path.is_file():
        raise FileNotFoundError(f"Event data not found: {event_path}")
    report, coefficients, correlations = estimate_research_models(pd.read_csv(event_path))
    paths = {
        "report": settings.data.processed_dir / "research_analysis.json",
        "coefficients": settings.data.processed_dir / "research_coefficients.csv",
        "correlations": settings.data.processed_dir / "research_correlations.csv",
    }
    _write_json(report, paths["report"])
    _write_csv(coefficients, paths["coefficients"])
    _write_csv(correlations, paths["correlations"])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    paths = run_research_pipeline(load_settings(args.config))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
