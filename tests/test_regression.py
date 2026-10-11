import pandas as pd

from fed_policy_intelligence.analysis.regression import estimate_nested_models


def test_estimate_nested_models_returns_frozen_comparisons() -> None:
    rows = []
    eras = ("1990-2004", "2005-2014", "2015-2025")
    for index in range(60):
        hawkish = float(index % 4 == 0)
        dovish = float(index % 4 == 1)
        outcome = 0.2 * hawkish - 0.1 * dovish + (index % 5) / 20
        rows.append(
            {
                "event_date": f"{2000 + index // 12:04d}-{index % 12 + 1:02d}-15",
                "event_era": eras[min(index // 20, 2)],
                "pre_event_vix": 15 + index / 10,
                "pre_event_vix_change_1d": ((index % 3) - 1) / 10,
                "pre_event_vix_change_5d": ((index % 5) - 2) / 10,
                "domain_policy_share_hawkish": hawkish,
                "domain_policy_share_dovish": dovish,
                "document_count": 1 + int(index % 10 == 0),
                "is_usable_1d": True,
                "is_usable_5d": True,
                "vix_change_1d": outcome,
                "vix_change_5d": outcome * 1.5,
                "vix_pct_change_1d": outcome / 15,
                "vix_pct_change_5d": outcome / 10,
            }
        )

    results, coefficients = estimate_nested_models(pd.DataFrame(rows))

    assert set(results["models"]) == {
        "absolute_1d",
        "absolute_5d",
        "percentage_1d",
        "percentage_5d",
    }
    assert results["models"]["absolute_1d"]["baseline"]["nobs"] == 60
    assert results["models"]["percentage_5d"]["enhanced"]["covariance"] == {
        "type": "HAC",
        "maxlags": 5,
    }
    assert "nlp_joint_wald_test" in results["models"]["absolute_1d"]["comparison"]
    assert "models" in results["sensitivity"]
    assert set(coefficients["model"]) == {"baseline", "enhanced"}
