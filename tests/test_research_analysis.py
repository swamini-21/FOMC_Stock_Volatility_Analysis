import numpy as np
import pandas as pd

from fed_policy_intelligence.analysis.research import estimate_research_models


def test_research_analysis_returns_explanatory_hierarchy() -> None:
    rng = np.random.default_rng(7)
    rows = []
    dates = pd.date_range("1995-01-15", periods=90, freq="90D")
    for index, date in enumerate(dates):
        hawkish, dovish, neutral = rng.dirichlet([2.0, 2.0, 6.0])
        sentiment = rng.normal(0, 0.25)
        fear = rng.beta(2, 8)
        outcome = (
            0.3 * hawkish
            - 0.2 * dovish
            + 0.1 * sentiment
            + rng.normal(0, 0.15)
        )
        era = "1990-2004" if date.year < 2005 else "2005-2014"
        if date.year >= 2015:
            era = "2015-2025"
        rows.append(
            {
                "event_date": date.date().isoformat(),
                "event_era": era,
                "pre_event_vix": 15 + rng.normal(0, 2),
                "pre_event_vix_change_1d": rng.normal(0, 0.5),
                "pre_event_vix_change_5d": rng.normal(0, 1),
                "pre_event_sp500_return_1d": rng.normal(0, 0.01),
                "pre_event_sp500_return_5d": rng.normal(0, 0.02),
                "pre_event_dgs2_basis_points_1d": rng.normal(0, 3),
                "pre_event_dgs10_basis_points_1d": rng.normal(0, 3),
                "document_count": 1 + int(index % 10 == 0),
                "domain_policy_share_hawkish": hawkish,
                "domain_policy_share_dovish": dovish,
                "domain_policy_share_neutral": neutral,
                "sentiment_score": sentiment,
                "emotion_probability_fear": fear,
                "is_usable_1d": True,
                "is_usable_5d": True,
                "is_usable_sp500_1d": True,
                "is_usable_sp500_5d": True,
                "is_usable_dgs2_1d": True,
                "is_usable_dgs2_5d": True,
                "is_usable_dgs10_1d": True,
                "is_usable_dgs10_5d": True,
                "vix_change_1d": outcome,
                "vix_change_5d": outcome * 1.5,
                "vix_pct_change_1d": outcome / 15,
                "vix_pct_change_5d": outcome / 10,
                "sp500_return_1d": -outcome / 100,
                "sp500_return_5d": -outcome / 70,
                "dgs2_basis_points_1d": outcome * 5,
                "dgs2_basis_points_5d": outcome * 7,
                "dgs10_basis_points_1d": outcome * 3,
                "dgs10_basis_points_5d": outcome * 4,
            }
        )

    report, coefficients, correlations = estimate_research_models(pd.DataFrame(rows))

    assert report["design"]["purpose"].startswith("explanatory")
    assert set(report["outcomes"]["absolute_1d"]["models"]) == {
        "baseline",
        "stance",
        "sentiment",
        "emotion",
    }
    assert "holm_p_value" in report["outcomes"]["absolute_1d"]["models"]["stance"][
        "block_wald_test"
    ]
    assert "crisis_exclusion_sensitivity" in report
    assert set(coefficients["model"]) == {"baseline", "stance", "sentiment", "emotion"}
    assert len(correlations) == 40
