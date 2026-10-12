import pandas as pd

from fed_policy_intelligence.analysis.market_data import download_market_controls


def test_download_market_controls_normalizes_sources() -> None:
    sp500 = pd.DataFrame(
        {"Adj Close": [4700.0, 4725.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
    )
    fred = {
        "DGS2": "DATE,DGS2\n2024-01-02,4.25\n2024-01-03,.\n2024-01-04,4.30\n",
        "DGS10": "observation_date,DGS10\n2024-01-02,3.95\n2024-01-04,4.00\n",
    }

    controls, report, raw = download_market_controls(
        "2024-01-01",
        "2024-01-05",
        sp500_loader=lambda _start, _end: sp500,
        fred_loader=lambda series_id: fred[series_id],
    )

    assert controls["trading_date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert raw["dgs2"]["dgs2"].tolist() == [4.25, 4.30]
    assert report["observations"] == {"sp500": 2, "dgs2": 2, "dgs10": 2}
    assert len(report["normalized_sha256"]) == 64
