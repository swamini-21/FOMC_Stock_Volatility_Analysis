"""Download and validate daily market controls used by the event study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from fed_policy_intelligence.settings import ProjectSettings, load_settings

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
TREASURY_SERIES = {"DGS2": "dgs2", "DGS10": "dgs10"}


def _default_sp500_loader(start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            'S&P 500 download requires the optional dependency: pip install -e ".[market-data]"'
        ) from exc
    exclusive_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    return yf.download(
        "^GSPC",
        start=start,
        end=exclusive_end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )


def _default_fred_loader(series_id: str) -> str:
    request = Request(
        FRED_URL.format(series_id=series_id),
        headers={"User-Agent": "federal-reserve-policy-intelligence/0.1"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _normalize_sp500(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Yahoo Finance returned no S&P 500 observations.")
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    price_column = "Adj Close" if "Adj Close" in normalized.columns else "Close"
    if price_column not in normalized.columns:
        raise ValueError("Yahoo Finance response has neither Adj Close nor Close.")
    dates = pd.to_datetime(normalized.index, errors="coerce", utc=True).tz_localize(None)
    output = pd.DataFrame(
        {
            "trading_date": dates.normalize(),
            "sp500_close": pd.to_numeric(normalized[price_column], errors="coerce").to_numpy(),
        }
    )
    return _validate_series(output, "sp500_close")


def _normalize_fred(payload: str, series_id: str, output_column: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(payload))
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_column not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"FRED response for {series_id} has unexpected columns.")
    output = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(frame[date_column], errors="coerce"),
            output_column: pd.to_numeric(frame[series_id], errors="coerce"),
        }
    )
    return _validate_series(output, output_column)


def _validate_series(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    clean = frame.dropna(subset=["trading_date", value_column]).copy()
    clean["trading_date"] = clean["trading_date"].dt.normalize()
    clean = clean.sort_values("trading_date").reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"No usable observations are available for {value_column}.")
    if clean["trading_date"].duplicated().any():
        raise ValueError(f"Duplicate dates found in {value_column}.")
    if not clean[value_column].gt(0).all():
        raise ValueError(f"Non-positive values found in {value_column}.")
    return clean


def download_market_controls(
    start: str,
    end: str,
    *,
    sp500_loader: Callable[[str, str], pd.DataFrame] = _default_sp500_loader,
    fred_loader: Callable[[str], str] = _default_fred_loader,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    """Download normalized S&P 500 and Treasury series with provenance."""

    if date.fromisoformat(start) >= date.fromisoformat(end):
        raise ValueError("Market-data start date must be earlier than end date.")
    sp500 = _normalize_sp500(sp500_loader(start, end))
    series = {"sp500": sp500}
    merged = sp500
    for series_id, column in TREASURY_SERIES.items():
        treasury = _normalize_fred(fred_loader(series_id), series_id, column)
        treasury = treasury.loc[treasury["trading_date"].between(start, end)]
        if treasury.empty:
            raise ValueError(f"FRED {series_id} has no observations in the requested period.")
        series[series_id.lower()] = treasury
        merged = merged.merge(treasury, on="trading_date", how="outer", validate="one_to_one")
    merged = merged.loc[merged["trading_date"].between(start, end)]
    merged = merged.sort_values("trading_date").reset_index(drop=True)
    csv_bytes = merged.to_csv(index=False).encode("utf-8")
    report: dict[str, object] = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "requested_start": start,
        "requested_end": end,
        "sources": {
            "sp500": {"provider": "Yahoo Finance via yfinance", "ticker": "^GSPC"},
            "dgs2": {"provider": "FRED", "series_id": "DGS2"},
            "dgs10": {"provider": "FRED", "series_id": "DGS10"},
        },
        "observations": {name: len(frame) for name, frame in series.items()},
        "coverage": {
            name: {
                "start": frame["trading_date"].min().date().isoformat(),
                "end": frame["trading_date"].max().date().isoformat(),
            }
            for name, frame in series.items()
        },
        "normalized_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    return merged, report, series


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_market_download(
    settings: ProjectSettings, start: str = "1993-01-01", end: str | None = None
) -> dict[str, Path]:
    selected_end = end or date.today().isoformat()
    controls, report, raw_series = download_market_controls(start, selected_end)
    raw_dir = settings.root_dir / "data" / "raw"
    paths = {
        "market_controls": settings.data.processed_dir / "market_controls.csv",
        "report": settings.data.processed_dir / "market_data_report.json",
    }
    _write_csv(controls, paths["market_controls"])
    _write_json(report, paths["report"])
    for name, frame in raw_series.items():
        raw_path = raw_dir / f"{name}.csv"
        _write_csv(frame, raw_path)
        paths[f"raw_{name}"] = raw_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)
    paths = run_market_download(load_settings(args.config), args.start, args.end)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
