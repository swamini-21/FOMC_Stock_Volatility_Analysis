# Additional market data

## Sources and scope

The market-data pipeline downloads three daily series from 1993 onward:

- S&P 500 index (`^GSPC`) from Yahoo Finance through `yfinance`.
- 2-year constant-maturity Treasury yield (`DGS2`) from FRED.
- 10-year constant-maturity Treasury yield (`DGS10`) from FRED.

Yahoo Finance is a convenient, unofficial programmatic source and may change without
notice. FRED republishes the Treasury series from the Federal Reserve Board's H.15
release. Every download records retrieval time, requested dates, coverage, observation
counts, source identifiers, and a SHA-256 hash of the normalized combined dataset.

Downloaded and generated data are ignored by Git. Normalized source-level files are
written to `data/raw`, while the combined table and provenance report are written to
`data/processed`.

## Event alignment

Because FOMC release times are unavailable, every outcome uses the first observation
strictly after the statement date. The event builder creates:

- 1-day and 5-day S&P 500 returns;
- 1-day and 5-day changes in 2-year Treasury yields, in basis points;
- 1-day and 5-day changes in 10-year Treasury yields, in basis points; and
- corresponding pre-event returns and yield changes for regression controls.

This conservative timing rule avoids treating a same-day close as post-announcement
when the release time is unknown, but it may omit part of the actual announcement
reaction.

## Reproduction

Install the downloader dependency and refresh the market series:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m pip install -e ".[market-data]"
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.analysis.market_data `
  --config config/config.yaml `
  --start 1993-01-01
```

Then rebuild the event table and statistical results:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.analysis.events --config config/config.yaml
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.analysis.research --config config/config.yaml
```

