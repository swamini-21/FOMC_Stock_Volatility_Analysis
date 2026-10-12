# Streamlit research explorer

## Purpose

The application presents cached FOMC language and market-reaction results as an
interactive observational research explorer. It does not forecast markets, generate
trading recommendations, establish causal effects, or call an LLM API.

## Views

- **Overview:** sample coverage, central finding, and policy-share history.
- **Announcement explorer:** original and cleaned statement text, policy stance,
  financial sentiment, and emotion probabilities.
- **Policy trends:** adjustable rolling-event policy-share trends.
- **Market reactions:** filtered VIX, S&P 500, and Treasury outcome plots with
  event-level data.
- **Statistical results:** nested-model block tests and coefficient confidence
  intervals.
- **Methodology:** design, inference choices, permitted interpretation, and known
  limitations.

Sidebar date and era controls filter event-level exploratory views. They do not
re-estimate the precomputed regression models. The statistical-results page states
this explicitly.

## Run locally

The data, NLP, event, and research pipelines must have produced the cached files in
`data/processed`. No Gemini key is required.

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m pip install -e ".[app]"
& "$FpiGpuEnv\Scripts\python.exe" -m streamlit run `
  src/fed_policy_intelligence/app/streamlit_app.py
```

Streamlit prints the local browser URL after startup. Stop the server with `Ctrl+C`.

## Testing

The automated suite uses a small committed cache fixture and Streamlit's application
testing interface. It opens every page, verifies key content, confirms missing-cache
errors are user-readable, and checks that malformed cached inputs are rejected. The
tests do not download data, load transformer models, or call an external API.

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m pytest tests/test_app_data.py tests/test_streamlit_app.py
```

