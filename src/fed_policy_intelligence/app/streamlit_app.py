"""Streamlit research explorer for FOMC language and market reactions."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fed_policy_intelligence.app.data import (
    EVENT_OUTCOMES,
    OUTCOME_LABELS,
    DashboardData,
    filter_events,
    load_dashboard_data,
    parse_probabilities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = Path(
    os.environ.get("FPI_DASHBOARD_DATA_DIR", PROJECT_ROOT / "data" / "processed")
).resolve()
STANCE_COLORS = {"hawkish": "#c65d4b", "dovish": "#2f7f73", "neutral": "#718096"}


@st.cache_data(show_spinner=False)
def _load_data(processed_dir: str) -> DashboardData:
    return load_dashboard_data(processed_dir)


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1280px; padding-top: 2rem;}
        [data-testid="stMetric"] {background: rgba(127, 127, 127, 0.10);
          border: 1px solid rgba(127, 127, 127, 0.24);
          border-radius: 10px; padding: 0.8rem;}
        .research-note {background: rgba(47, 127, 115, 0.16);
          border-left: 4px solid #2f7f73;
          padding: 0.8rem 1rem; border-radius: 4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _dominant_stance(events: pd.DataFrame) -> pd.Series:
    stance = events[
        [
            "domain_policy_share_hawkish",
            "domain_policy_share_dovish",
            "domain_policy_share_neutral",
        ]
    ].idxmax(axis=1)
    return stance.str.replace("domain_policy_share_", "", regex=False)


def _sidebar_filters(events: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Event filters")
    minimum = events["event_date"].min().date()
    maximum = events["event_date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Date range", value=(minimum, maximum), min_value=minimum, max_value=maximum
    )
    if len(selected_dates) != 2:
        selected_dates = (minimum, maximum)
    available_eras = sorted(events["event_era"].dropna().unique())
    eras = st.sidebar.multiselect("Historical eras", available_eras, default=available_eras)
    return filter_events(events, selected_dates[0], selected_dates[1], eras)


def _overview(data: DashboardData, events: pd.DataFrame) -> None:
    st.header("Research overview")
    st.markdown(
        '<div class="research-note"><b>Observational study:</b> results describe '
        "associations, not forecasts, trading signals, or causal effects.</div>",
        unsafe_allow_html=True,
    )
    statements = data.documents.loc[data.documents["document_type"].eq("statement")]
    cols = st.columns(4)
    cols[0].metric("FOMC statements", f"{len(statements):,}")
    cols[1].metric("Market events", f"{len(events):,}")
    cols[2].metric("First event year", str(events["event_date"].min().year))
    cols[3].metric("Latest event year", str(events["event_date"].max().year))

    st.subheader("Main result")
    st.write(
        "After adding S&P 500 and Treasury controls, no language block survives Holm "
        "correction and crisis exclusion. Policy stance shows suggestive directions, "
        "but the evidence is not stable enough for a general market-effect claim."
    )
    trend = events.melt(
        id_vars="event_date",
        value_vars=[
            "domain_policy_share_hawkish",
            "domain_policy_share_dovish",
            "domain_policy_share_neutral",
        ],
        var_name="stance",
        value_name="sentence_share",
    )
    trend["stance"] = trend["stance"].str.replace("domain_policy_share_", "", regex=False)
    figure = px.scatter(
        trend,
        x="event_date",
        y="sentence_share",
        color="stance",
        color_discrete_map=STANCE_COLORS,
        opacity=0.65,
        labels={"event_date": "Statement date", "sentence_share": "Sentence share"},
    )
    figure.update_layout(legend_title_text="Policy stance", height=430)
    st.plotly_chart(figure, width="stretch")


def _announcement_explorer(data: DashboardData, events: pd.DataFrame) -> None:
    st.header("Announcement explorer")
    statements = data.documents.loc[
        data.documents["document_type"].eq("statement")
        & data.documents["domain_policy_status"].eq("success")
        & data.documents["publication_date"].isin(events["event_date"])
    ].copy()
    statements["selection"] = (
        statements["publication_date"].dt.strftime("%Y-%m-%d")
        + " · "
        + statements["document_id"].astype(str)
    )
    selected = st.selectbox("FOMC statement", statements["selection"].tolist())
    row = statements.loc[statements["selection"].eq(selected)].iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Policy stance", str(row["domain_policy_stance"]).title())
    metrics[1].metric("Hawkish share", f"{row['domain_policy_share_hawkish']:.1%}")
    metrics[2].metric("Dovish share", f"{row['domain_policy_share_dovish']:.1%}")
    metrics[3].metric("Neutral share", f"{row['domain_policy_share_neutral']:.1%}")

    shares = pd.DataFrame(
        {
            "stance": ["hawkish", "dovish", "neutral"],
            "share": [
                row["domain_policy_share_hawkish"],
                row["domain_policy_share_dovish"],
                row["domain_policy_share_neutral"],
            ],
        }
    )
    figure = px.bar(
        shares,
        x="share",
        y="stance",
        orientation="h",
        color="stance",
        color_discrete_map=STANCE_COLORS,
        range_x=[0, 1],
    )
    figure.update_layout(showlegend=False, height=280, xaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")

    sentiment = parse_probabilities(row["sentiment_probabilities"])
    emotion = parse_probabilities(row["emotion_probabilities"])
    left, right = st.columns(2)
    left.subheader("Financial sentiment")
    left.dataframe(
        pd.DataFrame(sentiment.items(), columns=["label", "probability"]),
        hide_index=True,
        width="stretch",
    )
    right.subheader("Emotion probabilities")
    right.dataframe(
        pd.DataFrame(emotion.items(), columns=["label", "probability"]),
        hide_index=True,
        width="stretch",
    )
    raw_tab, cleaned_tab = st.tabs(["Original text", "Cleaned text"])
    raw_tab.text_area("Original statement", row["raw_text"], height=420, disabled=True)
    cleaned_tab.text_area("Cleaned statement", row["cleaned_text"], height=420, disabled=True)


def _policy_trends(events: pd.DataFrame) -> None:
    st.header("Historical policy trends")
    rolling_window = st.slider("Rolling event window", 1, 20, 8)
    columns = [
        "domain_policy_share_hawkish",
        "domain_policy_share_dovish",
        "domain_policy_share_neutral",
    ]
    trend = events[["event_date", *columns]].copy()
    trend[columns] = trend[columns].rolling(rolling_window, min_periods=1).mean()
    long = trend.melt("event_date", var_name="stance", value_name="sentence_share")
    long["stance"] = long["stance"].str.replace("domain_policy_share_", "", regex=False)
    figure = px.line(
        long,
        x="event_date",
        y="sentence_share",
        color="stance",
        color_discrete_map=STANCE_COLORS,
        labels={"event_date": "Statement date", "sentence_share": "Rolling mean share"},
    )
    figure.update_layout(legend_title_text="Policy stance", yaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")


def _market_reactions(events: pd.DataFrame) -> None:
    st.header("Market-reaction explorer")
    selected_label = st.selectbox("Market outcome", list(EVENT_OUTCOMES))
    outcome, unit = EVENT_OUTCOMES[selected_label]
    plotted = events.dropna(subset=[outcome]).copy()
    plotted["dominant_stance"] = _dominant_stance(plotted)
    scatter = px.scatter(
        plotted,
        x="domain_policy_net_stance",
        y=outcome,
        color="event_era",
        hover_data={"event_date": "|%Y-%m-%d", "pre_event_vix": ":.2f"},
        labels={
            "domain_policy_net_stance": "Net stance (hawkish − dovish)",
            outcome: unit,
            "event_era": "Era",
        },
        trendline="ols",
    )
    scatter.add_hline(y=0, line_dash="dot", line_color="#718096")
    st.plotly_chart(scatter, width="stretch")
    st.caption(
        "The fitted line is a descriptive bivariate trend. Statistical inference comes "
        "from the controlled robust regressions."
    )

    box = px.box(
        plotted,
        x="dominant_stance",
        y=outcome,
        color="dominant_stance",
        color_discrete_map=STANCE_COLORS,
        points="outliers",
        labels={"dominant_stance": "Dominant stance", outcome: unit},
    )
    box.update_layout(showlegend=False)
    st.plotly_chart(box, width="stretch")
    with st.expander("View event-level data"):
        st.dataframe(
            plotted[
                [
                    "event_date",
                    "event_era",
                    "dominant_stance",
                    "domain_policy_net_stance",
                    outcome,
                ]
            ],
            hide_index=True,
            width="stretch",
        )


def _statistical_results(data: DashboardData) -> None:
    st.header("Statistical results")
    st.caption(
        "These are precomputed full-sample estimates; sidebar event filters do not "
        "re-estimate the models."
    )
    outcome_key = st.selectbox(
        "Outcome", list(OUTCOME_LABELS), format_func=OUTCOME_LABELS.get
    )
    model_name = st.selectbox("Nested model", ["stance", "sentiment", "emotion"])
    record = data.research["outcomes"][outcome_key]["models"][model_name]
    block = record["block_wald_test"]
    metrics = st.columns(4)
    metrics[0].metric("Observations", f"{record['nobs']:,}")
    metrics[1].metric("Adjusted R²", f"{record['adjusted_r_squared']:.3f}")
    metrics[2].metric("Block p-value", f"{block['p_value']:.3f}")
    metrics[3].metric("Holm p-value", f"{block['holm_p_value']:.3f}")

    coefficient_data = data.coefficients.loc[
        data.coefficients["outcome_key"].eq(outcome_key)
        & data.coefficients["model"].eq(model_name)
        & ~data.coefficients["term"].eq("Intercept")
    ].copy()
    coefficient_data["significant_raw"] = coefficient_data["p_value"].lt(0.05)
    figure = go.Figure()
    for significant, group in coefficient_data.groupby("significant_raw"):
        figure.add_trace(
            go.Scatter(
                x=group["coefficient"],
                y=group["term"],
                mode="markers",
                name="Raw p < 0.05" if significant else "Raw p ≥ 0.05",
                marker_color="#2f7f73" if significant else "#718096",
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group["ci_upper"] - group["coefficient"],
                    "arrayminus": group["coefficient"] - group["ci_lower"],
                },
                customdata=group[["p_value"]],
                hovertemplate=(
                    "%{y}<br>Estimate: %{x:.4f}<br>p: "
                    "%{customdata[0]:.4f}<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0, line_dash="dot", line_color="#2d3748")
    figure.update_layout(height=max(430, 34 * len(coefficient_data)), xaxis_title="Estimate")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Color uses the coefficient's unadjusted p-value. The block-level Holm value "
        "is the appropriate multiplicity-aware summary shown above."
    )


def _methodology() -> None:
    st.header("Methodology and limitations")
    st.markdown(
        """
        **Design.** Statement-level observational event study using the first usable
        market observation strictly after each date-only FOMC release. Models add
        expert-supervised policy stance, financial sentiment, and fear sequentially
        to conventional pre-event market controls.

        **Inference.** One-day models use HC3 robust standard errors. Five-day models
        use HAC standard errors with five lags. Language-block tests are corrected
        across outcomes with Holm's method. Crisis-exclusion and influence diagnostics
        evaluate fragility.

        **Key limitations.** Publication times are unavailable; policy language is
        endogenous to economic conditions; the stance classifier has moderate rather
        than expert-level performance; event samples are limited; overlapping windows,
        influential crises, omitted policy surprises, and revised macroeconomic data
        constrain interpretation.

        **Permitted conclusion.** The application supports exploration of conditional
        associations. It does not establish causality, predict markets, or provide
        investment advice.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Federal Reserve Policy Intelligence",
        page_icon="📊",
        layout="wide",
    )
    _style()
    st.title("Federal Reserve Policy Intelligence")
    st.caption("An interactive, observational study of FOMC language and market reactions")
    try:
        data = _load_data(str(PROCESSED_DIR))
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Run the documented data, NLP, event, and research pipelines first.")
        st.stop()

    page = st.sidebar.radio(
        "Explore",
        [
            "Overview",
            "Announcement explorer",
            "Policy trends",
            "Market reactions",
            "Statistical results",
            "Methodology",
        ],
    )
    filtered_events = _sidebar_filters(data.events)
    if filtered_events.empty:
        st.warning("No events match the selected filters.")
        st.stop()

    pages = {
        "Overview": lambda: _overview(data, filtered_events),
        "Announcement explorer": lambda: _announcement_explorer(data, filtered_events),
        "Policy trends": lambda: _policy_trends(filtered_events),
        "Market reactions": lambda: _market_reactions(filtered_events),
        "Statistical results": lambda: _statistical_results(data),
        "Methodology": _methodology,
    }
    pages[page]()


if __name__ == "__main__":
    main()
