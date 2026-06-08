import random
import time

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from src.dashboard.data_access import (
    filter_news_data,
    get_summary_metrics,
    get_top_records,
    get_year_range,
    load_dashboard_data,
    normalize_dashboard_dataframe,
)


_DF = load_dashboard_data()
YEAR_MIN, YEAR_MAX = get_year_range(_DF)

DARK_TEMPLATE = "plotly_dark"
CHART_BG = "#111827"
PAPER_BG = "#111827"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f8fafc"
MUTED_COLOR = "#94a3b8"

_MARGIN = dict(l=20, r=20, t=55, b=35)


def register_callbacks(app):
    @app.callback(
        Output("category-filter", "value"),
        Output("document-type-filter", "value"),
        Output("year-range-filter", "value"),
        Output("search-input", "value"),
        Input("reset-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        return [], [], [YEAR_MIN, YEAR_MAX], ""

    @app.callback(
        Output("kpi-total-records", "children"),
        Output("kpi-category-count", "children"),
        Output("kpi-document-type-count", "children"),
        Output("kpi-avg-rating", "children"),
        Output("kpi-avg-popularity", "children"),
        Output("kpi-avg-content-length", "children"),
        Output("filter-summary", "children"),
        Output("top-records-table", "children"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_kpis_and_table(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)
        metrics = get_summary_metrics(df)

        summary = (
            f"Showing {metrics['total_records']:,} records "
            f"across {metrics['category_count']} categories "
            f"and {metrics['document_type_count']} document types."
        )

        table = _build_records_table(df)

        return (
            f"{metrics['total_records']:,}",
            str(metrics["category_count"]),
            str(metrics["document_type_count"]),
            f"{metrics['avg_rating']:.2f}",
            f"{metrics['avg_popularity']:.2f}",
            f"{metrics['avg_content_length']:.2f}",
            summary,
            table,
        )

    @app.callback(
        Output("top-records-chart", "figure"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_top_records_chart(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)

        if df.empty:
            return _empty_figure("No records match the selected filters.")

        top_df = get_top_records(df, n=12)

        if top_df.empty:
            return _empty_figure("No popularity data available.")

        top_df = top_df.copy()
        top_df["title_short"] = top_df["title"].astype(str).str.slice(0, 70)
        top_df.loc[top_df["title"].astype(str).str.len() > 70, "title_short"] += "..."

        fig = px.bar(
            top_df.sort_values("popularity", ascending=True),
            x="popularity",
            y="title_short",
            orientation="h",
            color="rating_score",
            color_continuous_scale="Blues",
            hover_name="title",
            hover_data={
                "category": True,
                "document_type": True,
                "published_year": True,
                "rating_score": ":.2f",
                "popularity": ":.2f",
                "content_length": True,
                "title_short": False,
            },
            labels={
                "popularity": "Popularity",
                "title_short": "",
                "rating_score": "Rating",
            },
            title="Top News Records by Popularity",
            template=DARK_TEMPLATE,
        )

        fig.update_traces(textposition="outside")
        _style_figure(fig, height=420)
        fig.update_layout(coloraxis_colorbar=dict(title="Rating"))

        return fig

    @app.callback(
        Output("rating-distribution-chart", "figure"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_rating_distribution_chart(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)

        if df.empty:
            return _empty_figure("No records match the selected filters.")

        plot_df = df.dropna(subset=["rating_score"]).copy()

        if plot_df.empty:
            return _empty_figure("No rating score data available.")

        fig = px.histogram(
            plot_df,
            x="rating_score",
            color="category",
            nbins=30,
            marginal="box",
            opacity=0.78,
            hover_data={
                "category": True,
                "document_type": True,
                "published_year": True,
                "popularity": ":.2f",
                "content_length": True,
            },
            labels={
                "rating_score": "Rating Score",
                "count": "Record Count",
                "category": "Category",
            },
            title="Rating Score Distribution",
            template=DARK_TEMPLATE,
        )

        _style_figure(fig, height=420)
        fig.update_layout(bargap=0.05)

        return fig

    @app.callback(
        Output("popularity-content-chart", "figure"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_popularity_content_chart(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)

        if df.empty:
            return _empty_figure("No records match the selected filters.")

        plot_df = df.dropna(subset=["content_length", "popularity", "rating_score"]).copy()

        if plot_df.empty:
            return _empty_figure("No content length or popularity data available.")

        plot_df["rating_size"] = plot_df["rating_score"].clip(lower=0.2, upper=10)

        fig = px.scatter(
            plot_df,
            x="content_length",
            y="popularity",
            color="category",
            size="rating_size",
            hover_name="title",
            hover_data={
                "document_type": True,
                "source_name": True,
                "published_year": True,
                "rating_score": ":.2f",
                "content_length": True,
                "popularity": ":.2f",
                "rating_size": False,
            },
            labels={
                "content_length": "Content Length",
                "popularity": "Popularity",
                "category": "Category",
            },
            title="Popularity vs Content Length",
            template=DARK_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )

        _style_figure(fig, height=420)

        return fig

    @app.callback(
        Output("year-trend-chart", "figure"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_year_trend_chart(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)

        if df.empty:
            return _empty_figure("No records match the selected filters.")

        plot_df = df[df["published_year"] > 0].copy()

        if plot_df.empty:
            return _empty_figure("No valid published year data available.")

        yearly = (
            plot_df.groupby("published_year")
            .agg(
                record_count=("record_id", "count"),
                avg_rating=("rating_score", "mean"),
                avg_popularity=("popularity", "mean"),
            )
            .reset_index()
            .sort_values("published_year")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=yearly["published_year"],
                y=yearly["record_count"],
                name="Record Count",
                marker_color="#38bdf8",
                opacity=0.75,
                hovertemplate="Year: %{x}<br>Records: %{y}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=yearly["published_year"],
                y=yearly["avg_rating"],
                name="Average Rating",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#22c55e", width=3),
                marker=dict(size=8),
                hovertemplate="Year: %{x}<br>Average rating: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="News Records Over Time",
            template=DARK_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=TEXT_COLOR),
            margin=_MARGIN,
            height=420,
            xaxis=dict(title="Published Year", gridcolor=GRID_COLOR),
            yaxis=dict(title="Record Count", gridcolor=GRID_COLOR),
            yaxis2=dict(
                title="Average Rating",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        return fig

    @app.callback(
        Output("document-type-chart", "figure"),
        Input("category-filter", "value"),
        Input("document-type-filter", "value"),
        Input("year-range-filter", "value"),
        Input("search-input", "value"),
    )
    def update_document_type_chart(categories, document_types, year_range, search_text):
        df = _get_filtered_df(categories, document_types, year_range, search_text)

        if df.empty:
            return _empty_figure("No records match the selected filters.")

        grouped = (
            df.groupby("document_type")
            .agg(
                record_count=("record_id", "count"),
                avg_rating=("rating_score", "mean"),
                avg_popularity=("popularity", "mean"),
            )
            .reset_index()
            .sort_values("record_count", ascending=False)
            .head(12)
            .sort_values("record_count", ascending=True)
        )

        if grouped.empty:
            return _empty_figure("No document type data available.")

        fig = px.bar(
            grouped,
            x="record_count",
            y="document_type",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Viridis",
            hover_name="document_type",
            hover_data={
                "record_count": True,
                "avg_rating": ":.2f",
                "avg_popularity": ":.2f",
            },
            labels={
                "record_count": "Record Count",
                "document_type": "Document Type",
                "avg_rating": "Avg Rating",
            },
            title="Document Type Breakdown",
            template=DARK_TEMPLATE,
        )

        _style_figure(fig, height=420)
        fig.update_layout(coloraxis_colorbar=dict(title="Avg Rating"))

        return fig

    @app.callback(
        Output("live-data-store", "data"),
        Output("live-ticker-chart", "figure"),
        Input("live-interval", "n_intervals"),
        State("live-data-store", "data"),
    )
    def update_live_ticker(n_intervals, live_data):
        live_data = live_data or []

        base_volume = max(8, int(len(_DF) / 150))
        simulated_records = max(0, int(random.gauss(base_volume, 3)))
        simulated_alerts = max(0, int(random.gauss(2, 1)))
        simulated_score = round(random.uniform(0.2, 10.0), 2)

        live_data.append(
            {
                "tick": int(n_intervals or 0),
                "time": time.strftime("%H:%M:%S"),
                "records": simulated_records,
                "alerts": simulated_alerts,
                "score": simulated_score,
            }
        )

        live_data = live_data[-60:]
        live_df = pd.DataFrame(live_data)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=live_df["tick"],
                y=live_df["records"],
                mode="lines+markers",
                name="Incoming Records",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.12)",
                customdata=live_df[["time", "alerts", "score"]],
                hovertemplate=(
                    "Tick: %{x}<br>"
                    "Time: %{customdata[0]}<br>"
                    "Incoming records: %{y}<br>"
                    "Simulated alerts: %{customdata[1]}<br>"
                    "Signal score: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=live_df["tick"],
                y=live_df["alerts"],
                mode="lines+markers",
                name="Alerts",
                line=dict(color="#f97316", width=2),
                marker=dict(size=5),
                yaxis="y2",
                hovertemplate="Tick: %{x}<br>Alerts: %{y}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Simulated Live News Monitoring Stream",
            template=DARK_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=TEXT_COLOR),
            margin=_MARGIN,
            height=420,
            xaxis=dict(title="Interval Tick", gridcolor=GRID_COLOR),
            yaxis=dict(title="Incoming Records", gridcolor=GRID_COLOR),
            yaxis2=dict(
                title="Alerts",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        return live_data, fig


def _get_filtered_df(categories, document_types, year_range, search_text):
    categories = categories or []
    document_types = document_types or []

    return filter_news_data(
        _DF,
        categories=categories,
        document_types=document_types,
        year_range=year_range,
        search_text=search_text,
    )


def _build_records_table(df):
    if df.empty:
        return dbc.Alert("No records match the selected filters.", color="warning", className="mb-0")

    top_df = get_top_records(df, n=10).copy()

    columns = [
        "title",
        "category",
        "document_type",
        "published_year",
        "rating_score",
        "popularity",
    ]

    existing_columns = [col for col in columns if col in top_df.columns]
    table_df = top_df[existing_columns].copy()

    if "title" in table_df.columns:
        table_df["title"] = table_df["title"].astype(str).str.slice(0, 95)
        table_df.loc[top_df["title"].astype(str).str.len() > 95, "title"] += "..."

    if "rating_score" in table_df.columns:
        table_df["rating_score"] = table_df["rating_score"].map(lambda value: f"{float(value):.2f}")

    if "popularity" in table_df.columns:
        table_df["popularity"] = table_df["popularity"].map(lambda value: f"{float(value):.2f}")

    header = html.Thead(
        html.Tr(
            [
                html.Th(col.replace("_", " ").title())
                for col in existing_columns
            ]
        )
    )

    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(row[col])
                    for col in existing_columns
                ]
            )
            for _, row in table_df.iterrows()
        ]
    )

    return dbc.Table(
        [header, body],
        bordered=False,
        hover=True,
        responsive=True,
        striped=True,
        color="dark",
        className="mb-0",
    )


def _style_figure(fig, height=420):
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR),
        margin=_MARGIN,
        height=height,
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color=TEXT_COLOR),
        ),
    )

    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

    return fig


def _empty_figure(message: str):
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=MUTED_COLOR),
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR),
        margin=_MARGIN,
        height=420,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig