"""
src/visualization/interactive_charts.py

Interactive visualization module for the News Media Monitoring Pipeline.

Creates 5 Plotly charts and saves each as a self-contained HTML file.
"""

from pathlib import Path
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

INTERACTIVE_OUT = Path("outputs/visualizations/interactive")
TEMPLATE = "plotly_white"


def _save_html(fig: go.Figure, stem: str, out_dir: Path = INTERACTIVE_OUT) -> str:
    """
    Save a Plotly figure as a self-contained interactive HTML file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{stem}.html"

    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
    )

    logger.info("Saved interactive chart: %s", path)

    return str(path)


def _prepare_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare cleaned news data for interactive plotting.
    """
    data = df.copy()

    text_defaults = {
        "title": "Untitled",
        "category": "unknown",
        "document_type": "unknown",
        "language": "unknown",
        "source_name": "unknown",
        "url": "",
    }

    for col, default in text_defaults.items():
        if col in data.columns:
            data[col] = (
                data[col]
                .fillna(default)
                .astype(str)
                .str.strip()
                .replace("", default)
            )

    numeric_cols = [
        "rating_score",
        "popularity",
        "content_length",
        "title_length",
        "published_year",
        "vote_average",
        "vote_count",
        "year",
        "wins",
        "losses",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "published_year" not in data.columns and "year" in data.columns:
        data["published_year"] = data["year"]

    return data


def _top_categories(data: pd.DataFrame, top_n: int = 8) -> list[str]:
    if "category" not in data.columns:
        return []

    return (
        data["category"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )


def interactive_popularity_vs_content_length(
    df: pd.DataFrame,
    out_dir: Path = INTERACTIVE_OUT,
) -> str:
    """
    Interactive scatter plot:
    content_length vs popularity, colored by category.
    """
    data = _prepare_news_df(df)

    plot_data = data.dropna(
        subset=["content_length", "popularity", "rating_score"]
    ).copy()

    fig = px.scatter(
        plot_data,
        x="content_length",
        y="popularity",
        color="category",
        size="rating_score",
        hover_name="title",
        hover_data={
            "document_type": True,
            "source_name": True,
            "published_year": True,
            "rating_score": ":.2f",
            "language": True,
            "content_length": True,
            "popularity": ":.2f",
        },
        labels={
            "content_length": "Content Length",
            "popularity": "Popularity",
            "category": "Category",
            "rating_score": "Rating Score",
        },
        title="Popularity vs Content Length – Interactive News Explorer",
        template=TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    fig.update_layout(
        legend_title="Category",
        font=dict(family="Inter", size=13),
        height=600,
    )

    return _save_html(fig, "popularity_vs_content_length_interactive", out_dir)


def interactive_top_categories_bar(
    df: pd.DataFrame,
    n: int = 10,
    out_dir: Path = INTERACTIVE_OUT,
) -> str:
    """
    Interactive horizontal bar chart showing top categories by record count.
    """
    data = _prepare_news_df(df)

    counts = (
        data
        .groupby("category")
        .agg(
            record_count=("record_id", "count"),
            avg_rating=("rating_score", "mean"),
            avg_popularity=("popularity", "mean"),
            avg_content_length=("content_length", "mean"),
        )
        .reset_index()
        .sort_values("record_count", ascending=False)
        .head(n)
        .sort_values("record_count", ascending=True)
    )

    fig = px.bar(
        counts,
        x="record_count",
        y="category",
        orientation="h",
        color="avg_rating",
        hover_name="category",
        hover_data={
            "record_count": True,
            "avg_rating": ":.2f",
            "avg_popularity": ":.2f",
            "avg_content_length": ":.1f",
        },
        labels={
            "record_count": "Number of Records",
            "category": "Category",
            "avg_rating": "Average Rating",
        },
        title=f"Top {n} News Categories by Record Count",
        template=TEMPLATE,
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        font=dict(family="Inter", size=13),
        height=520,
    )

    return _save_html(fig, "top_categories_interactive_bar", out_dir)


def interactive_records_over_years(
    df: pd.DataFrame,
    out_dir: Path = INTERACTIVE_OUT,
) -> str:
    """
    Interactive line chart showing record count and average rating by year.
    """
    data = _prepare_news_df(df)

    plot_data = data.dropna(subset=["published_year"]).copy()

    if plot_data.empty:
        plot_data = pd.DataFrame({
            "published_year": [0],
            "record_count": [0],
            "avg_rating": [0],
            "avg_popularity": [0],
        })
    else:
        plot_data["published_year"] = plot_data["published_year"].astype(int)

        plot_data = (
            plot_data
            .groupby("published_year")
            .agg(
                record_count=("record_id", "count"),
                avg_rating=("rating_score", "mean"),
                avg_popularity=("popularity", "mean"),
            )
            .reset_index()
            .sort_values("published_year")
        )

    fig = px.line(
        plot_data,
        x="published_year",
        y="record_count",
        markers=True,
        hover_data={
            "record_count": True,
            "avg_rating": ":.2f",
            "avg_popularity": ":.2f",
        },
        labels={
            "published_year": "Published Year",
            "record_count": "Number of Records",
            "avg_rating": "Average Rating",
            "avg_popularity": "Average Popularity",
        },
        title="News Records Over Time",
        template=TEMPLATE,
    )

    fig.update_traces(
        line_width=2.5,
        marker=dict(size=8),
    )

    fig.update_layout(
        font=dict(family="Inter", size=13),
        height=480,
    )

    return _save_html(fig, "records_over_years_interactive_line", out_dir)


def interactive_rating_by_category_boxplot(
    df: pd.DataFrame,
    out_dir: Path = INTERACTIVE_OUT,
) -> str:
    """
    Interactive box plot showing rating score distribution by category.
    """
    data = _prepare_news_df(df)

    top_categories = _top_categories(data, top_n=8)

    plot_data = data[
        data["category"].isin(top_categories)
    ].dropna(subset=["category", "rating_score"]).copy()

    category_order = (
        plot_data
        .groupby("category")["rating_score"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig = px.box(
        plot_data,
        x="category",
        y="rating_score",
        color="category",
        category_orders={"category": category_order},
        hover_name="title",
        hover_data={
            "document_type": True,
            "source_name": True,
            "published_year": True,
            "popularity": ":.2f",
            "content_length": True,
        },
        labels={
            "category": "Category",
            "rating_score": "Rating Score",
        },
        title="Rating Score Distribution by News Category",
        template=TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    fig.update_layout(
        showlegend=False,
        font=dict(family="Inter", size=13),
        height=550,
    )

    return _save_html(fig, "rating_by_category_interactive_boxplot", out_dir)


def interactive_news_multi_layout(
    df: pd.DataFrame,
    out_dir: Path = INTERACTIVE_OUT,
) -> str:
    """
    2x2 interactive Plotly dashboard for the news media monitoring dataset.
    """
    data = _prepare_news_df(df)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top Categories by Record Count",
            "Rating Score Distribution",
            "Records by Document Type",
            "Popularity vs Content Length",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    category_counts = (
        data["category"]
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )

    fig.add_trace(
        go.Bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation="h",
            marker_color="#1a6faf",
            name="Category Count",
            hovertemplate="Category: %{y}<br>Records: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=data["rating_score"].dropna(),
            nbinsx=25,
            marker_color="#2ca02c",
            name="Rating Distribution",
            hovertemplate="Rating: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    doc_counts = (
        data["document_type"]
        .value_counts()
        .head(10)
    )

    fig.add_trace(
        go.Bar(
            x=doc_counts.index,
            y=doc_counts.values,
            marker_color="#ff7f0e",
            name="Document Type Count",
            hovertemplate="Document type: %{x}<br>Records: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    scatter_data = data.dropna(
        subset=["content_length", "popularity", "rating_score"]
    ).copy()

    fig.add_trace(
        go.Scatter(
            x=scatter_data["content_length"],
            y=scatter_data["popularity"],
            mode="markers",
            marker=dict(
                color=scatter_data["rating_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Rating", x=1.02, len=0.45, y=0.16),
                size=8,
                opacity=0.7,
            ),
            text=scatter_data["title"],
            customdata=scatter_data[
                ["category", "document_type", "published_year"]
            ].fillna("unknown"),
            name="News Records",
            hovertemplate=(
                "%{text}<br>"
                "Category: %{customdata[0]}<br>"
                "Document type: %{customdata[1]}<br>"
                "Year: %{customdata[2]}<br>"
                "Content length: %{x}<br>"
                "Popularity: %{y}<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text="News Media Monitoring Interactive Dashboard",
        title_font=dict(size=18),
        template=TEMPLATE,
        height=760,
        width=1150,
        showlegend=False,
        font=dict(family="Inter", size=11),
    )

    fig.update_xaxes(title_text="Records", row=1, col=1)
    fig.update_xaxes(title_text="Rating Score", row=1, col=2)
    fig.update_xaxes(title_text="Document Type", row=2, col=1)
    fig.update_xaxes(title_text="Content Length", row=2, col=2)

    fig.update_yaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Records", row=2, col=1)
    fig.update_yaxes(title_text="Popularity", row=2, col=2)

    return _save_html(fig, "interactive_news_dashboard", out_dir)


INTERACTIVE_CHART_FUNCTIONS = [
    interactive_popularity_vs_content_length,
    interactive_top_categories_bar,
    interactive_records_over_years,
    interactive_rating_by_category_boxplot,
    interactive_news_multi_layout,
]


def generate_all_interactive_charts(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/interactive",
) -> list[str]:
    """
    Generate all 5 interactive charts and return saved HTML paths.
    """
    output_dir = Path(output_dir)

    saved_paths = []

    for chart_func in INTERACTIVE_CHART_FUNCTIONS:
        path = chart_func(df, out_dir=output_dir)
        saved_paths.append(path)

    return saved_paths