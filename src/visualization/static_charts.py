"""
src/visualization/static_charts.py

Static visualization functions for the News Media Monitoring Pipeline.

This module creates 8 static charts using matplotlib and seaborn.
Each chart:
- accepts a pandas DataFrame and output directory
- uses matplotlib's object-oriented API
- saves both PNG at 300 dpi and PDF
"""

from pathlib import Path
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


def _ensure_output_dir(output_dir) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _save(fig, output_dir, filename: str) -> dict:
    """
    Save a matplotlib figure as PNG and PDF.
    """
    output_path = _ensure_output_dir(output_dir)

    png_path = output_path / f"{filename}.png"
    pdf_path = output_path / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    logger.info("Saved static chart: %s and %s", png_path, pdf_path)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def _prepare_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the cleaned news dataset for plotting.
    """
    data = df.copy()

    text_defaults = {
        "category": "unknown",
        "document_type": "unknown",
        "language": "unknown",
        "source_name": "unknown",
        "title": "Untitled",
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
        "vote_average",
        "vote_count",
        "published_year",
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


def _top_categories(data: pd.DataFrame, top_n: int = 8) -> list:
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


def plot_top_categories_bar(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    Horizontal bar chart showing the top news categories by record count.
    """
    data = _prepare_news_df(df)

    counts = (
        data["category"]
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(counts.index, counts.values)
    ax.set_title("Top 10 News Categories by Record Count", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Category")

    for i, value in enumerate(counts.values):
        ax.text(value + max(counts.values) * 0.01, i, str(value), va="center", fontsize=9)

    fig.tight_layout()

    paths = _save(fig, output_dir, "top_categories_bar")
    return fig, paths


def plot_document_type_counts(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    Bar chart showing record counts by document type.
    """
    data = _prepare_news_df(df)

    counts = (
        data["document_type"]
        .value_counts()
        .head(12)
        .reset_index()
    )

    counts.columns = ["document_type", "count"]

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=counts,
        x="count",
        y="document_type",
        hue="document_type",
        legend=False,
        ax=ax,
        palette="viridis",
    )

    ax.set_title("Document Types in the News Media Pipeline", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Document Type")

    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=3)

    fig.tight_layout()

    paths = _save(fig, output_dir, "document_type_counts")
    return fig, paths


def plot_rating_distribution(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    Histogram showing the distribution of rating_score values.
    """
    data = _prepare_news_df(df)

    plot_data = data.dropna(subset=["rating_score"])

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.histplot(
        data=plot_data,
        x="rating_score",
        bins=25,
        kde=True,
        ax=ax,
    )

    ax.set_title("Distribution of News Rating Scores", fontsize=14, weight="bold")
    ax.set_xlabel("Rating Score")
    ax.set_ylabel("Record Count")

    fig.tight_layout()

    paths = _save(fig, output_dir, "rating_score_distribution")
    return fig, paths


def plot_rating_by_category_boxplot(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
    top_n: int = 8,
):
    """
    Boxplot comparing rating_score distributions across top categories.
    """
    data = _prepare_news_df(df)

    top_categories = _top_categories(data, top_n=top_n)

    plot_data = data[
        data["category"].isin(top_categories)
    ].dropna(subset=["rating_score", "category"])

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=plot_data,
        x="category",
        y="rating_score",
        hue="category",
        legend=False,
        ax=ax,
        palette="Set2",
    )

    ax.set_title("Rating Score Distribution by News Category", fontsize=14, weight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Rating Score")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    paths = _save(fig, output_dir, "rating_by_category_boxplot")
    return fig, paths


def plot_popularity_vs_content_length_scatter(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
    top_n: int = 8,
):
    """
    Scatter plot showing the relationship between article length and popularity.
    """
    data = _prepare_news_df(df)

    top_categories = _top_categories(data, top_n=top_n)

    plot_data = data[
        data["category"].isin(top_categories)
    ].dropna(subset=["content_length", "popularity", "category"])

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=plot_data,
        x="content_length",
        y="popularity",
        hue="category",
        size="rating_score" if "rating_score" in plot_data.columns else None,
        sizes=(30, 180),
        alpha=0.75,
        ax=ax,
        palette="tab10",
    )

    ax.set_title("Popularity vs Content Length", fontsize=14, weight="bold")
    ax.set_xlabel("Content Length")
    ax.set_ylabel("Popularity")

    if ax.legend_:
        ax.legend(
            title="Category / Rating",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

    fig.tight_layout()

    paths = _save(fig, output_dir, "popularity_vs_content_length_scatter")
    return fig, paths


def plot_average_rating_over_years(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    Dual-axis chart:
    - bars show number of records per year
    - line shows average rating_score per year
    """
    data = _prepare_news_df(df)

    plot_data = data.dropna(subset=["published_year"]).copy()
    plot_data["published_year"] = plot_data["published_year"].astype(int)

    yearly = (
        plot_data
        .groupby("published_year")
        .agg(
            record_count=("record_id", "count"),
            mean_rating=("rating_score", "mean"),
        )
        .reset_index()
        .sort_values("published_year")
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.bar(
        yearly["published_year"],
        yearly["record_count"],
        alpha=0.65,
        label="Record Count",
    )

    ax1.set_xlabel("Published Year")
    ax1.set_ylabel("Number of Records")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()

    ax2.plot(
        yearly["published_year"],
        yearly["mean_rating"],
        marker="o",
        linewidth=2,
        label="Average Rating",
    )

    ax2.set_ylabel("Average Rating Score")

    ax1.set_title("News Volume and Average Rating Over Time", fontsize=14, weight="bold")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()

    paths = _save(fig, output_dir, "average_rating_over_years")
    return fig, paths


def plot_numeric_correlation_heatmap(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    Heatmap of correlations among numeric columns.
    """
    data = _prepare_news_df(df)

    candidate_cols = [
        "rating_score",
        "popularity",
        "content_length",
        "title_length",
        "vote_average",
        "vote_count",
        "published_year",
        "wins",
        "losses",
    ]

    numeric_cols = [
        col for col in candidate_cols
        if col in data.columns and data[col].notna().sum() > 1
    ]

    corr = data[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Correlation Heatmap of Numeric News Features", fontsize=14, weight="bold")

    fig.tight_layout()

    paths = _save(fig, output_dir, "numeric_correlation_heatmap")
    return fig, paths


def plot_news_dashboard_subplots(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
):
    """
    2x2 static dashboard combining the most useful news monitoring views.
    """
    data = _prepare_news_df(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1, ax2, ax3, ax4 = axes.flatten()

    category_counts = data["category"].value_counts().head(8).sort_values(ascending=True)

    ax1.barh(category_counts.index, category_counts.values)
    ax1.set_title("Top Categories")
    ax1.set_xlabel("Records")
    ax1.set_ylabel("Category")

    rating_data = data.dropna(subset=["rating_score"])

    sns.histplot(
        data=rating_data,
        x="rating_score",
        bins=20,
        kde=True,
        ax=ax2,
    )

    ax2.set_title("Rating Score Distribution")
    ax2.set_xlabel("Rating Score")
    ax2.set_ylabel("Count")

    scatter_data = data.dropna(subset=["content_length", "popularity"])

    sns.scatterplot(
        data=scatter_data,
        x="content_length",
        y="popularity",
        hue="category",
        legend=False,
        alpha=0.65,
        ax=ax3,
    )

    ax3.set_title("Popularity vs Content Length")
    ax3.set_xlabel("Content Length")
    ax3.set_ylabel("Popularity")

    year_data = data.dropna(subset=["published_year"]).copy()

    if not year_data.empty:
        year_data["published_year"] = year_data["published_year"].astype(int)

        yearly_counts = (
            year_data
            .groupby("published_year")
            .size()
            .reset_index(name="count")
            .sort_values("published_year")
        )

        ax4.plot(
            yearly_counts["published_year"],
            yearly_counts["count"],
            marker="o",
            linewidth=2,
        )

        ax4.set_title("Records Over Time")
        ax4.set_xlabel("Published Year")
        ax4.set_ylabel("Record Count")
        ax4.tick_params(axis="x", rotation=45)
    else:
        ax4.text(0.5, 0.5, "No year data available", ha="center", va="center")
        ax4.set_axis_off()

    fig.suptitle("News Media Monitoring Static Dashboard", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    paths = _save(fig, output_dir, "news_dashboard_subplots")
    return fig, paths


STATIC_CHART_FUNCTIONS = [
    plot_top_categories_bar,
    plot_document_type_counts,
    plot_rating_distribution,
    plot_rating_by_category_boxplot,
    plot_popularity_vs_content_length_scatter,
    plot_average_rating_over_years,
    plot_numeric_correlation_heatmap,
    plot_news_dashboard_subplots,
]


def generate_all_static_charts(
    df: pd.DataFrame,
    output_dir="outputs/visualizations/static",
) -> list[dict]:
    """
    Generate all 8 static charts and return saved file paths.
    """
    saved_paths = []

    for chart_func in STATIC_CHART_FUNCTIONS:
        fig, paths = chart_func(df, output_dir=output_dir)
        saved_paths.append(paths)
        plt.close(fig)

    return saved_paths