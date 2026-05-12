"""
src/analytics/aggregator.py

GroupBy and aggregation utilities for Lab 10 - News Media Monitoring Pipeline.

Movie example:
- genre summaries
- yearly trends
- top movies per genre by revenue

News equivalent:
- category summaries
- yearly article trends
- top articles per category by estimated_value / engagement_score
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.pivot_builder import add_primary_category, add_analysis_year

logger = logging.getLogger(__name__)


def category_summary(
    df: pd.DataFrame,
    category_col: str = "primary_category",
) -> pd.DataFrame:
    """
    Compute grouped category-level summary.

    Uses named aggregations with count, mean, sum, median, min, max.
    This satisfies the Lab 10 groupby requirement.
    """
    df = df.copy()

    if category_col not in df.columns:
        df = add_primary_category(df, output_col=category_col)

    numeric_cols = [
        "rating_score",
        "popularity",
        "content_length",
        "engagement_score",
        "estimated_value",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    summary = (
        df.groupby(category_col)
        .agg(
            article_count=("record_id", "count"),
            avg_rating_score=("rating_score", "mean"),
            median_rating_score=("rating_score", "median"),
            total_estimated_value=("estimated_value", "sum"),
            avg_estimated_value=("estimated_value", "mean"),
            median_estimated_value=("estimated_value", "median"),
            max_estimated_value=("estimated_value", "max"),
            total_engagement_score=("engagement_score", "sum"),
            avg_engagement_score=("engagement_score", "mean"),
            avg_popularity=("popularity", "mean"),
            total_content_length=("content_length", "sum"),
            median_content_length=("content_length", "median"),
        )
        .reset_index()
        .sort_values("total_estimated_value", ascending=False)
    )

    logger.info("category_summary: created summary with shape=%s", summary.shape)

    return summary


def document_type_summary(
    df: pd.DataFrame,
    document_col: str = "document_type_mysql",
) -> pd.DataFrame:
    """
    Compute grouped document-type summary.

    Falls back to document_type if document_type_mysql is not available.
    """
    df = df.copy()

    if document_col not in df.columns:
        document_col = "document_type"

    if document_col not in df.columns:
        df[document_col] = "unknown"

    for col in ["rating_score", "estimated_value", "engagement_score", "content_length"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    summary = (
        df.groupby(document_col)
        .agg(
            article_count=("record_id", "count"),
            avg_rating_score=("rating_score", "mean"),
            median_rating_score=("rating_score", "median"),
            total_estimated_value=("estimated_value", "sum"),
            avg_estimated_value=("estimated_value", "mean"),
            total_engagement_score=("engagement_score", "sum"),
            avg_content_length=("content_length", "mean"),
        )
        .reset_index()
        .sort_values("article_count", ascending=False)
    )

    logger.info("document_type_summary: created summary with shape=%s", summary.shape)

    return summary


def yearly_trends(
    df: pd.DataFrame,
    year_col: str = "analysis_year",
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Compute yearly news trends.

    Includes article count, total estimated value, mean rating, median rating,
    and total engagement.
    """
    df = df.copy()

    if year_col not in df.columns:
        df = add_analysis_year(df, output_col=year_col)

    for col in ["rating_score", "estimated_value", "engagement_score", "content_length"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    valid = df.dropna(subset=[year_col]).copy()
    valid[year_col] = valid[year_col].astype(int)

    if start_year is not None:
        valid = valid[valid[year_col] >= start_year]

    if end_year is not None:
        valid = valid[valid[year_col] <= end_year]

    trends = (
        valid.groupby(year_col)
        .agg(
            article_count=("record_id", "count"),
            avg_rating_score=("rating_score", "mean"),
            median_rating_score=("rating_score", "median"),
            total_estimated_value=("estimated_value", "sum"),
            avg_estimated_value=("estimated_value", "mean"),
            total_engagement_score=("engagement_score", "sum"),
            total_content_length=("content_length", "sum"),
        )
        .reset_index()
        .sort_values(year_col)
    )

    logger.info("yearly_trends: created trends with shape=%s", trends.shape)

    return trends


def top_n_per_group(
    df: pd.DataFrame,
    group_col: str = "primary_category",
    sort_col: str = "estimated_value",
    n: int = 3,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return top N articles per group using groupby().apply().

    This satisfies the Lab 10 top N per group requirement.
    Compatible with pandas 3.x because it does not use include_groups=True.
    """
    df = df.copy()

    if group_col not in df.columns:
        df = add_primary_category(df, output_col=group_col)

    if sort_col not in df.columns:
        raise KeyError(f"Missing sort column: {sort_col}")

    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce").fillna(0)

    def pick_top(group: pd.DataFrame) -> pd.DataFrame:
        if ascending:
            return group.nsmallest(n, sort_col)
        return group.nlargest(n, sort_col)

    top_df = (
        df.groupby(group_col, group_keys=False)
        .apply(pick_top)
        .reset_index(drop=True)
    )

    display_cols = [
        "record_id",
        "title_mysql",
        "title_mongo",
        "title",
        group_col,
        "document_type_mysql",
        "document_type",
        "rating_score",
        "engagement_score",
        "estimated_value",
    ]

    existing_cols = [col for col in display_cols if col in top_df.columns]

    top_df = top_df[existing_cols].copy()

    logger.info(
        "top_n_per_group: group=%s sort=%s n=%d -> %d rows",
        group_col,
        sort_col,
        n,
        len(top_df),
    )

    return top_df


def add_group_average_columns(
    df: pd.DataFrame,
    group_col: str = "primary_category",
) -> pd.DataFrame:
    """
    Demonstrate groupby().transform() by adding same-shape group averages.
    """
    df = df.copy()

    if group_col not in df.columns:
        df = add_primary_category(df, output_col=group_col)

    for col in ["rating_score", "estimated_value"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["category_avg_rating_score"] = (
        df.groupby(group_col)["rating_score"].transform("mean")
    )

    df["category_avg_estimated_value"] = (
        df.groupby(group_col)["estimated_value"].transform("mean")
    )

    logger.info("add_group_average_columns: added transformed group averages")

    return df


def filter_large_categories(
    df: pd.DataFrame,
    group_col: str = "primary_category",
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Demonstrate groupby().filter() by keeping only groups with enough records.
    """
    df = df.copy()

    if group_col not in df.columns:
        df = add_primary_category(df, output_col=group_col)

    filtered = df.groupby(group_col).filter(lambda group: len(group) >= min_count)

    logger.info(
        "filter_large_categories: kept %d/%d rows using min_count=%d",
        len(filtered),
        len(df),
        min_count,
    )

    return filtered.reset_index(drop=True)


def save_yearly_trends_chart(
    trends_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/yearly_trends.png",
    year_col: str = "analysis_year",
) -> str:
    """
    Save a yearly trends chart.

    Shows article count and total estimated value over time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trends_df.empty:
        logger.warning("save_yearly_trends_chart skipped because trends_df is empty")
        return str(output_path)

    plt.figure(figsize=(10, 6))
    plt.plot(trends_df[year_col], trends_df["article_count"], marker="o", label="Article count")

    if "total_estimated_value" in trends_df.columns:
        scaled_value = trends_df["total_estimated_value"] / max(
            trends_df["total_estimated_value"].max(),
            1,
        )
        scaled_value = scaled_value * trends_df["article_count"].max()
        plt.plot(
            trends_df[year_col],
            scaled_value,
            marker="o",
            label="Total estimated value (scaled)",
        )

    plt.title("Yearly News Trends")
    plt.xlabel("Year")
    plt.ylabel("Count / Scaled Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved yearly trends chart to %s", output_path)

    return str(output_path)


def save_aggregation_outputs(
    category_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    top_df: pd.DataFrame,
    document_df: pd.DataFrame | None = None,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Save groupby outputs to CSV.

    Saves both news-specific and lab-compatible file names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_path = output_dir / "category_analysis.csv"
    genre_alias_path = output_dir / "genre_analysis.csv"
    yearly_path = output_dir / "yearly_trends.csv"
    top_path = output_dir / "top_articles_per_category.csv"

    category_df.to_csv(category_path, index=False)
    category_df.to_csv(genre_alias_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    top_df.to_csv(top_path, index=False)

    paths = {
        "category_analysis": str(category_path),
        "genre_analysis_alias": str(genre_alias_path),
        "yearly_trends": str(yearly_path),
        "top_articles": str(top_path),
    }

    if document_df is not None:
        document_path = output_dir / "document_type_analysis.csv"
        document_df.to_csv(document_path, index=False)
        paths["document_type_analysis"] = str(document_path)

    logger.info("Saved aggregation outputs to %s", output_dir)

    return paths