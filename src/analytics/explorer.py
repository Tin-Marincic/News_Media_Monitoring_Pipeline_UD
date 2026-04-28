"""
src/analytics/explorer.py

Exploratory data analysis functions for the News Media Monitoring Pipeline.

This module:
- inspects DataFrame structure
- prints df.info()
- generates descriptive statistics
- produces value_counts and nunique reports
- extracts published_year from published_date / release_date / year
- saves distribution charts for rating_score, popularity, language,
  published year, category, document type, and source.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def inspect_shape(df: pd.DataFrame) -> dict:
    """
    Return basic structure information about the DataFrame.
    """
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "cells": df.size,
        "column_names": df.columns.tolist(),
    }

    logger.info(
        "DataFrame shape inspected: %d rows x %d columns",
        info["rows"],
        info["columns"],
    )

    return info


def print_info(df: pd.DataFrame) -> None:
    """
    Print df.info() to show column names, non-null values, dtypes,
    and memory usage.
    """
    logger.info("Printing DataFrame info")
    df.info(memory_usage="deep")


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns.
    """
    logger.info("Generating numeric descriptive statistics")
    return df.describe()


def describe_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for all available columns.
    Useful for both numeric and categorical overview.
    """
    logger.info("Generating full descriptive statistics")
    return df.describe(include="all")


def value_counts_report(
    df: pd.DataFrame,
    cols: Optional[list] = None,
    top_n: int = 15,
) -> dict:
    """
    Generate value_counts and nunique report for selected categorical columns.

    If cols is not provided, this function automatically checks common
    News Media Monitoring Pipeline columns.
    """
    if cols is None:
        possible_cols = [
            "document_type",
            "category",
            "source_name",
            "source_path",
            "language",
            "extraction_library",
            "file_name",
            "original_language",
            "genres",
            "status",
        ]

        cols = [col for col in possible_cols if col in df.columns]

    report = {}

    for col in cols:
        counts = df[col].value_counts(dropna=False).head(top_n)
        n_unique = df[col].nunique(dropna=True)

        report[col] = {
            "counts": counts,
            "nunique": n_unique,
        }

        logger.info(
            "Value counts generated for column '%s': %d unique values",
            col,
            n_unique,
        )

    return report


def nunique_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return number of unique values for every column.
    """
    logger.info("Generating nunique report for all columns")

    result = pd.DataFrame({
        "column": df.columns,
        "nunique": [df[col].nunique(dropna=True) for col in df.columns],
        "dtype": [str(df[col].dtype) for col in df.columns],
    })

    return result.sort_values("nunique", ascending=False).reset_index(drop=True)


def extract_published_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add or update published_year column.

    Priority:
    1. published_date
    2. publishedAt
    3. release_date
    4. fetched_at
    5. extraction_timestamp
    6. year
    """
    logger.info("Extracting published_year")

    df = df.copy()

    date_candidates = [
        "published_date",
        "publishedAt",
        "release_date",
        "fetched_at",
        "extraction_timestamp",
    ]

    for col in date_candidates:
        if col in df.columns:
            parsed_dates = pd.to_datetime(df[col], errors="coerce")
            if parsed_dates.notna().sum() > 0:
                df["published_year"] = parsed_dates.dt.year
                logger.info(
                    "published_year extracted from %s: %d non-null values",
                    col,
                    df["published_year"].notna().sum(),
                )

                # Keep compatibility with older Lab 8 naming
                df["release_year"] = df["published_year"]
                return df

    if "year" in df.columns:
        df["published_year"] = pd.to_numeric(df["year"], errors="coerce")
        df["release_year"] = df["published_year"]

        logger.info(
            "published_year created from year column: %d non-null values",
            df["published_year"].notna().sum(),
        )
    else:
        logger.warning("No date/year column found; published_year not created")

    return df


def extract_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias for the professor's Lab 8 naming.
    In the news pipeline, this extracts published_year and mirrors it
    into release_year.
    """
    logger.info("extract_release_year() called; using extract_published_year() for news data")
    return extract_published_year(df)


def _save_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 30,
    log_y: bool = False,
) -> Optional[str]:
    """
    Save a histogram for a numeric column.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if clean_series.empty:
        logger.warning("Skipping histogram '%s' because the series is empty", title)
        return None

    plt.figure(figsize=(9, 5))
    clean_series.plot(kind="hist", bins=bins, edgecolor="white")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    if log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved histogram chart: %s", output_path)

    return str(output_path)


def _save_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    top_n: int = 10,
) -> Optional[str]:
    """
    Save a bar chart for a categorical column.
    """
    counts = series.value_counts(dropna=False).head(top_n)

    if counts.empty:
        logger.warning("Skipping bar chart '%s' because the series is empty", title)
        return None

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", edgecolor="white")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved bar chart: %s", output_path)

    return str(output_path)


def plot_distributions(
    df: pd.DataFrame,
    output_dir: str = "data/processed/analytics/charts",
) -> list:
    """
    Save separate charts for key News Media Monitoring Pipeline distributions:
    - rating_score
    - popularity / content length / mentions
    - language
    - published_year
    - category
    - document_type
    - source_name

    Returns a list of saved chart paths.
    """
    logger.info("Creating news EDA distribution charts")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = extract_published_year(df)

    saved_charts = []

    # Rating score distribution
    if "rating_score" in df.columns:
        chart = _save_histogram(
            series=df["rating_score"],
            title="News Rating Score Distribution",
            xlabel="Rating Score",
            output_path=output_path / "rating_score_distribution.png",
            bins=20,
        )
        if chart:
            saved_charts.append(chart)

    # Backward-compatible chart name for older notebook/pipeline checks
    if "vote_average" in df.columns:
        chart = _save_histogram(
            series=df["vote_average"],
            title="Rating Distribution",
            xlabel="Rating / Vote Average",
            output_path=output_path / "rating_distribution.png",
            bins=20,
        )
        if chart:
            saved_charts.append(chart)

    # Popularity / content length / mentions distribution
    if "popularity" in df.columns:
        chart = _save_histogram(
            series=df["popularity"],
            title="News Popularity Distribution",
            xlabel="Popularity / Mentions / Content Length",
            output_path=output_path / "popularity_distribution.png",
            bins=20,
            log_y=True,
        )
        if chart:
            saved_charts.append(chart)

    if "mentions" in df.columns:
        chart = _save_histogram(
            series=df["mentions"],
            title="Mentions Distribution",
            xlabel="Mentions",
            output_path=output_path / "mentions_distribution.png",
            bins=20,
            log_y=True,
        )
        if chart:
            saved_charts.append(chart)

    if "content_length" in df.columns:
        chart = _save_histogram(
            series=df["content_length"],
            title="Content Length Distribution",
            xlabel="Content Length",
            output_path=output_path / "content_length_distribution.png",
            bins=30,
            log_y=True,
        )
        if chart:
            saved_charts.append(chart)

    # Language distribution
    language_col = None

    if "language" in df.columns:
        language_col = "language"
    elif "original_language" in df.columns:
        language_col = "original_language"

    if language_col:
        chart = _save_bar_chart(
            series=df[language_col],
            title="Top News Languages",
            xlabel="Language",
            ylabel="Record Count",
            output_path=output_path / "language_distribution.png",
            top_n=10,
        )
        if chart:
            saved_charts.append(chart)

    # Published year / release year distribution
    year_col = None

    if "published_year" in df.columns:
        year_col = "published_year"
    elif "release_year" in df.columns:
        year_col = "release_year"

    if year_col:
        year_counts = df[year_col].dropna().value_counts().sort_index()

        if not year_counts.empty:
            plt.figure(figsize=(10, 5))
            year_counts.plot(kind="line", marker="o")

            plt.title("News Records per Published Year")
            plt.xlabel("Published Year")
            plt.ylabel("Record Count")

            chart_path = output_path / "published_year_distribution.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()

            saved_charts.append(str(chart_path))
            logger.info("Saved published year chart: %s", chart_path)

            # Backward-compatible file name
            legacy_chart_path = output_path / "release_year_distribution.png"
            plt.figure(figsize=(10, 5))
            year_counts.plot(kind="line", marker="o")

            plt.title("Records per Year")
            plt.xlabel("Year")
            plt.ylabel("Record Count")

            plt.tight_layout()
            plt.savefig(legacy_chart_path, dpi=120, bbox_inches="tight")
            plt.close()

            saved_charts.append(str(legacy_chart_path))
            logger.info("Saved legacy release year chart: %s", legacy_chart_path)

    # Category distribution
    if "category" in df.columns:
        chart = _save_bar_chart(
            series=df["category"],
            title="Top News Categories",
            xlabel="Category",
            ylabel="Record Count",
            output_path=output_path / "category_distribution.png",
            top_n=10,
        )
        if chart:
            saved_charts.append(chart)

    # Backward-compatible genres chart using category/genres
    if "genres" in df.columns:
        genre_series = (
            df["genres"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )

        chart = _save_bar_chart(
            series=genre_series,
            title="Top Categories / Genre-like Labels",
            xlabel="Category",
            ylabel="Count",
            output_path=output_path / "genre_distribution.png",
            top_n=10,
        )
        if chart:
            saved_charts.append(chart)

    # Document type distribution
    if "document_type" in df.columns:
        chart = _save_bar_chart(
            series=df["document_type"],
            title="Document Type Distribution",
            xlabel="Document Type",
            ylabel="Record Count",
            output_path=output_path / "document_type_distribution.png",
            top_n=10,
        )
        if chart:
            saved_charts.append(chart)

    # Source distribution
    if "source_name" in df.columns:
        chart = _save_bar_chart(
            series=df["source_name"],
            title="Top News Sources",
            xlabel="Source",
            ylabel="Record Count",
            output_path=output_path / "source_distribution.png",
            top_n=10,
        )
        if chart:
            saved_charts.append(chart)

    logger.info("News EDA distribution charts complete: %d charts saved", len(saved_charts))

    return saved_charts