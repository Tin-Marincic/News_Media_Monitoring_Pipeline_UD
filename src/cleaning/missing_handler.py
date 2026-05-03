"""
src/cleaning/missing_handler.py

Missing-value handling for the News Media Monitoring Pipeline.

This module:
- reports missing values
- drops rows missing critical identifiers/titles
- fills missing text fields with placeholders
- replaces unrealistic zero values with NaN
- fills numeric missing values with median values
- drops columns with very high missingness
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a missing-value report for all columns with missing values.
    """
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    report = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values,
        "dtype": [str(dtype) for dtype in df.dtypes],
    })

    report = report[report["missing_count"] > 0].sort_values(
        "missing_pct",
        ascending=False,
    ).reset_index(drop=True)

    logger.info("Missing value report generated for %d columns", len(report))

    return report


def drop_rows_missing_critical_fields(
    df: pd.DataFrame,
    critical_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Drop rows that are missing critical fields.

    For news data, title is the main critical field. If record_id exists,
    we also require it because it is used for duplicate checks and validation.
    """
    df = df.copy()

    if critical_columns is None:
        critical_columns = [col for col in ["record_id", "title"] if col in df.columns]

    before = len(df)

    for col in critical_columns:
        if col not in df.columns:
            logger.warning("Critical column %s not found; skipping", col)
            continue

        df = df.dropna(subset=[col])

        if df[col].dtype == "object" or str(df[col].dtype) in ["string", "str"]:
            df = df[df[col].astype(str).str.strip() != ""]

    after = len(df)
    dropped = before - after

    logger.info(
        "drop_rows_missing_critical_fields: dropped %d rows using columns=%s",
        dropped,
        critical_columns,
    )

    return df.reset_index(drop=True)


def drop_rows_missing_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible helper matching the professor's function name.

    Drops rows where title is missing or empty.
    """
    if "title" not in df.columns:
        logger.warning("drop_rows_missing_title skipped because title column is missing")
        return df.copy().reset_index(drop=True)

    before = len(df)

    df = df.copy()
    df = df.dropna(subset=["title"])
    df = df[df["title"].astype(str).str.strip() != ""]

    after = len(df)
    dropped = before - after

    logger.info("drop_rows_missing_title: dropped %d rows", dropped)

    return df.reset_index(drop=True)


def fill_missing_text_fields(
    df: pd.DataFrame,
    text_columns: Optional[list] = None,
    placeholder: str = "No text available.",
) -> pd.DataFrame:
    """
    Fill missing text fields with a placeholder.

    For the news pipeline, useful fields include:
    title, description, content_text, overview, text, raw_text, processed_text.
    """
    df = df.copy()

    if text_columns is None:
        possible_cols = [
            "title",
            "description",
            "content_text",
            "overview",
            "text",
            "raw_text",
            "processed_text",
            "preview_text",
            "source_name",
            "category",
            "document_type",
            "language",
        ]
        text_columns = [col for col in possible_cols if col in df.columns]

    for col in text_columns:
        if col not in df.columns:
            continue

        missing_before = df[col].isna().sum()

        df[col] = df[col].fillna(placeholder)

        empty_mask = df[col].astype(str).str.strip() == ""
        empty_count = int(empty_mask.sum())

        df.loc[empty_mask, col] = placeholder

        logger.info(
            "fill_missing_text_fields: %s -> filled %d missing and %d empty values",
            col,
            missing_before,
            empty_count,
        )

    return df


def fill_missing_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible helper matching the professor's function name.

    In the news pipeline, overview is an alias for content_text.
    """
    df = df.copy()

    if "overview" in df.columns:
        before = df["overview"].isna().sum()
        df["overview"] = df["overview"].fillna("No overview available.")
        empty_mask = df["overview"].astype(str).str.strip() == ""
        df.loc[empty_mask, "overview"] = "No overview available."

        logger.info("fill_missing_overview: filled %d rows", before)

    elif "content_text" in df.columns:
        before = df["content_text"].isna().sum()
        df["content_text"] = df["content_text"].fillna("No content available.")
        empty_mask = df["content_text"].astype(str).str.strip() == ""
        df.loc[empty_mask, "content_text"] = "No content available."

        logger.info("fill_missing_overview/content_text: filled %d rows", before)

    else:
        logger.warning("fill_missing_overview skipped because no overview/content_text column exists")

    return df


def replace_zero_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replace zero values with NaN in selected numeric columns.

    In the news pipeline, zeros can mean missing or unavailable values
    for mentions, rating_score, sentiment_score, or content_length.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning("replace_zero_with_nan skipped missing column: %s", col)
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")

        zeros = int((numeric_col == 0).sum())

        df[col] = numeric_col.replace(0, np.nan)

        logger.info(
            "replace_zero_with_nan: %s -> replaced %d zeros",
            col,
            zeros,
        )

    return df


def fill_numeric_with_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Fill missing numeric values with the median of each column.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning("fill_numeric_with_median skipped missing column: %s", col)
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        filled = int(df[col].isna().sum())
        median_val = df[col].median()

        if pd.isna(median_val):
            logger.warning(
                "fill_numeric_with_median skipped %s because median is NaN",
                col,
            )
            continue

        df[col] = df[col].fillna(median_val)

        logger.info(
            "fill_numeric_with_median: %s -> filled %d with %.2f",
            col,
            filled,
            median_val,
        )

    return df


def drop_high_missingness_columns(
    df: pd.DataFrame,
    threshold: float = 0.60,
    protected_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Drop columns whose missing ratio is above threshold.

    Protected columns are never dropped even if they have high missingness.
    """
    df = df.copy()

    if protected_columns is None:
        protected_columns = [
            "record_id",
            "title",
            "content_text",
            "overview",
            "category",
            "document_type",
            "language",
            "rating_score",
            "published_date",
            "published_year",
            "source_name",
            "source_path",
            "url",
        ]

    missing_ratio = df.isna().mean()

    to_drop = [
        col for col, ratio in missing_ratio.items()
        if ratio > threshold and col not in protected_columns
    ]

    if to_drop:
        df = df.drop(columns=to_drop)

        logger.info(
            "drop_high_missingness_columns: dropped columns=%s using threshold=%.2f",
            to_drop,
            threshold,
        )
    else:
        logger.info(
            "drop_high_missingness_columns: no columns dropped using threshold=%.2f",
            threshold,
        )

    return df