"""
src/cleaning/type_converter.py

Type conversion utilities for the News Media Monitoring Pipeline.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert available date columns to datetime.
    """
    df = df.copy()

    date_cols = [
        "published_date",
        "publishedAt",
        "release_date",
        "fetched_at",
        "extraction_timestamp",
    ]

    for col in date_cols:
        if col not in df.columns:
            continue

        df[col] = pd.to_datetime(df[col], errors="coerce")

        nat_count = int(df[col].isna().sum())

        logger.info("convert_dates: %s -> datetime, %d NaT values", col, nat_count)

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to appropriate pandas numeric types.
    """
    df = df.copy()

    float_cols = [
        "rating_score",
        "sentiment_score",
        "mentions",
        "popularity",
        "vote_average",
        "vote_count",
        "wins",
        "losses",
        "nominations",
        "awards",
        "Average Sentiment",
        "Average Mentions",
        "Highest Mentions",
    ]

    int_cols = [
        "record_id",
        "id",
        "page_number",
        "paragraph_number",
        "run_number",
        "published_year",
        "release_year",
        "year",
        "content_length",
        "title_length",
        "Total Articles",
        "Total Mentions",
        "Politics Articles",
        "Business Articles",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            logger.info("convert_numeric_columns: %s -> float32", col)

    for col in int_cols:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric_col.round().astype("Int64")
            logger.info("convert_numeric_columns: %s -> Int64", col)

    return df


def convert_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert low-cardinality text columns to category.
    """
    df = df.copy()

    cat_cols = [
        "language",
        "original_language",
        "category",
        "genres",
        "document_type",
        "source_name",
        "extraction_library",
        "file_name",
        "sheet_name",
        "best_picture",
        "bold",
        "italic",
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            logger.info("convert_category_columns: %s -> category", col)

    return df


def memory_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Print and return memory usage before and after conversion.
    """
    mb_before = df_before.memory_usage(deep=True).sum() / 1024**2
    mb_after = df_after.memory_usage(deep=True).sum() / 1024**2

    saved = mb_before - mb_after
    pct = (saved / mb_before * 100) if mb_before > 0 else 0

    print(f"Memory before: {mb_before:.2f} MB")
    print(f"Memory after:  {mb_after:.2f} MB")
    print(f"Saved:         {saved:.2f} MB  ({pct:.1f}%)")

    logger.info("memory_report: before=%.4f MB after=%.4f MB saved=%.4f MB pct=%.2f",
                mb_before, mb_after, saved, pct)

    return {
        "before_mb": mb_before,
        "after_mb": mb_after,
        "saved_mb": saved,
        "saved_pct": pct,
    }


def convert_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all type conversions.
    """
    logger.info("Starting type conversion workflow")

    df = df.copy()

    df = convert_dates(df)
    df = convert_numeric_columns(df)
    df = convert_category_columns(df)

    logger.info("Type conversion workflow complete")

    return df