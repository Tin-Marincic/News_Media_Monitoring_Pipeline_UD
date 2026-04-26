"""
src/analytics/selector.py

Selection and filtering helpers for the News Media Monitoring Pipeline.

Covers:
- column selection
- loc
- iloc
- boolean filtering
- isin filtering, including exclusion
- between filtering

The function names remain general enough for Lab 8, but the default
columns are adapted to news analytics fields such as rating_score,
mentions, popularity, category, document_type, and language.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def select_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Return a DataFrame with only the requested columns that actually exist.
    """
    existing = [col for col in cols if col in df.columns]

    logger.info(
        "Selecting columns: %d/%d requested columns found",
        len(existing),
        len(cols),
    )

    return df[existing]


def loc_filter(
    df: pd.DataFrame,
    min_rating_score: float = 7.0,
    result_cols: list = None,
    rating_col: str = "rating_score",
) -> pd.DataFrame:
    """
    Demonstrate label-based filtering with df.loc.

    Select news/document records with rating_score greater than or equal
    to min_rating_score.
    """
    if rating_col not in df.columns:
        # Backward-compatible fallback
        if "vote_average" in df.columns:
            rating_col = "vote_average"
        else:
            logger.warning("loc_filter skipped because no rating column was found")
            return pd.DataFrame()

    if result_cols is None:
        result_cols = [
            "record_id",
            "title",
            "document_type",
            "category",
            "rating_score",
            "mentions",
            "popularity",
            "language",
        ]

    result_cols = [col for col in result_cols if col in df.columns]

    df = df.copy()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    mask = df[rating_col] >= min_rating_score

    result = df.loc[mask, result_cols]

    logger.info(
        "loc_filter: %s >= %.2f returned %d rows",
        rating_col,
        min_rating_score,
        len(result),
    )

    return result


def iloc_sample(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """
    Demonstrate positional filtering with df.iloc.

    Returns every N-th row from the DataFrame.
    """
    if step <= 0:
        raise ValueError("step must be greater than 0")

    result = df.iloc[::step]

    logger.info(
        "iloc_sample: every %d-th row returned %d rows",
        step,
        len(result),
    )

    return result


def boolean_filter(
    df: pd.DataFrame,
    min_rating_score: float = 5.0,
    min_mentions: int = 1,
    min_popularity: float = 10.0,
) -> pd.DataFrame:
    """
    Demonstrate combined boolean filtering.

    Select news/document records that are:
    - high enough in rating_score
    - have enough mentions if available
    - have enough popularity/content length
    """
    df = df.copy()

    rating_col = "rating_score" if "rating_score" in df.columns else "vote_average"
    mentions_col = "mentions" if "mentions" in df.columns else "vote_count"
    popularity_col = "popularity" if "popularity" in df.columns else "content_length"

    required_cols = [rating_col, mentions_col, popularity_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(
            "boolean_filter skipped because these columns are missing: %s",
            missing_cols,
        )
        return pd.DataFrame()

    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df[mentions_col] = pd.to_numeric(df[mentions_col], errors="coerce")
    df[popularity_col] = pd.to_numeric(df[popularity_col], errors="coerce")

    mask = (
        (df[rating_col] > min_rating_score)
        & (df[mentions_col].fillna(0) >= min_mentions)
        & (df[popularity_col].fillna(0) > min_popularity)
    )

    result = df[mask]

    logger.info(
        "boolean_filter: %d rows matched %s > %.2f, %s >= %d, %s > %.2f",
        len(result),
        rating_col,
        min_rating_score,
        mentions_col,
        min_mentions,
        popularity_col,
        min_popularity,
    )

    return result


def isin_filter(
    df: pd.DataFrame,
    values: list = None,
    column: str = "language",
    exclude: bool = False,
) -> pd.DataFrame:
    """
    Demonstrate filtering with isin.

    By default, filters the language column.

    If exclude=False:
        return rows where selected column is in the given list.

    If exclude=True:
        return rows where selected column is NOT in the given list.
    """
    if column not in df.columns:
        # Backward-compatible fallback
        if column == "language" and "original_language" in df.columns:
            column = "original_language"
        elif "category" in df.columns:
            column = "category"
        elif "document_type" in df.columns:
            column = "document_type"
        else:
            logger.warning("isin_filter skipped because no suitable column was found")
            return pd.DataFrame()

    if values is None:
        if column == "language":
            values = ["en", "unknown"]
        elif column == "category":
            values = ["news_api", "Politics", "Business", "Technology", "Sports"]
        elif column == "document_type":
            values = ["news_api", "json", "pdf", "word", "excel"]
        else:
            values = df[column].dropna().astype(str).unique().tolist()[:5]

    mask = df[column].astype(str).isin([str(value) for value in values])

    if exclude:
        mask = ~mask

    result = df[mask]

    logger.info(
        "isin_filter: column=%s exclude=%s values=%s returned %d rows",
        column,
        exclude,
        values,
        len(result),
    )

    return result


def between_filter(
    df: pd.DataFrame,
    col: str = "rating_score",
    low: float = 2.0,
    high: float = 8.0,
) -> pd.DataFrame:
    """
    Demonstrate numeric range filtering with between.

    Select rows where selected column is between low and high.
    """
    if col not in df.columns:
        # Backward-compatible fallback
        if col == "rating_score" and "vote_average" in df.columns:
            col = "vote_average"
        elif "content_length" in df.columns:
            col = "content_length"
        else:
            logger.warning("between_filter skipped because column '%s' is missing", col)
            return pd.DataFrame()

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df[df[col].between(low, high, inclusive="both")]

    logger.info(
        "between_filter: %s between %.2f and %.2f returned %d rows",
        col,
        low,
        high,
        len(result),
    )

    return result