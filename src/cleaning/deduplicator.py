"""
src/cleaning/deduplicator.py

Deduplication helpers for the News Media Monitoring Pipeline.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully identical duplicate rows.

    Some integrated news pipeline columns can contain lists/dicts from MongoDB.
    pandas drop_duplicates cannot compare unhashable objects directly, so this
    function creates a temporary hashable/string version only for duplicate
    detection, then returns the original rows.
    """
    before = len(df)

    df = df.copy()

    comparable_df = df.copy()

    for col in comparable_df.columns:
        comparable_df[col] = comparable_df[col].apply(
            lambda value: str(value) if isinstance(value, (list, dict, set, tuple)) else value
        )

    duplicate_mask = comparable_df.duplicated(keep="first")

    df = df.loc[~duplicate_mask].copy()

    after = len(df)

    logger.info("drop_exact_duplicates: removed %d rows", before - after)

    return df.reset_index(drop=True)


def count_duplicates(df: pd.DataFrame, col: str = "record_id") -> int:
    """
    Count duplicate values in a selected column.
    """
    if col not in df.columns:
        logger.warning("count_duplicates: column %s not found", col)
        return 0

    return int(df.duplicated(subset=[col]).sum())


def drop_duplicate_ids(df: pd.DataFrame, id_col: str = "record_id") -> pd.DataFrame:
    """
    Remove duplicate records based on an ID column.

    For the news pipeline, preferred ID columns are:
    record_id, id, article_id, document_id, local_id.
    """
    df = df.copy()

    possible_id_cols = [id_col, "record_id", "id", "article_id", "document_id", "local_id"]

    actual_col = None

    for col in possible_id_cols:
        if col in df.columns:
            actual_col = col
            break

    if actual_col is None:
        logger.warning("drop_duplicate_ids: no ID column found, skipping")
        return df.reset_index(drop=True)

    before = len(df)

    df = df.drop_duplicates(subset=[actual_col], keep="first")

    after = len(df)

    logger.info(
        "drop_duplicate_ids (%s): removed %d rows",
        actual_col,
        before - after,
    )

    return df.reset_index(drop=True)


def drop_duplicate_urls(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """
    Remove duplicate article URLs while keeping rows with missing URLs.

    Missing URLs are not treated as duplicates because many non-API records
    such as PDF/OCR/scraped records may not have URLs.
    """
    df = df.copy()

    if url_col not in df.columns:
        logger.warning("drop_duplicate_urls: url column not found, skipping")
        return df.reset_index(drop=True)

    before = len(df)

    has_url = df[url_col].notna() & (df[url_col].astype(str).str.strip() != "")
    with_url = df[has_url].drop_duplicates(subset=[url_col], keep="first")
    without_url = df[~has_url]

    df = pd.concat([with_url, without_url], ignore_index=True)

    after = len(df)

    logger.info("drop_duplicate_urls: removed %d rows", before - after)

    return df.reset_index(drop=True)


def drop_duplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records using title plus date/category when available.

    This is useful for repeated news articles collected from API pagination.
    """
    df = df.copy()

    if "title" not in df.columns:
        logger.warning("drop_duplicate_titles: title column not found, skipping")
        return df.reset_index(drop=True)

    before = len(df)

    # Only deduplicate title + date where a real date exists.
    # Rows without dates are kept because PDFs, OCR records, scraped records,
    # and multimedia metadata may naturally share titles/names.
    date_col = None

    if "published_date" in df.columns:
        date_col = "published_date"
    elif "publishedAt" in df.columns:
        date_col = "publishedAt"
    elif "release_date" in df.columns:
        date_col = "release_date"

    if date_col is not None:
        has_date = df[date_col].notna()

        with_date = df[has_date].drop_duplicates(
            subset=["title", date_col],
            keep="first",
        )

        without_date = df[~has_date]

        df = pd.concat([with_date, without_date], ignore_index=True)

        logger.info(
            "drop_duplicate_titles: removed %d rows using title + %s only where date exists",
            before - len(df),
            date_col,
        )

    else:
        logger.info(
            "drop_duplicate_titles: no date column found, skipping title-based deduplication"
        )

    return df.reset_index(drop=True)