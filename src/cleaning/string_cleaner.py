"""
src/cleaning/string_cleaner.py

String cleaning utilities for the News Media Monitoring Pipeline.

Based on the Lab 9 string-cleaning requirements:
- remove extra whitespace
- normalize language codes
- clean text/content columns
- extract year from date columns
- normalize category/genre-like labels
"""

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
RE_MULTI_SPACE = re.compile(r"\s+")
RE_YEAR_PARENS = re.compile(r"\(\d{4}\)")
RE_SPECIAL_CHARS = re.compile(r"[^\w\s\-\'\"\.,!?:/&]")
RE_HTML = re.compile(r"<[^>]+>")
RE_VALID_LANGUAGE = re.compile(r"^[a-z]{2}$")


def clean_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean title values:
    - fill missing titles
    - strip whitespace
    - collapse multiple spaces
    - remove year in parentheses, e.g. (2026)
    """
    df = df.copy()

    if "title" not in df.columns:
        logger.warning("clean_title skipped because title column is missing")
        return df

    before = df["title"].copy()

    df["title"] = (
        df["title"]
        .fillna("Untitled Record")
        .astype(str)
        .str.strip()
        .str.replace(RE_MULTI_SPACE, " ", regex=True)
        .str.replace(RE_YEAR_PARENS, "", regex=True)
        .str.strip()
    )

    empty_mask = df["title"].str.len() == 0
    df.loc[empty_mask, "title"] = "Untitled Record"

    changed = (before.fillna("").astype(str) != df["title"].astype(str)).sum()
    logger.info("clean_title: %d titles modified", changed)

    return df


def clean_language_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize language codes.

    For news data, the main column is language.
    For Lab 8 compatibility, original_language is also updated if present.
    """
    df = df.copy()

    language_col = None

    if "language" in df.columns:
        language_col = "language"
    elif "original_language" in df.columns:
        language_col = "original_language"

    if language_col is None:
        logger.warning("clean_language_code skipped because no language column exists")
        return df

    lang = (
        df[language_col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_values = ["", "none", "nan", "null", "n/a"]
    lang = lang.where(~lang.isin(invalid_values), "unknown")

    valid_mask = lang.str.fullmatch(RE_VALID_LANGUAGE.pattern)
    lang = lang.where(valid_mask | lang.eq("unknown"), "unknown")

    df[language_col] = lang

    if "language" in df.columns:
        df["language"] = lang

    if "original_language" in df.columns:
        df["original_language"] = lang

    logger.info("clean_language_code: normalized language codes")

    return df


def clean_overview_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the main article text field.

    In the news pipeline, content_text is the main text field.
    overview is kept as a Lab 8 compatibility alias.
    """
    df = df.copy()

    text_columns = [
        col for col in [
            "content_text",
            "overview",
            "description",
            "text",
            "processed_text",
            "raw_text",
            "preview_text",
        ]
        if col in df.columns
    ]

    if not text_columns:
        logger.warning("clean_overview_text skipped because no text columns exist")
        return df

    for col in text_columns:
        before = df[col].copy()

        df[col] = (
            df[col]
            .fillna("No content available.")
            .astype(str)
            .str.strip()
            .str.replace(RE_HTML, " ", regex=True)
            .str.replace(RE_MULTI_SPACE, " ", regex=True)
            .str.strip()
        )

        empty_mask = df[col].str.len() == 0
        df.loc[empty_mask, col] = "No content available."

        changed = (before.fillna("").astype(str) != df[col].astype(str)).sum()
        logger.info("clean_overview_text: %s cleaned, %d values modified", col, changed)

    if "content_text" in df.columns and "overview" in df.columns:
        df["overview"] = df["content_text"]

    return df


def extract_year_from_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract year from published_date/release_date.

    For news data:
    - preferred input: published_date
    - compatibility input: release_date
    - output: published_year and release_year
    """
    df = df.copy()

    date_col = None

    if "published_date" in df.columns:
        date_col = "published_date"
    elif "publishedAt" in df.columns:
        date_col = "publishedAt"
    elif "release_date" in df.columns:
        date_col = "release_date"

    if date_col is None:
        logger.warning("extract_year_from_release_date skipped because no date column exists")
        return df

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

    df["published_year"] = parsed_dates.dt.year.astype("Int64")
    df["release_year"] = df["published_year"]

    valid_years = df["published_year"].notna().sum()

    logger.info(
        "extract_year_from_release_date: extracted %d years from %s",
        valid_years,
        date_col,
    )

    return df


def clean_genre_string(df: pd.DataFrame, col: str = "genres") -> pd.DataFrame:
    """
    Normalize category/genre-like labels.

    In the news pipeline:
    - category and document_type are the meaningful labels
    - genres is a compatibility alias from Lab 8
    """
    df = df.copy()

    label_columns = []

    for candidate in [col, "category", "document_type", "source_name"]:
        if candidate in df.columns and candidate not in label_columns:
            label_columns.append(candidate)

    if not label_columns:
        logger.warning("clean_genre_string skipped because no label columns exist")
        return df

    for label_col in label_columns:
        df[label_col] = (
            df[label_col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(RE_MULTI_SPACE, " ", regex=True)
            .str.replace(r"\s*,\s*", ", ", regex=True)
        )

        df[label_col] = df[label_col].replace({
            "": "unknown",
            "nan": "unknown",
            "none": "unknown",
            "null": "unknown",
            "n/a": "unknown",
        })

        logger.info("clean_genre_string: %s normalized", label_col)

    if "category" in df.columns and "genres" in df.columns:
        df["genres"] = df["category"]

    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full string-cleaning workflow.
    """
    logger.info("Starting string cleaning workflow")

    df = df.copy()

    df = clean_title(df)
    df = clean_language_code(df)
    df = clean_overview_text(df)
    df = extract_year_from_release_date(df)
    df = clean_genre_string(df)

    logger.info("String cleaning workflow complete")

    return df