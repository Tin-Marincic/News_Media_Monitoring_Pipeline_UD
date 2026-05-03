"""
src/analytics/regex_ops.py

Regular-expression operations for the News Media Monitoring Pipeline.

This module covers:
- extracting years and numbers from news/article titles
- filtering titles by prefix
- counting topic-related terms in article descriptions/content
- detecting unusually short descriptions/content
- parsing categories, document types, and genre-like labels
- returning most common categories
- validating news/document IDs and URLs using pre-compiled regex patterns

The function names keep compatibility with the original Lab 8 wording
(title, overview, genres), but the logic is adapted to news data.
"""

import collections
import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Pre-compiled regex patterns
# ------------------------------------------------------------

_YEAR_IN_TITLE = re.compile(r"\((\d{4})\)")
_VALID_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_IN_TITLE = re.compile(r"(\d+)")

_URL_PATTERN = re.compile(
    r"^https?://[^\s/$.?#].[^\s]*$",
    re.IGNORECASE,
)

_NUMERIC_ID = re.compile(r"^\d{1,12}$")

_CATEGORY_TEXT = re.compile(r"[A-Za-z][A-Za-z\s&\-]+")

# Topic terms useful for news-media monitoring.
_TOPIC_TERMS = re.compile(
    r"\b(?:election|politics|government|ai|technology|market|business|economy|"
    r"climate|sports|crime|police|war|health|education|inflation|banking|"
    r"cyber|security|startup|media|court|policy|finance)\b",
    re.IGNORECASE,
)

# Kept for backward compatibility with previous function names.
_CRIME_TERMS = re.compile(
    r"\b(?:murder|kill|killer|crime|criminal|detective|investigation|investigate|police|case)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------
# Title regex operations
# ------------------------------------------------------------

def extract_year_from_title(titles: pd.Series) -> pd.Series:
    """
    Extract a four-digit year from parentheses in news/article titles.

    Example:
    'Election Analysis (2026)' -> '2026'
    """
    result = titles.fillna("").astype(str).str.extract(
        _YEAR_IN_TITLE.pattern,
        expand=False,
    )

    logger.info("Titles with year in parentheses: %d", result.notna().sum())

    return result


def extract_any_year_from_title(titles: pd.Series) -> pd.Series:
    """
    Extract any valid 19xx or 20xx year from the title, even if it is not
    inside parentheses.

    Example:
    'GATE 2026 final answer key' -> '2026'
    """
    result = titles.fillna("").astype(str).str.extract(
        f"({_VALID_YEAR.pattern})",
        expand=False,
    )

    logger.info("Titles with any valid year: %d", result.notna().sum())

    return result


def filter_titles_starting_with(
    df: pd.DataFrame,
    prefix: str = "The",
) -> pd.DataFrame:
    """
    Filter news/article records whose title starts with the selected prefix.
    Uses regex and escapes the prefix safely.
    """
    if "title" not in df.columns:
        logger.warning("filter_titles_starting_with skipped because title column is missing")
        return pd.DataFrame()

    pattern = rf"^{re.escape(prefix)}\b"

    mask = df["title"].fillna("").astype(str).str.contains(
        pattern,
        case=False,
        na=False,
        regex=True,
    )

    result = df[mask]

    logger.info(
        'Titles starting with "%s": %d rows',
        prefix,
        len(result),
    )

    return result


def extract_number_from_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a title_number column with the first number found in the title.

    Example:
    'GATE 2026 final answer key' -> 2026
    """
    if "title" not in df.columns:
        logger.warning("extract_number_from_title skipped because title column is missing")
        return df.copy()

    df = df.copy()

    df["title_number"] = df["title"].fillna("").astype(str).str.extract(
        _NUMBER_IN_TITLE.pattern,
        expand=False,
    )

    logger.info(
        "Titles containing at least one number: %d",
        df["title_number"].notna().sum(),
    )

    return df


# ------------------------------------------------------------
# Content / overview regex operations
# ------------------------------------------------------------

def _get_content_column(df: pd.DataFrame) -> str | None:
    """
    Return the best available text column for regex analysis.
    """
    candidates = [
        "content_text",
        "overview",
        "description",
        "text",
        "processed_text",
        "raw_text",
        "preview_text",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def topic_overview_count(df: pd.DataFrame) -> int:
    """
    Count how many records contain news-topic terms in the main text field.
    """
    content_col = _get_content_column(df)

    if content_col is None:
        logger.warning("topic_overview_count skipped because no content column was found")
        return 0

    mask = df[content_col].fillna("").astype(str).str.contains(
        _TOPIC_TERMS.pattern,
        case=False,
        na=False,
        regex=True,
    )

    count = int(mask.sum())

    logger.info("Topic-related content rows: %d", count)

    return count


def topic_overview_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows where the main text field contains news-topic terms.
    """
    content_col = _get_content_column(df)

    if content_col is None:
        logger.warning("topic_overview_rows skipped because no content column was found")
        return pd.DataFrame()

    mask = df[content_col].fillna("").astype(str).str.contains(
        _TOPIC_TERMS.pattern,
        case=False,
        na=False,
        regex=True,
    )

    cols = [
        col for col in [
            "record_id",
            "title",
            content_col,
            "category",
            "document_type",
            "source_name",
        ]
        if col in df.columns
    ]

    result = df.loc[mask, cols]

    logger.info("Topic-related content rows returned: %d", len(result))

    return result


def crime_overview_count(df: pd.DataFrame) -> int:
    """
    Backward-compatible Lab 8 function name.

    Counts records containing crime/police/investigation-related terms
    in the main content field.
    """
    content_col = _get_content_column(df)

    if content_col is None:
        logger.warning("crime_overview_count skipped because no content column was found")
        return 0

    mask = df[content_col].fillna("").astype(str).str.contains(
        _CRIME_TERMS.pattern,
        case=False,
        na=False,
        regex=True,
    )

    count = int(mask.sum())

    logger.info("Crime-related news/content rows: %d", count)

    return count


def crime_overview_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible Lab 8 function name.

    Return rows where the main content field contains crime-related terms.
    """
    content_col = _get_content_column(df)

    if content_col is None:
        logger.warning("crime_overview_rows skipped because no content column was found")
        return pd.DataFrame()

    mask = df[content_col].fillna("").astype(str).str.contains(
        _CRIME_TERMS.pattern,
        case=False,
        na=False,
        regex=True,
    )

    cols = [
        col for col in [
            "record_id",
            "title",
            content_col,
            "category",
            "document_type",
            "source_name",
        ]
        if col in df.columns
    ]

    result = df.loc[mask, cols]

    logger.info("Crime-related content rows returned: %d", len(result))

    return result


def short_overviews(
    df: pd.DataFrame,
    max_chars: int = 40,
) -> pd.DataFrame:
    """
    Return rows where the main content/description field is unusually short.
    """
    content_col = _get_content_column(df)

    if content_col is None:
        logger.warning("short_overviews skipped because no content column was found")
        return pd.DataFrame()

    content_text = df[content_col].fillna("").astype(str)

    mask = content_text.str.len() < max_chars

    cols = [
        col for col in [
            "record_id",
            "title",
            content_col,
            "category",
            "document_type",
        ]
        if col in df.columns
    ]

    result = df.loc[mask, cols]

    logger.info(
        "Short content rows under %d characters: %d rows",
        max_chars,
        len(result),
    )

    return result


# ------------------------------------------------------------
# Category / genre-like parsing
# ------------------------------------------------------------

def _parse_category_value(value: Any) -> list:
    """
    Parse category-like labels from multiple possible formats:

    1. Comma-separated text:
       "Politics, Business"

    2. List-like text:
       "['Politics', 'Technology']"

    3. Single text value:
       "news_api"

    4. JSON-like text:
       {"name": "Technology"}
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # JSON-like "name": "Technology"
    json_names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)

    if json_names:
        return [item.strip() for item in json_names if item.strip()]

    # Comma-separated categories
    if "," in text:
        categories = [part.strip(" []'\"") for part in text.split(",")]
        return [category for category in categories if category]

    # General category text fallback
    matches = _CATEGORY_TEXT.findall(text)

    cleaned = [match.strip() for match in matches if match.strip()]

    return cleaned if cleaned else [text]


def extract_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible Lab 8 function name.

    For the news pipeline, this parses category/document_type/source labels
    into a genre_list/category_list column.

    Priority:
    1. category
    2. genres
    3. document_type
    4. source_name
    """
    df = df.copy()

    possible_columns = [
        "category",
        "genres",
        "document_type",
        "source_name",
        "source_path",
    ]

    category_col = None

    for col in possible_columns:
        if col in df.columns:
            category_col = col
            break

    if category_col is None:
        logger.warning("No category-like column found")
        df["genre_list"] = [[] for _ in range(len(df))]
        df["category_list"] = df["genre_list"]
        return df

    df["genre_list"] = df[category_col].apply(_parse_category_value)
    df["category_list"] = df["genre_list"]

    has_categories = df["genre_list"].apply(
        lambda value: isinstance(value, list) and len(value) > 0
    )

    logger.info(
        "Category labels extracted from column '%s': %d rows with labels",
        category_col,
        int(has_categories.sum()),
    )

    return df


def top_genres(
    df: pd.DataFrame,
    n: int = 15,
) -> list:
    """
    Backward-compatible Lab 8 function name.

    Return the most common category/genre-like labels as:
    [('news_api', 10), ('pdf', 5)]
    """
    if "genre_list" not in df.columns:
        logger.info("genre_list column not found; running extract_genres first")
        df = extract_genres(df)

    all_categories = []

    for genre_list in df["genre_list"].dropna():
        if isinstance(genre_list, list):
            all_categories.extend(genre_list)

    top = collections.Counter(all_categories).most_common(n)

    logger.info("Top category labels calculated: %d labels returned", len(top))

    return top


# ------------------------------------------------------------
# ID / URL validation
# ------------------------------------------------------------

def validate_news_id(id_value: Any) -> bool:
    """
    Validate local/news/document IDs.

    Valid formats:
    - positive numeric local ID
    - non-empty string IDs
    """
    if pd.isna(id_value):
        return False

    id_text = str(id_value).strip()

    if not id_text:
        return False

    if _NUMERIC_ID.match(id_text):
        return int(id_text) > 0

    return len(id_text) >= 3


def validate_url(url_value: Any) -> bool:
    """
    Validate URL format for news article records.
    """
    if pd.isna(url_value):
        return False

    url_text = str(url_value).strip()

    return bool(_URL_PATTERN.match(url_text))


def validate_movie_id(id_value: Any) -> bool:
    """
    Backward-compatible alias for old movie-focused function.
    In the news pipeline, this validates local/news/document IDs.
    """
    return validate_news_id(id_value)


def validate_movie_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible Lab 8 function name.

    Adds:
    - is_valid_news_id
    - is_valid_url, if URL column exists

    It checks the first available ID column from:
    record_id, id, article_id, document_id, local_id
    """
    df = df.copy()

    possible_id_columns = [
        "record_id",
        "id",
        "article_id",
        "document_id",
        "local_id",
    ]

    id_col = None

    for col in possible_id_columns:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        logger.warning("No news/document ID column found")
        df["is_valid_news_id"] = False
    else:
        df["is_valid_news_id"] = df[id_col].apply(validate_news_id)

        logger.info(
            "News/document ID validation completed using column '%s': %d valid IDs",
            id_col,
            int(df["is_valid_news_id"].sum()),
        )

    if "url" in df.columns:
        df["is_valid_url"] = df["url"].apply(validate_url)

        logger.info(
            "URL validation completed: %d valid URLs",
            int(df["is_valid_url"].sum()),
        )

    # Compatibility column name in case old notebook expects it
    df["is_valid_movie_id"] = df["is_valid_news_id"]

    return df

# -------------------------------------------------------------------
# Lab 9 regex-based cleaning and validation helpers
# -------------------------------------------------------------------

_DATE_YYYY_MM_DD = re.compile(r"^\d{4}-\d{2}-\d{2}")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$", re.IGNORECASE)
_NUMERIC_VALUE = re.compile(r"[-+]?\d*\.?\d+")


def detect_invalid_date_formats(
    df: pd.DataFrame,
    date_col: str = "published_date",
) -> pd.DataFrame:
    """
    Detect rows where a date column does not match a simple YYYY-MM-DD pattern.

    This is a regex-based pre-validation helper. It does not replace
    pd.to_datetime, but it helps identify suspicious original strings.
    """
    if date_col not in df.columns:
        logger.warning("detect_invalid_date_formats skipped because %s is missing", date_col)
        return pd.DataFrame()

    date_text = df[date_col].dropna().astype(str).str.strip()

    invalid_mask = ~date_text.str.match(_DATE_YYYY_MM_DD.pattern, na=False)

    invalid_dates = date_text[invalid_mask]

    result = pd.DataFrame({
        "row_index": invalid_dates.index,
        "column": date_col,
        "value": invalid_dates.values,
        "issue": "Invalid date format; expected YYYY-MM-DD",
    })

    logger.info(
        "detect_invalid_date_formats: %d invalid values found in %s",
        len(result),
        date_col,
    )

    return result


def detect_invalid_language_codes(
    df: pd.DataFrame,
    language_col: str = "language",
) -> pd.DataFrame:
    """
    Detect invalid language codes.

    Accepts two-letter language codes and 'unknown'.
    """
    if language_col not in df.columns:
        if "original_language" in df.columns:
            language_col = "original_language"
        else:
            logger.warning("detect_invalid_language_codes skipped because no language column exists")
            return pd.DataFrame()

    lang_text = df[language_col].dropna().astype(str).str.strip().str.lower()

    valid_mask = lang_text.str.match(_LANGUAGE_CODE.pattern, na=False) | lang_text.eq("unknown")

    invalid_values = lang_text[~valid_mask]

    result = pd.DataFrame({
        "row_index": invalid_values.index,
        "column": language_col,
        "value": invalid_values.values,
        "issue": "Invalid language code",
    })

    logger.info(
        "detect_invalid_language_codes: %d invalid values found in %s",
        len(result),
        language_col,
    )

    return result


def extract_numeric_values_from_text(
    text_series: pd.Series,
) -> pd.Series:
    """
    Extract the first numeric value from a text Series.

    Examples:
    'Article mentions: 25' -> 25
    'Score 4.5/10' -> 4.5
    """
    result = (
        text_series
        .fillna("")
        .astype(str)
        .str.extract(f"({_NUMERIC_VALUE.pattern})", expand=False)
    )

    logger.info(
        "extract_numeric_values_from_text: extracted %d numeric values",
        result.notna().sum(),
    )

    return result


def add_extracted_number_column(
    df: pd.DataFrame,
    source_col: str = "title",
    output_col: str = "extracted_number",
) -> pd.DataFrame:
    """
    Add a column containing the first numeric value extracted from text.
    """
    df = df.copy()

    if source_col not in df.columns:
        logger.warning("add_extracted_number_column skipped because %s is missing", source_col)
        df[output_col] = pd.NA
        return df

    df[output_col] = extract_numeric_values_from_text(df[source_col])

    logger.info(
        "add_extracted_number_column: created %s from %s",
        output_col,
        source_col,
    )

    return df


def flag_short_content(
    df: pd.DataFrame,
    text_col: str = "content_text",
    min_chars: int = 40,
) -> pd.DataFrame:
    """
    Add a boolean flag for suspiciously short article/document content.
    """
    df = df.copy()

    if text_col not in df.columns:
        if "overview" in df.columns:
            text_col = "overview"
        else:
            logger.warning("flag_short_content skipped because no content column exists")
            df["is_short_content"] = False
            return df

    df["is_short_content"] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.len()
        .lt(min_chars)
    )

    logger.info(
        "flag_short_content: %d rows flagged under %d characters",
        int(df["is_short_content"].sum()),
        min_chars,
    )

    return df