import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleaning.missing_handler import (
    report_missing,
    drop_rows_missing_title,
    fill_missing_overview,
    replace_zero_with_nan,
    fill_numeric_with_median,
    drop_high_missingness_columns,
)

from src.cleaning.string_cleaner import (
    clean_title,
    clean_language_code,
    clean_overview_text,
    extract_year_from_release_date,
    clean_genre_string,
)

from src.cleaning.deduplicator import (
    drop_exact_duplicates,
    drop_duplicate_ids,
    drop_duplicate_urls,
    drop_duplicate_titles,
    count_duplicates,
)

from src.cleaning.type_converter import (
    convert_dates,
    convert_numeric_columns,
    convert_category_columns,
)

from src.cleaning.validator import (
    validate_no_null_titles,
    validate_rating_score_range,
    validate_language_codes,
    run_all_validations,
)

from src.cleaning.clean_pipeline import run_cleaning_pipeline


@pytest.fixture
def sample_news():
    """
    Small news-media dataset with intentional data quality issues.
    """
    return pd.DataFrame({
        "record_id": [1, 2, 2, 3, 4],
        "title": [
            "  Election   Update (2026) ",
            "AI Market Growth",
            "AI Market Growth",
            "",
            "Sports Highlights",
        ],
        "content_text": [
            "  <p>Election news text</p>  ",
            None,
            None,
            "   ",
            "Sports story content",
        ],
        "overview": [
            "  <p>Election news text</p>  ",
            None,
            None,
            "   ",
            "Sports story content",
        ],
        "category": [
            " Politics ",
            "BUSINESS",
            "BUSINESS",
            "sports",
            "SPORTS",
        ],
        "document_type": [
            " news_api ",
            "NEWS_API",
            "NEWS_API",
            "excel",
            "excel",
        ],
        "language": [
            " EN ",
            "unknown",
            "unknown",
            "bad-language",
            None,
        ],
        "rating_score": [4.0, 0.0, 0.0, None, 6.0],
        "popularity": [10, 0, 0, 5, None],
        "vote_average": [4.0, 0.0, 0.0, None, 6.0],
        "vote_count": [1, 0, 0, 2, None],
        "published_date": [
            "2026-03-01",
            "2026-03-02",
            "2026-03-02",
            "invalid-date",
            "2026-03-04",
        ],
        "url": [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/b",
            None,
            "https://example.com/c",
        ],
        "content_length": [25, 0, 0, 1, 20],
        "title_length": [28, 16, 16, 0, 17],
    })


def test_report_missing_detects_missing_values(sample_news):
    result = report_missing(sample_news)

    assert not result.empty
    assert "content_text" in result["column"].values
    assert "rating_score" in result["column"].values


def test_drop_rows_missing_title_removes_empty_string(sample_news):
    result = drop_rows_missing_title(sample_news)

    assert result["title"].isna().sum() == 0
    assert (result["title"].str.strip() == "").sum() == 0


def test_fill_missing_overview_no_nulls_remain(sample_news):
    result = fill_missing_overview(sample_news)

    assert result["overview"].isna().sum() == 0


def test_fill_missing_overview_uses_placeholder(sample_news):
    result = fill_missing_overview(sample_news)

    filled = result.loc[sample_news["overview"].isna(), "overview"]
    assert filled.str.contains("available", case=False).all()


def test_replace_zero_with_nan_on_rating_score(sample_news):
    result = replace_zero_with_nan(sample_news, columns=["rating_score"])

    assert (result["rating_score"] == 0).sum() == 0
    assert result["rating_score"].isna().sum() >= 3


def test_fill_numeric_with_median(sample_news):
    result = replace_zero_with_nan(sample_news, columns=["rating_score"])
    result = fill_numeric_with_median(result, columns=["rating_score"])

    assert result["rating_score"].isna().sum() == 0


def test_drop_high_missingness_columns():
    df = pd.DataFrame({
        "title": ["A", "B", "C"],
        "content_text": ["x", "y", "z"],
        "mostly_missing": [None, None, "value"],
    })

    result = drop_high_missingness_columns(
        df,
        threshold=0.50,
        protected_columns=["title", "content_text"],
    )

    assert "mostly_missing" not in result.columns


def test_clean_title_strips_whitespace_and_removes_year(sample_news):
    result = clean_title(sample_news)

    assert result.loc[0, "title"] == "Election Update"
    assert not result["title"].str.startswith(" ").any()
    assert not result["title"].str.endswith(" ").any()
    assert result["title"].str.contains(r"\(\d{4}\)", regex=True).sum() == 0


def test_clean_language_code_normalizes_values(sample_news):
    result = clean_language_code(sample_news)

    assert result.loc[0, "language"] == "en"
    assert result.loc[3, "language"] == "unknown"
    assert result.loc[4, "language"] == "unknown"


def test_clean_overview_text_removes_html(sample_news):
    result = clean_overview_text(sample_news)

    assert "<p>" not in result.loc[0, "content_text"]
    assert result["content_text"].isna().sum() == 0


def test_extract_year_from_release_date(sample_news):
    result = extract_year_from_release_date(sample_news)

    assert "published_year" in result.columns
    assert "release_year" in result.columns
    assert result.loc[0, "published_year"] == 2026


def test_clean_genre_string_normalizes_labels(sample_news):
    result = clean_genre_string(sample_news)

    assert result.loc[0, "category"] == "politics"
    assert result.loc[1, "document_type"] == "news_api"


def test_drop_exact_duplicates_removes_copies():
    df = pd.DataFrame({
        "record_id": [1, 1, 2],
        "title": ["A", "A", "B"],
        "tags": [["x"], ["x"], ["y"]],
    })

    result = drop_exact_duplicates(df)

    assert len(result) == 2


def test_count_duplicates_returns_correct_number(sample_news):
    assert count_duplicates(sample_news, col="record_id") == 1


def test_drop_duplicate_ids_keeps_first(sample_news):
    result = drop_duplicate_ids(sample_news, id_col="record_id")

    assert count_duplicates(result, col="record_id") == 0
    assert len(result) == 4


def test_drop_duplicate_urls(sample_news):
    result = drop_duplicate_urls(sample_news)

    non_null_urls = result.dropna(subset=["url"])
    assert count_duplicates(non_null_urls, col="url") == 0


def test_drop_duplicate_titles_uses_title_and_date(sample_news):
    result = drop_duplicate_titles(sample_news)

    duplicate_count = result.duplicated(
        subset=["title", "published_date"]
    ).sum()

    assert duplicate_count == 0


def test_convert_dates_produces_datetime_type(sample_news):
    result = convert_dates(sample_news)

    assert pd.api.types.is_datetime64_any_dtype(result["published_date"])


def test_convert_dates_bad_values_become_nat(sample_news):
    result = convert_dates(sample_news)

    nat_count = result["published_date"].isna().sum()
    assert nat_count >= 1


def test_convert_numeric_columns(sample_news):
    result = convert_numeric_columns(sample_news)

    assert str(result["rating_score"].dtype) == "float32"


def test_convert_category_columns(sample_news):
    result = convert_category_columns(sample_news)

    assert str(result["language"].dtype) == "category"


def test_validate_no_null_titles_passes_on_clean_data():
    df = pd.DataFrame({
        "title": ["Election Update", "AI Market Growth"],
    })

    validate_no_null_titles(df)


def test_validate_no_null_titles_fails_on_empty_title():
    df = pd.DataFrame({
        "title": ["Election Update", ""],
    })

    with pytest.raises(AssertionError):
        validate_no_null_titles(df)


def test_validate_rating_score_fails_on_out_of_range():
    df = pd.DataFrame({
        "rating_score": [5.0, 15.0],
    })

    with pytest.raises(AssertionError):
        validate_rating_score_range(df)


def test_validate_language_codes_accepts_unknown():
    df = pd.DataFrame({
        "language": ["en", "unknown"],
    })

    validate_language_codes(df)


def test_full_cleaning_pipeline_passes_validations(sample_news):
    result = run_cleaning_pipeline(sample_news, save=False)

    validation_result = run_all_validations(result)

    assert validation_result["failed"] == 0
    assert validation_result["passed"] >= 5