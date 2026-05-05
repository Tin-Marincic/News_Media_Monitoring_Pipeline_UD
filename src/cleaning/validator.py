"""
src/cleaning/validator.py

Validation checks for the cleaned News Media Monitoring Pipeline dataset.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MIN_NEWS_YEAR = 1900
MAX_NEWS_YEAR = 2100
MIN_SCORE = 0.0
MAX_SCORE = 10.0


def validate_no_null_titles(df: pd.DataFrame) -> None:
    """
    Validate that title exists and is not empty.
    """
    assert "title" in df.columns, "Missing title column"

    assert df["title"].notna().all(), \
        f'Found {df["title"].isna().sum()} null titles'

    assert (df["title"].astype(str).str.strip() != "").all(), \
        "Found rows with empty string titles"

    logger.info("validate_no_null_titles: PASSED")


def validate_rating_score_range(df: pd.DataFrame) -> None:
    """
    Validate rating_score or vote_average is within 0-10.
    """
    score_col = None

    if "rating_score" in df.columns:
        score_col = "rating_score"
    elif "vote_average" in df.columns:
        score_col = "vote_average"

    if score_col is None:
        logger.warning("validate_rating_score_range skipped because no score column found")
        return

    non_null = pd.to_numeric(df[score_col], errors="coerce").dropna()

    assert non_null.between(MIN_SCORE, MAX_SCORE).all(), \
        f"{score_col} out of range [{MIN_SCORE}, {MAX_SCORE}]"

    logger.info("validate_rating_score_range (%s): PASSED", score_col)


def validate_year_range(df: pd.DataFrame) -> None:
    """
    Validate published_year/release_year/year range.
    """
    year_col = None

    for col in ["published_year", "release_year", "year"]:
        if col in df.columns:
            year_col = col
            break

    if year_col is None:
        logger.warning("validate_year_range skipped because no year column found")
        return

    non_null = pd.to_numeric(df[year_col], errors="coerce").dropna()

    if non_null.empty:
        logger.warning("validate_year_range skipped because all year values are missing")
        return

    assert non_null.between(MIN_NEWS_YEAR, MAX_NEWS_YEAR).all(), \
        f"{year_col} out of range [{MIN_NEWS_YEAR}, {MAX_NEWS_YEAR}]"

    logger.info("validate_year_range (%s): PASSED", year_col)


def validate_no_duplicate_ids(df: pd.DataFrame, id_col: str = "record_id") -> None:
    """
    Validate that ID column has no duplicates.
    """
    if id_col not in df.columns:
        logger.warning("validate_no_duplicate_ids skipped because %s is missing", id_col)
        return

    dup_count = int(df.duplicated(subset=[id_col]).sum())

    assert dup_count == 0, f"Found {dup_count} duplicate values in column {id_col}"

    logger.info("validate_no_duplicate_ids (%s): PASSED", id_col)


def validate_language_codes(df: pd.DataFrame) -> None:
    """
    Validate language codes.

    Accepts:
    - valid two-letter language codes like en, de, fr
    - unknown, because many integrated records do not store language.
    """
    language_col = None

    if "language" in df.columns:
        language_col = "language"
    elif "original_language" in df.columns:
        language_col = "original_language"

    if language_col is None:
        logger.warning("validate_language_codes skipped because no language column found")
        return

    non_null = df[language_col].dropna().astype(str).str.lower()

    valid = non_null.str.match(r"^[a-z]{2}$") | non_null.eq("unknown")

    assert valid.all(), f"Found {(~valid).sum()} invalid language codes"

    logger.info("validate_language_codes (%s): PASSED", language_col)


def validate_content_length(df: pd.DataFrame) -> None:
    """
    Validate content_length and title_length are non-negative.
    """
    for col in ["content_length", "title_length"]:
        if col not in df.columns:
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce").dropna()

        assert (numeric_col >= 0).all(), f"{col} contains negative values"

        logger.info("validate_content_length (%s): PASSED", col)


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate important cleaned dataset columns exist.
    """
    required = [
        "title",
        "content_text",
        "category",
        "document_type",
        "rating_score",
    ]

    missing = [col for col in required if col not in df.columns]

    assert not missing, f"Missing required cleaned columns: {missing}"

    logger.info("validate_required_columns: PASSED")


def run_all_validations(df: pd.DataFrame) -> dict:
    """
    Run all validations and return a result summary.
    """
    checks = [
        validate_required_columns,
        validate_no_null_titles,
        validate_rating_score_range,
        validate_year_range,
        validate_no_duplicate_ids,
        validate_language_codes,
        validate_content_length,
    ]

    passed = 0
    failed = 0
    failures = []

    for check in checks:
        try:
            check(df)
            print(f"  PASSED: {check.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {check.__name__} -> {e}")
            logger.error("Validation failed: %s -> %s", check.__name__, e)
            failed += 1
            failures.append({
                "check": check.__name__,
                "error": str(e),
            })

    print(f"\nValidation complete: {passed} passed, {failed} failed")

    logger.info("Validation complete: %d passed, %d failed", passed, failed)

    if failed > 0:
        raise AssertionError(f"{failed} validation checks failed: {failures}")

    return {
        "passed": passed,
        "failed": failed,
        "failures": failures,
    }