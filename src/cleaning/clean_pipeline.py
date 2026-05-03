"""
src/cleaning/clean_pipeline.py

Reusable cleaning pipeline for the News Media Monitoring Pipeline.
"""

import logging
from pathlib import Path

import pandas as pd

from src.cleaning.missing_handler import (
    report_missing,
    drop_rows_missing_critical_fields,
    fill_missing_text_fields,
    fill_missing_overview,
    replace_zero_with_nan,
    fill_numeric_with_median,
    drop_high_missingness_columns,
)
from src.cleaning.string_cleaner import (
    clean_string_columns,
)
from src.cleaning.deduplicator import (
    drop_exact_duplicates,
    drop_duplicate_ids,
    drop_duplicate_urls,
    drop_duplicate_titles,
)
from src.cleaning.type_converter import (
    convert_all_types,
    memory_report,
)
from src.cleaning.validator import run_all_validations

logger = logging.getLogger(__name__)

CLEANED_DIR = Path("data/processed/cleaned")
MISSING_REPORT_PATH = CLEANED_DIR / "missing_report.csv"
CLEANED_DATA_PATH = CLEANED_DIR / "cleaned_data.csv"
CLEAN_CSV_PATH = CLEANED_DIR / "clean.csv"


def run_cleaning_pipeline(
    df_raw: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the full Lab 9 cleaning workflow.

    Steps:
    1. Generate missing-value report
    2. Drop rows missing critical fields
    3. Clean strings
    4. Fill missing text fields
    5. Replace zero-as-missing values
    6. Fill numeric missing values with medians
    7. Remove duplicates
    8. Drop high-missingness columns
    9. Convert data types
    10. Validate cleaned data
    11. Save cleaned CSV files
    """
    logger.info("=== Starting Lab 9 cleaning pipeline: %d rows ===", len(df_raw))

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    df = df_raw.copy()

    # Step 1: missing report before cleaning
    logger.info("Step 1: generate missing-value report")
    missing_report = report_missing(df)

    if save:
        missing_report.to_csv(MISSING_REPORT_PATH, index=False)
        logger.info("Saved missing report to %s", MISSING_REPORT_PATH)

    # Step 2: critical fields
    logger.info("Step 2: drop rows missing critical fields")
    df = drop_rows_missing_critical_fields(df, critical_columns=["record_id", "title"])

    # Step 3: string cleaning before deduplication
    logger.info("Step 3: clean string columns")
    df = clean_string_columns(df)

    # Step 4: missing text
    logger.info("Step 4: fill missing text fields")
    df = fill_missing_text_fields(df)
    df = fill_missing_overview(df)

    # Step 5: zero-as-missing
    logger.info("Step 5: replace zero-as-missing numeric values")
    df = replace_zero_with_nan(
        df,
        columns=[
            "mentions",
            "sentiment_score",
            "rating_score",
            "popularity",
            "vote_average",
            "vote_count",
        ],
    )

    # Step 6: numeric imputation
    logger.info("Step 6: fill numeric missing values with medians")
    df = fill_numeric_with_median(
        df,
        columns=[
            "rating_score",
            "content_length",
            "title_length",
            "popularity",
            "vote_average",
            "vote_count",
        ],
    )

    # Step 7: deduplication
    logger.info("Step 7: remove duplicates")
    before_dedup = len(df)

    df = drop_exact_duplicates(df)
    df = drop_duplicate_urls(df, url_col="url")
    df = drop_duplicate_titles(df)
    df = drop_duplicate_ids(df, id_col="record_id")

    logger.info("Deduplication removed %d rows total", before_dedup - len(df))

    # Step 8: drop very sparse columns after required fields are protected
    logger.info("Step 8: drop high-missingness columns")
    df = drop_high_missingness_columns(df, threshold=0.60)

    # Step 9: type conversion
    logger.info("Step 9: convert data types")
    df_before_types = df.copy()
    df = convert_all_types(df)
    memory_report(df_before_types, df)

    # Step 10: validation
    logger.info("Step 10: run validation")
    run_all_validations(df)

    # Step 11: save
    if save:
        df.to_csv(CLEANED_DATA_PATH, index=False)
        df.to_csv(CLEAN_CSV_PATH, index=False)

        logger.info("Saved cleaned dataset to %s (%d rows)", CLEANED_DATA_PATH, len(df))
        logger.info("Saved cleaned dataset alias to %s (%d rows)", CLEAN_CSV_PATH, len(df))

    logger.info("=== Cleaning pipeline complete: %d rows remain ===", len(df))

    return df


def run_cleaning_pipeline_from_csv(
    input_path: str = "data/processed/analytics/raw_news_data.csv",
    save: bool = True,
) -> pd.DataFrame:
    """
    Load raw Lab 8 CSV and run the cleaning pipeline.
    """
    logger.info("Loading raw CSV for cleaning: %s", input_path)

    df_raw = pd.read_csv(input_path)

    logger.info("Loaded raw CSV for cleaning: shape=%s", df_raw.shape)

    return run_cleaning_pipeline(df_raw, save=save)