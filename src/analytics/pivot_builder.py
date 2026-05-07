"""
src/analytics/pivot_builder.py

Pivot and reshaping utilities for Lab 10 - News Media Monitoring Pipeline.

This module keeps the professor-style function names:
- wide_to_long
- long_to_wide
- build_pivot_table
- build_crosstab

It also adds project-specific helpers:
- add_primary_category
- add_analysis_year
- wide_to_long_metrics
- build_category_year_pivot
- build_language_decade_crosstab
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Professor-style generic functions
# -------------------------------------------------------------------

def wide_to_long(
    df: pd.DataFrame,
    id_vars: list,
    value_vars: list,
    var_name: str = "metric",
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Convert a wide DataFrame into long format using pd.melt().
    """
    long_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )

    logger.info(
        "Wide->Long: %d rows x %d cols -> %d rows x %d cols",
        len(df),
        len(df.columns),
        len(long_df),
        len(long_df.columns),
    )

    return long_df


def long_to_wide(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
) -> pd.DataFrame:
    """
    Convert a long DataFrame back into wide format using pivot().
    """
    wide_df = df.pivot(
        index=index_col,
        columns=columns_col,
        values=values_col,
    )

    wide_df.columns.name = None
    wide_df = wide_df.reset_index()

    logger.info(
        "Long->Wide: %d rows x %d cols",
        len(wide_df),
        len(wide_df.columns),
    )

    return wide_df


def build_pivot_table(
    df: pd.DataFrame,
    values,
    index,
    columns,
    aggfunc: str = "mean",
    fill_value=0,
    margins: bool = False,
) -> pd.DataFrame:
    """
    Build a generic pivot table.
    """
    pt = pd.pivot_table(
        df,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
    )

    logger.info("Pivot table shape: %s", pt.shape)

    return pt


def build_crosstab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    normalize=False,
) -> pd.DataFrame:
    """
    Build a generic crosstab table.
    """
    ct = pd.crosstab(
        df[row_col],
        df[col_col],
        margins=True,
        normalize=normalize,
    )

    logger.info("Crosstab shape: %s", ct.shape)

    return ct


# -------------------------------------------------------------------
# News Media Monitoring specific helpers
# -------------------------------------------------------------------

def add_primary_category(
    df: pd.DataFrame,
    output_col: str = "primary_category",
) -> pd.DataFrame:
    """
    Add or standardize a primary_category column.

    For the News Media Monitoring Pipeline, primary_category comes from:
    category -> genres -> document_type -> unknown
    """
    df = df.copy()

    if output_col in df.columns:
        df[output_col] = df[output_col].fillna("unknown").astype(str)

    elif "category" in df.columns:
        df[output_col] = df["category"].fillna("unknown").astype(str)

    elif "genres" in df.columns:
        df[output_col] = df["genres"].fillna("unknown").astype(str)

    elif "document_type" in df.columns:
        df[output_col] = df["document_type"].fillna("unknown").astype(str)

    elif "document_type_mysql" in df.columns:
        df[output_col] = df["document_type_mysql"].fillna("unknown").astype(str)

    else:
        df[output_col] = "unknown"

    df[output_col] = (
        df[output_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "": "unknown",
            "nan": "unknown",
            "none": "unknown",
            "null": "unknown",
        })
    )

    logger.info("add_primary_category: created/standardized %s", output_col)

    return df


def add_analysis_year(
    df: pd.DataFrame,
    output_col: str = "analysis_year",
) -> pd.DataFrame:
    """
    Add one clean year column for Lab 10 analysis.

    It checks:
    published_year, published_year_mysql, published_year_mongo,
    release_year, year, then date columns.
    """
    df = df.copy()

    candidate_year_cols = [
        "published_year",
        "published_year_mysql",
        "published_year_mongo",
        "release_year",
        "year",
    ]

    year_series = None

    for col in candidate_year_cols:
        if col in df.columns:
            year_series = pd.to_numeric(df[col], errors="coerce")
            break

    if year_series is None:
        candidate_date_cols = [
            "published_date",
            "published_date_mysql",
            "published_date_mongo",
            "release_date",
        ]

        for col in candidate_date_cols:
            if col in df.columns:
                year_series = pd.to_datetime(df[col], errors="coerce").dt.year
                break

    if year_series is None:
        df[output_col] = pd.NA
    else:
        df[output_col] = year_series.astype("Int64")

    logger.info(
        "add_analysis_year: created %s with %d non-null values",
        output_col,
        df[output_col].notna().sum(),
    )

    return df


def wide_to_long_metrics(
    df: pd.DataFrame,
    id_vars: list = None,
    value_vars: list = None,
    var_name: str = "metric",
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Convert at least three numeric Lab 10 metrics from wide to long format.

    News equivalent of movie metrics:
    - popularity
    - engagement_score
    - estimated_value
    - rating_score
    - content_length
    """
    df = df.copy()

    if id_vars is None:
        possible_id_vars = [
            "record_id",
            "title",
            "primary_category",
            "analysis_year",
            "document_type",
            "document_type_mysql",
        ]
        id_vars = [col for col in possible_id_vars if col in df.columns]

    if value_vars is None:
        possible_values = [
            "popularity",
            "engagement_score",
            "estimated_value",
            "rating_score",
            "content_length",
        ]
        value_vars = [col for col in possible_values if col in df.columns]

    if len(value_vars) < 3:
        raise ValueError(
            f"Need at least 3 numeric value columns for Lab 10 melt(). Found: {value_vars}"
        )

    long_df = wide_to_long(
        df=df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )

    logger.info(
        "wide_to_long_metrics: used id_vars=%s value_vars=%s",
        id_vars,
        value_vars,
    )

    return long_df


def long_to_wide_metrics(
    long_df: pd.DataFrame,
    index_cols: list = None,
    metric_col: str = "metric",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Convert the long metrics table back to a wide format.
    Uses pivot_table instead of pivot because record/title combinations may repeat.
    """
    if index_cols is None:
        possible_index_cols = [
            "record_id",
            "title",
            "primary_category",
            "analysis_year",
        ]
        index_cols = [col for col in possible_index_cols if col in long_df.columns]

    wide_df = long_df.pivot_table(
        index=index_cols,
        columns=metric_col,
        values=value_col,
        aggfunc="mean",
    ).reset_index()

    wide_df.columns.name = None

    logger.info(
        "long_to_wide_metrics: converted long shape %s to wide shape %s",
        long_df.shape,
        wide_df.shape,
    )

    return wide_df


def build_category_year_pivot(
    df: pd.DataFrame,
    value_col: str = "estimated_value",
    index_col: str = "analysis_year",
    columns_col: str = "primary_category",
    aggfunc: str = "mean",
    margins: bool = True,
) -> pd.DataFrame:
    """
    Build a pivot table showing estimated_value by year and category.

    This is the news-project equivalent of:
    revenue_usd by release_year and primary_genre.
    """
    df = df.copy()

    if index_col not in df.columns:
        df = add_analysis_year(df, output_col=index_col)

    if columns_col not in df.columns:
        df = add_primary_category(df, output_col=columns_col)

    if value_col not in df.columns:
        raise KeyError(f"Missing pivot value column: {value_col}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    pivot = build_pivot_table(
        df=df,
        values=value_col,
        index=index_col,
        columns=columns_col,
        aggfunc=aggfunc,
        fill_value=0,
        margins=margins,
    )

    logger.info(
        "build_category_year_pivot: created pivot with shape=%s",
        pivot.shape,
    )

    return pivot


def build_document_category_pivot(
    df: pd.DataFrame,
    value_col: str = "rating_score",
) -> pd.DataFrame:
    """
    Build a pivot table by document type and primary category.
    """
    df = df.copy()
    df = add_primary_category(df)

    document_col = "document_type"

    if document_col not in df.columns and "document_type_mysql" in df.columns:
        document_col = "document_type_mysql"

    if document_col not in df.columns:
        raise KeyError("No document_type column found")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    pivot = build_pivot_table(
        df=df,
        values=value_col,
        index=document_col,
        columns="primary_category",
        aggfunc="mean",
        fill_value=0,
        margins=True,
    )

    logger.info(
        "build_document_category_pivot: created pivot with shape=%s",
        pivot.shape,
    )

    return pivot


def build_language_decade_crosstab(
    df: pd.DataFrame,
    language_col: str = "language",
    year_col: str = "analysis_year",
) -> pd.DataFrame:
    """
    Create a language-versus-decade cross-tabulation.
    """
    df = df.copy()

    if year_col not in df.columns:
        df = add_analysis_year(df, output_col=year_col)

    if language_col not in df.columns:
        if "original_language" in df.columns:
            language_col = "original_language"
        else:
            df[language_col] = "unknown"

    years = pd.to_numeric(df[year_col], errors="coerce")
    df["decade"] = (years // 10 * 10).astype("Int64").astype(str) + "s"
    df.loc[years.isna(), "decade"] = "unknown"

    crosstab = build_crosstab(
        df=df,
        row_col=language_col,
        col_col="decade",
        normalize=False,
    )

    logger.info(
        "build_language_decade_crosstab: created crosstab with shape=%s",
        crosstab.shape,
    )

    return crosstab


def save_pivot_outputs(
    long_df: pd.DataFrame,
    category_year_pivot: pd.DataFrame,
    language_decade_crosstab: pd.DataFrame,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Save Lab 10 reshaping and pivot outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "news_metrics_long.csv"
    pivot_path = output_dir / "pivot_category_year.csv"
    crosstab_path = output_dir / "language_decade_crosstab.csv"

    long_df.to_csv(long_path, index=False)
    category_year_pivot.to_csv(pivot_path)
    language_decade_crosstab.to_csv(crosstab_path)

    logger.info("Saved long metrics to %s", long_path)
    logger.info("Saved category/year pivot to %s", pivot_path)
    logger.info("Saved language/decade crosstab to %s", crosstab_path)

    return {
        "long_path": str(long_path),
        "pivot_path": str(pivot_path),
        "crosstab_path": str(crosstab_path),
    }