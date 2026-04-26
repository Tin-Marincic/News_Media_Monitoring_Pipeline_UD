"""
src/analytics/quality_report.py

Systematic data quality assessment for the News Media Monitoring Pipeline.

Covers:
- completeness
- validity
- consistency
- uniqueness
- missing value analysis
- zero-as-missing detection
- IQR outlier detection
- rating_score validation
- duplicate ID / URL detection
- missing/invalid title detection
- inconsistent date/year format detection
- missing-value heatmap
- full quality report CSV export
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _severity_from_pct(pct: float) -> str:
    """
    Convert a percentage into a simple severity label.
    """
    if pct > 30:
        return "HIGH"
    if pct > 10:
        return "MEDIUM"
    return "LOW"


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame listing every column with at least one missing value,
    together with count, percentage, and severity.
    """
    logger.info("Generating missing value report")

    if df.empty:
        logger.warning("Missing value report requested for empty DataFrame")
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct", "severity"])

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    report = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    })

    report = report[report["missing_count"] > 0].copy()
    report["severity"] = report["missing_pct"].apply(_severity_from_pct)

    report = report.sort_values(
        "missing_count",
        ascending=False,
    ).reset_index(drop=True)

    logger.info("Columns with missing values: %d", len(report))

    return report


def zero_as_missing(
    df: pd.DataFrame,
    cols: list = None,
) -> pd.DataFrame:
    """
    Detect columns where zero values may indicate missing or incomplete
    analytics data in the news pipeline.

    In this project, zero can be suspicious in fields such as:
    - mentions
    - sentiment_score
    - rating_score
    - content_length
    - title_length
    """
    logger.info("Running zero-as-missing check")

    if df.empty:
        logger.warning("Zero-as-missing check requested for empty DataFrame")
        return pd.DataFrame(columns=["column", "issue", "count", "pct", "severity"])

    if cols is None:
        possible_cols = [
            "mentions",
            "sentiment_score",
            "rating_score",
            "content_length",
            "title_length",
            "popularity",
            "vote_average",
            "vote_count",
        ]
        cols = [col for col in possible_cols if col in df.columns]

    rows = []

    for col in cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")

        n_zero = int((numeric_col.fillna(0) == 0).sum())
        pct = round(n_zero / len(df) * 100, 2)

        if n_zero > 0:
            rows.append({
                "column": col,
                "issue": "Zero values may represent missing or incomplete analytics data",
                "count": n_zero,
                "pct": pct,
                "severity": "HIGH" if pct > 50 else "MEDIUM",
            })

    logger.info("Zero-as-missing check complete for columns: %s", cols)

    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection for all numeric columns.

    Returns:
    column, q1, q3, iqr, lower_bound, upper_bound, outliers, outlier_pct
    """
    logger.info("Running IQR-based outlier detection")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    rows = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)

        n_outliers = int(outlier_mask.sum())
        outlier_pct = round(n_outliers / len(series) * 100, 2)

        if n_outliers > 0:
            rows.append({
                "column": col,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outliers": n_outliers,
                "outlier_pct": outlier_pct,
            })

    report = pd.DataFrame(rows)

    logger.info("Outlier detection complete: %d numeric columns with outliers", len(report))

    return report


def rating_validity_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate available numeric rating/score fields.

    Expected ranges:
    - rating_score: 0 to 10
    - vote_average: 0 to 10
    - rating: 0 to 10
    - sentiment_score: 0 to 1
    """
    logger.info("Running news rating/score validity checks")

    rows = []

    rating_0_to_10_cols = [
        col for col in [
            "rating_score",
            "vote_average",
            "rating",
            "relevance_score",
            "quality_score",
        ]
        if col in df.columns
    ]

    for col in rating_0_to_10_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")

        invalid_mask = numeric_col.notna() & (
            (numeric_col < 0) | (numeric_col > 10)
        )

        invalid_count = int(invalid_mask.sum())
        pct = round(invalid_count / len(df) * 100, 2) if len(df) else 0

        if invalid_count > 0:
            rows.append({
                "column": col,
                "issue": "Score outside valid 0-10 range",
                "count": invalid_count,
                "pct": pct,
                "severity": "HIGH",
            })

    if "sentiment_score" in df.columns:
        sentiment = pd.to_numeric(df["sentiment_score"], errors="coerce")

        invalid_mask = sentiment.notna() & (
            (sentiment < 0) | (sentiment > 1)
        )

        invalid_count = int(invalid_mask.sum())
        pct = round(invalid_count / len(df) * 100, 2) if len(df) else 0

        if invalid_count > 0:
            rows.append({
                "column": "sentiment_score",
                "issue": "Sentiment score outside valid 0-1 range",
                "count": invalid_count,
                "pct": pct,
                "severity": "HIGH",
            })

    logger.info("Rating/score validity check complete: %d issues", len(rows))

    return pd.DataFrame(rows)


def duplicate_id_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect duplicate identifiers from available news/document ID columns.

    Duplicate URLs are especially important in news pipelines because the same
    article may be ingested multiple times from the API.
    """
    logger.info("Running duplicate ID / URL checks")

    possible_id_cols = [
        "record_id",
        "id",
        "article_id",
        "document_id",
        "local_id",
        "url",
        "source_path",
    ]

    rows = []

    for col in possible_id_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if series.empty:
            continue

        duplicate_count = int(series.duplicated().sum())
        pct = round(duplicate_count / len(df) * 100, 2) if len(df) else 0

        if duplicate_count > 0:
            rows.append({
                "column": col,
                "issue": "Duplicate identifiers / repeated records",
                "count": duplicate_count,
                "pct": pct,
                "severity": "HIGH" if col in ["record_id", "url"] else "MEDIUM",
            })

    logger.info("Duplicate ID / URL check complete: %d issues", len(rows))

    return pd.DataFrame(rows)


def title_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing, empty, or invalid news/article/document titles.
    """
    logger.info("Running title quality checks")

    rows = []

    if "title" not in df.columns:
        rows.append({
            "column": "title",
            "issue": "Missing title column",
            "count": len(df),
            "pct": 100.0,
            "severity": "HIGH",
        })
        return pd.DataFrame(rows)

    title_series = df["title"]

    missing_count = int(title_series.isna().sum())
    missing_pct = round(missing_count / len(df) * 100, 2) if len(df) else 0

    if missing_count > 0:
        rows.append({
            "column": "title",
            "issue": "Missing titles",
            "count": missing_count,
            "pct": missing_pct,
            "severity": _severity_from_pct(missing_pct),
        })

    title_text = title_series.fillna("").astype(str).str.strip()

    empty_count = int((title_text.str.len() == 0).sum())
    empty_pct = round(empty_count / len(df) * 100, 2) if len(df) else 0

    if empty_count > 0:
        rows.append({
            "column": "title",
            "issue": "Empty titles",
            "count": empty_count,
            "pct": empty_pct,
            "severity": "MEDIUM",
        })

    invalid_count = int(title_text.str.fullmatch(r"[\W_]+").fillna(False).sum())
    invalid_pct = round(invalid_count / len(df) * 100, 2) if len(df) else 0

    if invalid_count > 0:
        rows.append({
            "column": "title",
            "issue": "Invalid title format",
            "count": invalid_count,
            "pct": invalid_pct,
            "severity": "MEDIUM",
        })

    too_short_count = int((title_text.str.len() < 5).sum())
    too_short_pct = round(too_short_count / len(df) * 100, 2) if len(df) else 0

    if too_short_count > 0:
        rows.append({
            "column": "title",
            "issue": "Unusually short titles",
            "count": too_short_count,
            "pct": too_short_pct,
            "severity": "LOW",
        })

    logger.info("Title quality check complete: %d issues", len(rows))

    return pd.DataFrame(rows)


def content_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect empty or unusually short content/description fields.
    """
    logger.info("Running content quality checks")

    rows = []

    content_col = None

    for candidate in ["content_text", "overview", "description", "text", "processed_text", "raw_text"]:
        if candidate in df.columns:
            content_col = candidate
            break

    if content_col is None:
        rows.append({
            "column": "content_text",
            "issue": "Missing content text column",
            "count": len(df),
            "pct": 100.0,
            "severity": "HIGH",
        })
        return pd.DataFrame(rows)

    content_text = df[content_col].fillna("").astype(str).str.strip()

    empty_count = int((content_text.str.len() == 0).sum())
    empty_pct = round(empty_count / len(df) * 100, 2) if len(df) else 0

    if empty_count > 0:
        rows.append({
            "column": content_col,
            "issue": "Empty content text",
            "count": empty_count,
            "pct": empty_pct,
            "severity": _severity_from_pct(empty_pct),
        })

    short_count = int((content_text.str.len() < 30).sum())
    short_pct = round(short_count / len(df) * 100, 2) if len(df) else 0

    if short_count > 0:
        rows.append({
            "column": content_col,
            "issue": "Unusually short content text",
            "count": short_count,
            "pct": short_pct,
            "severity": "MEDIUM" if short_pct > 10 else "LOW",
        })

    logger.info("Content quality check complete: %d issues", len(rows))

    return pd.DataFrame(rows)


def format_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect inconsistent news date/year formats.
    """
    logger.info("Running format consistency checks")

    rows = []

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

        original_non_null = df[col].notna().sum()
        parsed_dates = pd.to_datetime(df[col], errors="coerce")
        invalid_dates = int(original_non_null - parsed_dates.notna().sum())

        pct = round(invalid_dates / len(df) * 100, 2) if len(df) else 0

        if invalid_dates > 0:
            rows.append({
                "column": col,
                "issue": "Invalid date format",
                "count": invalid_dates,
                "pct": pct,
                "severity": "MEDIUM",
            })

    year_cols = [
        "published_year",
        "release_year",
        "year",
    ]

    for col in year_cols:
        if col not in df.columns:
            continue

        numeric_year = pd.to_numeric(df[col], errors="coerce")

        invalid_year_mask = numeric_year.notna() & (
            (numeric_year < 1900) | (numeric_year > 2100)
        )

        invalid_year_count = int(invalid_year_mask.sum())
        pct = round(invalid_year_count / len(df) * 100, 2) if len(df) else 0

        if invalid_year_count > 0:
            rows.append({
                "column": col,
                "issue": "Year outside expected news/document range",
                "count": invalid_year_count,
                "pct": pct,
                "severity": "MEDIUM",
            })

    logger.info("Format consistency check complete: %d issues", len(rows))

    return pd.DataFrame(rows)


def full_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all quality checks and return one combined issues DataFrame.

    Columns:
    column, issue, count, pct, severity
    """
    logger.info("Running full news data quality audit")

    issues = []

    # 1. Completeness
    missing_report = missing_value_report(df)

    for _, row in missing_report.iterrows():
        issues.append({
            "column": row["column"],
            "issue": "Missing values",
            "count": int(row["missing_count"]),
            "pct": float(row["missing_pct"]),
            "severity": row["severity"],
        })

    # 2. Zero-as-missing / incomplete numeric values
    zero_report = zero_as_missing(df)

    if not zero_report.empty:
        issues.extend(zero_report.to_dict("records"))

    # 3. Rating / score validity
    rating_report = rating_validity_report(df)

    if not rating_report.empty:
        issues.extend(rating_report.to_dict("records"))

    # 4. Duplicate IDs / URLs
    duplicate_report = duplicate_id_report(df)

    if not duplicate_report.empty:
        issues.extend(duplicate_report.to_dict("records"))

    # 5. Title quality
    title_report = title_quality_report(df)

    if not title_report.empty:
        issues.extend(title_report.to_dict("records"))

    # 6. Content quality
    content_report = content_quality_report(df)

    if not content_report.empty:
        issues.extend(content_report.to_dict("records"))

    # 7. Format consistency
    consistency_report = format_consistency_report(df)

    if not consistency_report.empty:
        issues.extend(consistency_report.to_dict("records"))

    # 8. Outliers
    outliers = outlier_report(df)

    for _, row in outliers.iterrows():
        issues.append({
            "column": row["column"],
            "issue": "IQR outliers detected",
            "count": int(row["outliers"]),
            "pct": float(row["outlier_pct"]),
            "severity": _severity_from_pct(float(row["outlier_pct"])),
        })

    quality_df = pd.DataFrame(issues)

    if not quality_df.empty:
        severity_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        quality_df["severity_rank"] = quality_df["severity"].map(severity_order).fillna(3)

        quality_df = quality_df.sort_values(
            ["severity_rank", "pct"],
            ascending=[True, False],
        ).drop(columns=["severity_rank"]).reset_index(drop=True)

    logger.info("Full news quality report complete: %d issues found", len(quality_df))

    return quality_df


def save_quality_report(
    quality_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/reports/full_quality_report.csv",
) -> None:
    """
    Save the full quality report as CSV.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    quality_df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Saved full quality report to CSV: %s", output_path)


def save_missing_heatmap(
    df: pd.DataFrame,
    output_path: str = "data/processed/analytics/charts/missing_values_heatmap.png",
) -> None:
    """
    Save a heatmap showing missing value patterns across a row sample.

    If no missing values exist, still save a simple chart stating that
    no missing values were detected. This keeps the Lab 8 output complete.
    """
    logger.info("Creating missing value heatmap")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        plt.figure(figsize=(8, 3))
        plt.text(
            0.5,
            0.5,
            "DataFrame is empty\nNo missing-value heatmap available",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.title("Missing Value Pattern - News Media Dataset")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()

        logger.warning("Saved empty DataFrame missing-value placeholder: %s", output_path)
        return

    cols_with_missing = df.columns[df.isna().any()].tolist()

    if not cols_with_missing:
        plt.figure(figsize=(8, 3))
        plt.text(
            0.5,
            0.5,
            "No missing values detected in the dataset",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.title("Missing Value Pattern - News Media Dataset")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()

        logger.info("No missing values found; saved placeholder heatmap: %s", output_path)
        return

    sample_size = min(200, len(df))
    sample = df[cols_with_missing].sample(sample_size, random_state=42)

    plt.figure(figsize=(12, 5))

    plt.imshow(
        sample.isna().T,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(label="Missing value: 1 = yes, 0 = no")
    plt.yticks(
        range(len(cols_with_missing)),
        cols_with_missing,
        fontsize=9,
    )

    plt.xlabel(f"Sample rows, n={len(sample)}")
    plt.ylabel("Columns")
    plt.title("Missing Value Pattern - News Media Dataset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved missing value heatmap: %s", output_path)