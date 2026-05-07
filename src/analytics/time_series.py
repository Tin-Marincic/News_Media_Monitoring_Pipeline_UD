"""
src/analytics/time_series.py

Time series utilities for Lab 10 - News Media Monitoring Pipeline.

Movie example:
- release_date
- monthly revenue
- yearly revenue
- rolling revenue averages

News equivalent:
- published_date
- monthly article estimated value / engagement
- yearly article trends
- rolling estimated value averages
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def parse_news_dates(
    df: pd.DataFrame,
    date_col: str | None = None,
    output_col: str = "analysis_date",
) -> pd.DataFrame:
    """
    Parse the best available date column into datetime64.

    Priority:
    published_date_mysql -> published_date -> published_date_mongo
    -> publishedAt -> fetched_at -> extraction_timestamp
    """
    df = df.copy()

    if date_col is None:
        candidates = [
            "published_date_mysql",
            "published_date",
            "published_date_mongo",
            "publishedAt",
            "fetched_at",
            "extraction_timestamp",
            "release_date",
        ]

        date_col = next((col for col in candidates if col in df.columns), None)

    if date_col is None:
        logger.warning("parse_news_dates: no date column found")
        df[output_col] = pd.NaT
        return df

    df[output_col] = pd.to_datetime(df[date_col], errors="coerce")

    valid_count = int(df[output_col].notna().sum())
    missing_count = int(df[output_col].isna().sum())

    logger.info(
        "parse_news_dates: parsed %s into %s; valid=%d missing=%d",
        date_col,
        output_col,
        valid_count,
        missing_count,
    )

    return df


def add_date_components(
    df: pd.DataFrame,
    date_col: str = "analysis_date",
) -> pd.DataFrame:
    """
    Extract date components from a datetime column.

    Adds:
    - analysis_year
    - analysis_month
    - analysis_month_name
    - analysis_weekday
    - analysis_weekday_name
    - analysis_quarter
    """
    df = df.copy()

    if date_col not in df.columns:
        df = parse_news_dates(df, output_col=date_col)

    dates = pd.to_datetime(df[date_col], errors="coerce")

    df["analysis_year"] = dates.dt.year.astype("Int64")
    df["analysis_month"] = dates.dt.month.astype("Int64")
    df["analysis_month_name"] = dates.dt.month_name()
    df["analysis_weekday"] = dates.dt.weekday.astype("Int64")
    df["analysis_weekday_name"] = dates.dt.day_name()
    df["analysis_quarter"] = dates.dt.quarter.astype("Int64")

    logger.info(
        "add_date_components: added components from %s with %d valid dates",
        date_col,
        dates.notna().sum(),
    )

    return df


def build_monthly_time_series(
    df: pd.DataFrame,
    date_col: str = "analysis_date",
    value_col: str = "estimated_value",
) -> pd.DataFrame:
    """
    Build monthly time series.

    Aggregates:
    - article_count
    - total value
    - average value
    - median value
    """
    df = df.copy()

    if date_col not in df.columns:
        df = parse_news_dates(df, output_col=date_col)

    if value_col not in df.columns:
        raise KeyError(f"Missing value column: {value_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    valid = df.dropna(subset=[date_col]).copy()

    if valid.empty:
        logger.warning("build_monthly_time_series: no valid dates available")
        return pd.DataFrame(
            columns=[
                "analysis_date",
                "article_count",
                f"total_{value_col}",
                f"avg_{value_col}",
                f"median_{value_col}",
            ]
        )

    valid = valid.set_index(date_col).sort_index()

    monthly = (
        valid.resample("ME")
        .agg(
            article_count=("record_id", "count"),
            total_value=(value_col, "sum"),
            avg_value=(value_col, "mean"),
            median_value=(value_col, "median"),
        )
        .reset_index()
    )

    monthly = monthly.rename(
        columns={
            "total_value": f"total_{value_col}",
            "avg_value": f"avg_{value_col}",
            "median_value": f"median_{value_col}",
        }
    )

    logger.info(
        "build_monthly_time_series: created monthly table with shape=%s",
        monthly.shape,
    )

    return monthly


def resample_yearly(
    df: pd.DataFrame,
    date_col: str = "analysis_date",
    value_col: str = "estimated_value",
) -> pd.DataFrame:
    """
    Resample the data to yearly totals.
    """
    df = df.copy()

    if date_col not in df.columns:
        df = parse_news_dates(df, output_col=date_col)

    if value_col not in df.columns:
        raise KeyError(f"Missing value column: {value_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    valid = df.dropna(subset=[date_col]).copy()

    if valid.empty:
        logger.warning("resample_yearly: no valid dates available")
        return pd.DataFrame(
            columns=["analysis_date", "article_count", f"total_{value_col}"]
        )

    valid = valid.set_index(date_col).sort_index()

    yearly = (
        valid.resample("YE")
        .agg(
            article_count=("record_id", "count"),
            total_value=(value_col, "sum"),
            avg_value=(value_col, "mean"),
        )
        .reset_index()
    )

    yearly = yearly.rename(
        columns={
            "total_value": f"total_{value_col}",
            "avg_value": f"avg_{value_col}",
        }
    )

    logger.info("resample_yearly: created yearly table with shape=%s", yearly.shape)

    return yearly


def resample_quarterly(
    df: pd.DataFrame,
    date_col: str = "analysis_date",
    value_col: str = "estimated_value",
) -> pd.DataFrame:
    """
    Resample the data to quarterly totals.
    """
    df = df.copy()

    if date_col not in df.columns:
        df = parse_news_dates(df, output_col=date_col)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    valid = df.dropna(subset=[date_col]).copy()

    if valid.empty:
        logger.warning("resample_quarterly: no valid dates available")
        return pd.DataFrame(
            columns=["analysis_date", "article_count", f"total_{value_col}"]
        )

    valid = valid.set_index(date_col).sort_index()

    quarterly = (
        valid.resample("QE")
        .agg(
            article_count=("record_id", "count"),
            total_value=(value_col, "sum"),
            avg_value=(value_col, "mean"),
        )
        .reset_index()
    )

    quarterly = quarterly.rename(
        columns={
            "total_value": f"total_{value_col}",
            "avg_value": f"avg_{value_col}",
        }
    )

    logger.info(
        "resample_quarterly: created quarterly table with shape=%s",
        quarterly.shape,
    )

    return quarterly


def add_rolling_averages(
    monthly_df: pd.DataFrame,
    value_col: str = "total_estimated_value",
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Add rolling averages to a monthly time series.

    Lab 10 requires rolling averages with window sizes 3, 6, and 12.
    """
    df = monthly_df.copy()

    if df.empty:
        logger.warning("add_rolling_averages skipped because monthly_df is empty")
        return df

    if value_col not in df.columns:
        raise KeyError(f"Missing rolling value column: {value_col}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    for window in windows:
        df[f"rolling_{window}_month_avg"] = (
            df[value_col]
            .rolling(window=window, min_periods=1)
            .mean()
        )

    logger.info(
        "add_rolling_averages: added rolling windows=%s using %s",
        windows,
        value_col,
    )

    return df


def save_time_series_chart(
    monthly_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/rolling_estimated_value.png",
    date_col: str = "analysis_date",
    value_col: str = "total_estimated_value",
) -> str:
    """
    Save a time series chart with rolling averages.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if monthly_df.empty:
        logger.warning("save_time_series_chart skipped because monthly_df is empty")
        return str(output_path)

    df = monthly_df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    plt.figure(figsize=(11, 6))
    plt.plot(df[date_col], df[value_col], marker="o", label=value_col)

    for col in ["rolling_3_month_avg", "rolling_6_month_avg", "rolling_12_month_avg"]:
        if col in df.columns:
            plt.plot(df[date_col], df[col], marker="o", label=col)

    plt.title("Monthly News Estimated Value with Rolling Averages")
    plt.xlabel("Month")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved time series chart to %s", output_path)

    return str(output_path)


def save_time_series_outputs(
    monthly_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Save time series outputs to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = output_dir / "monthly_time_series.csv"
    yearly_path = output_dir / "yearly_time_series.csv"
    quarterly_path = output_dir / "quarterly_time_series.csv"

    monthly_df.to_csv(monthly_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    quarterly_df.to_csv(quarterly_path, index=False)

    logger.info("Saved monthly time series to %s", monthly_path)
    logger.info("Saved yearly time series to %s", yearly_path)
    logger.info("Saved quarterly time series to %s", quarterly_path)

    return {
        "monthly_time_series": str(monthly_path),
        "yearly_time_series": str(yearly_path),
        "quarterly_time_series": str(quarterly_path),
    }