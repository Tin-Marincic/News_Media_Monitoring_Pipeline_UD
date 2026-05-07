"""
src/analytics/data_combiner.py

Data combining utilities for Lab 10 - News Media Monitoring Pipeline.

This module adapts the professor's movie-based MySQL/MongoDB merge example
to the News Media Monitoring Pipeline.

Main key:
- record_id

Main sources:
- MySQL metrics table: news_article_metrics
- cleaned CSV / MongoDB-style metadata: cleaned_data.csv
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare metadata from the cleaned CSV / MongoDB-style DataFrame.

    This keeps descriptive/news fields that are not all stored in MySQL.
    """
    metadata_df = df.copy()

    if "record_id" not in metadata_df.columns:
        metadata_df["record_id"] = range(1, len(metadata_df) + 1)

    desired_cols = [
        "record_id",
        "title",
        "description",
        "url",
        "source_path",
        "file_name",
        "category",
        "document_type",
        "source_name",
        "language",
        "content_text",
        "published_date",
        "published_year",
    ]

    existing_cols = [col for col in desired_cols if col in metadata_df.columns]

    metadata_df = metadata_df[existing_cols].copy()

    metadata_df["record_id"] = pd.to_numeric(
        metadata_df["record_id"],
        errors="coerce",
    )

    metadata_df = metadata_df.dropna(subset=["record_id"])
    metadata_df["record_id"] = metadata_df["record_id"].astype("int64")

    metadata_df = metadata_df.drop_duplicates(subset=["record_id"], keep="first")

    logger.info("Prepared metadata DataFrame with shape=%s", metadata_df.shape)

    return metadata_df


def merge_mysql_mongodb(
    mysql_df: pd.DataFrame,
    mongo_df: pd.DataFrame,
    on: str = "record_id",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Professor-style function name.

    Merge MySQL metrics data with MongoDB/cleaned metadata data.
    For this project, the shared key is record_id instead of tmdb_id.
    """
    logger.info(
        'Merging MySQL (%d rows) and metadata/MongoDB (%d rows) on "%s" with how="%s"',
        len(mysql_df),
        len(mongo_df),
        on,
        how,
    )

    mysql_df = mysql_df.copy()
    mongo_df = mongo_df.copy()

    mysql_df[on] = pd.to_numeric(mysql_df[on], errors="coerce")
    mongo_df[on] = pd.to_numeric(mongo_df[on], errors="coerce")

    mysql_df = mysql_df.dropna(subset=[on])
    mongo_df = mongo_df.dropna(subset=[on])

    mysql_df[on] = mysql_df[on].astype("int64")
    mongo_df[on] = mongo_df[on].astype("int64")

    merged = pd.merge(
        mysql_df,
        mongo_df,
        on=on,
        how=how,
        suffixes=("_mysql", "_mongo"),
    )

    logger.info(
        "Merged result: %d rows, %d columns",
        len(merged),
        len(merged.columns),
    )

    return merged


def merge_metadata_with_metrics(
    metadata_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    how: str = "inner",
    key: str = "record_id",
) -> pd.DataFrame:
    """
    More descriptive project-specific wrapper.

    Metadata = cleaned CSV / MongoDB-style data.
    Metrics = MySQL news_article_metrics table.
    """
    return merge_mysql_mongodb(
        mysql_df=metrics_df,
        mongo_df=metadata_df,
        on=key,
        how=how,
    )


def demonstrate_join_types(
    mysql_df: pd.DataFrame,
    mongo_df: pd.DataFrame,
    on: str = "record_id",
) -> dict:
    """
    Professor-style function name.

    Demonstrate inner, left, right, and outer joins and return row counts.
    """
    results = {}

    for how in ["inner", "left", "right", "outer"]:
        merged = merge_mysql_mongodb(
            mysql_df=mysql_df,
            mongo_df=mongo_df,
            on=on,
            how=how,
        )

        results[how] = len(merged)

        print(f"  {how:6s} join -> {len(merged):5d} rows")

    return results


def compare_join_types(
    metadata_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    key: str = "record_id",
) -> pd.DataFrame:
    """
    Return join comparison as a DataFrame for saving/displaying.
    """
    join_counts = demonstrate_join_types(
        mysql_df=metrics_df,
        mongo_df=metadata_df,
        on=key,
    )

    result = pd.DataFrame([
        {
            "join_type": join_type,
            "row_count": row_count,
        }
        for join_type, row_count in join_counts.items()
    ])

    logger.info("Join type comparison complete")

    return result


def concat_dataframes(
    dfs: list[pd.DataFrame],
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Professor-style function name.

    Concatenate DataFrames with the same or similar schema.
    """
    combined = pd.concat(
        dfs,
        axis=0,
        ignore_index=reset_index,
    )

    logger.info(
        "Concatenated %d DataFrames into %d rows",
        len(dfs),
        len(combined),
    )

    return combined


def concatenate_same_schema(
    dataframes: list[pd.DataFrame],
    ignore_index: bool = True,
) -> pd.DataFrame:
    """
    Project-specific wrapper around concat_dataframes().
    """
    return concat_dataframes(
        dataframes,
        reset_index=ignore_index,
    )


def save_join_count_chart(
    join_counts_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/join_type_row_counts.png",
) -> str:
    """
    Save a bar chart comparing row counts for each join type.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(join_counts_df["join_type"], join_counts_df["row_count"])
    plt.title("Join Type Row Count Comparison")
    plt.xlabel("Join Type")
    plt.ylabel("Row Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved join count chart to %s", output_path)

    return str(output_path)


def save_combined_outputs(
    combined_df: pd.DataFrame,
    join_counts_df: pd.DataFrame,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Save combined analysis outputs to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "combined_news_metrics.csv"
    join_counts_path = output_dir / "join_type_row_counts.csv"

    combined_df.to_csv(combined_path, index=False)
    join_counts_df.to_csv(join_counts_path, index=False)

    logger.info("Saved combined DataFrame to %s", combined_path)
    logger.info("Saved join counts to %s", join_counts_path)

    return {
        "combined_path": str(combined_path),
        "join_counts_path": str(join_counts_path),
    }