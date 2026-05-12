"""
src/analytics/insight_reporter.py

Insight reporting utilities for Lab 10 - News Media Monitoring Pipeline.

This module answers analytical questions using the combined MySQL + cleaned
metadata dataset and saves charts/reports for documentation.

Lab 10 requires at least 4 analytical questions with quantified findings.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.aggregator import (
    category_summary,
    document_type_summary,
    yearly_trends,
    top_n_per_group,
)
from src.analytics.pivot_builder import add_primary_category, add_analysis_year

logger = logging.getLogger(__name__)


def _safe_title_col(df: pd.DataFrame) -> str | None:
    """
    Return the best available title column after merges.
    """
    for col in ["title_mysql", "title_mongo", "title"]:
        if col in df.columns:
            return col
    return None


def _safe_document_col(df: pd.DataFrame) -> str:
    """
    Return the best available document type column after merges.
    """
    for col in ["document_type_mysql", "document_type_mongo", "document_type"]:
        if col in df.columns:
            return col
    return "document_type"


def prepare_insight_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare combined dataset for insight analysis.
    """
    df = df.copy()

    df = add_primary_category(df)
    df = add_analysis_year(df)

    numeric_cols = [
        "rating_score",
        "popularity",
        "content_length",
        "title_length",
        "engagement_score",
        "estimated_value",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    doc_col = _safe_document_col(df)

    if doc_col not in df.columns:
        df[doc_col] = "unknown"

    df["analysis_document_type"] = (
        df[doc_col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    )

    title_col = _safe_title_col(df)

    if title_col is None:
        df["analysis_title"] = "Untitled Record"
    else:
        df["analysis_title"] = (
            df[title_col]
            .fillna("Untitled Record")
            .astype(str)
            .str.strip()
            .replace({"": "Untitled Record"})
        )

    logger.info("Prepared insight dataset with shape=%s", df.shape)

    return df


def generate_insight_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quantified answers to analytical questions.

    Questions are adapted to the News Media Monitoring Pipeline.
    """
    df = prepare_insight_dataset(df)

    category_df = category_summary(df)
    document_df = document_type_summary(df, document_col="analysis_document_type")
    yearly_df = yearly_trends(df)
    top_articles_df = top_n_per_group(
        df,
        group_col="primary_category",
        sort_col="estimated_value",
        n=1,
    )

    insights = []

    # Q1: Which category has the highest total estimated value?
    if not category_df.empty:
        top_category = category_df.sort_values(
            "total_estimated_value",
            ascending=False,
        ).iloc[0]

        insights.append({
            "question": "Which news category has the highest total estimated value?",
            "answer": (
                f"The highest total estimated value belongs to "
                f"'{top_category['primary_category']}' with "
                f"{top_category['total_estimated_value']:.2f} total estimated value "
                f"across {int(top_category['article_count'])} records."
            ),
            "main_metric": "total_estimated_value",
            "metric_value": round(float(top_category["total_estimated_value"]), 2),
            "supporting_count": int(top_category["article_count"]),
        })

    # Q2: Which category has the highest average rating?
    if not category_df.empty:
        top_rating_category = category_df.sort_values(
            "avg_rating_score",
            ascending=False,
        ).iloc[0]

        insights.append({
            "question": "Which category has the highest average rating score?",
            "answer": (
                f"The category with the highest average rating score is "
                f"'{top_rating_category['primary_category']}' with an average "
                f"rating score of {top_rating_category['avg_rating_score']:.2f}."
            ),
            "main_metric": "avg_rating_score",
            "metric_value": round(float(top_rating_category["avg_rating_score"]), 2),
            "supporting_count": int(top_rating_category["article_count"]),
        })

    # Q3: Which document type appears most often?
    if not document_df.empty:
        top_doc = document_df.sort_values(
            "article_count",
            ascending=False,
        ).iloc[0]

        insights.append({
            "question": "Which document type appears most often in the dataset?",
            "answer": (
                f"The most frequent document type is "
                f"'{top_doc['analysis_document_type']}' with "
                f"{int(top_doc['article_count'])} records."
            ),
            "main_metric": "article_count",
            "metric_value": int(top_doc["article_count"]),
            "supporting_count": int(top_doc["article_count"]),
        })

    # Q4: Which individual record has the highest estimated value?
    if not df.empty:
        top_record = df.sort_values(
            "estimated_value",
            ascending=False,
        ).iloc[0]

        insights.append({
            "question": "Which individual record has the highest estimated value?",
            "answer": (
                f"The record with the highest estimated value is "
                f"'{top_record['analysis_title']}' "
                f"(record_id={top_record['record_id']}) with "
                f"{top_record['estimated_value']:.2f} estimated value."
            ),
            "main_metric": "estimated_value",
            "metric_value": round(float(top_record["estimated_value"]), 2),
            "supporting_count": 1,
        })

    # Q5: What is the main yearly trend?
    if not yearly_df.empty:
        top_year = yearly_df.sort_values(
            "article_count",
            ascending=False,
        ).iloc[0]

        insights.append({
            "question": "Which year has the most dated news records?",
            "answer": (
                f"The year with the most dated records is "
                f"{int(top_year['analysis_year'])}, with "
                f"{int(top_year['article_count'])} dated records."
            ),
            "main_metric": "article_count_by_year",
            "metric_value": int(top_year["article_count"]),
            "supporting_count": int(top_year["article_count"]),
        })

    # Q6: How many category leaders are produced by top-N analysis?
    if not top_articles_df.empty:
        insights.append({
            "question": "How many top category-level records were identified?",
            "answer": (
                f"The top-N category analysis identified "
                f"{len(top_articles_df)} leading records across categories."
            ),
            "main_metric": "top_records_per_category",
            "metric_value": int(len(top_articles_df)),
            "supporting_count": int(len(top_articles_df)),
        })

    insight_df = pd.DataFrame(insights)

    logger.info("Generated %d analytical insights", len(insight_df))

    return insight_df


def save_top_categories_chart(
    category_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/top_categories_estimated_value.png",
    top_n: int = 10,
) -> str:
    """
    Save chart: top categories by total estimated value.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if category_df.empty:
        logger.warning("save_top_categories_chart skipped because category_df is empty")
        return str(output_path)

    plot_df = category_df.sort_values(
        "total_estimated_value",
        ascending=False,
    ).head(top_n)

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["primary_category"], plot_df["total_estimated_value"])
    plt.title("Top Categories by Total Estimated Value")
    plt.xlabel("Category")
    plt.ylabel("Total Estimated Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved top categories chart to %s", output_path)

    return str(output_path)


def save_document_type_chart(
    document_df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/document_type_counts.png",
    top_n: int = 10,
) -> str:
    """
    Save chart: document type counts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if document_df.empty:
        logger.warning("save_document_type_chart skipped because document_df is empty")
        return str(output_path)

    doc_col = "analysis_document_type"

    if doc_col not in document_df.columns:
        doc_col = document_df.columns[0]

    plot_df = document_df.sort_values(
        "article_count",
        ascending=False,
    ).head(top_n)

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df[doc_col], plot_df["article_count"])
    plt.title("Top Document Types by Record Count")
    plt.xlabel("Document Type")
    plt.ylabel("Record Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved document type chart to %s", output_path)

    return str(output_path)


def save_rating_value_scatter(
    df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/rating_vs_estimated_value.png",
) -> str:
    """
    Save scatter chart: rating_score vs estimated_value.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_insight_dataset(df)

    plt.figure(figsize=(9, 6))
    plt.scatter(df["rating_score"], df["estimated_value"], alpha=0.6)
    plt.title("Rating Score vs Estimated Value")
    plt.xlabel("Rating Score")
    plt.ylabel("Estimated Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved rating vs estimated value scatter to %s", output_path)

    return str(output_path)


def save_top_articles_chart(
    df: pd.DataFrame,
    output_path: str = "data/processed/analytics/lab10/top_articles_estimated_value.png",
    top_n: int = 10,
) -> str:
    """
    Save chart: top records/articles by estimated value.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_insight_dataset(df)

    plot_df = df.sort_values(
        "estimated_value",
        ascending=False,
    ).head(top_n).copy()

    plot_df["short_title"] = plot_df["analysis_title"].str.slice(0, 35)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["short_title"], plot_df["estimated_value"])
    plt.title("Top Records by Estimated Value")
    plt.xlabel("Record Title")
    plt.ylabel("Estimated Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved top articles chart to %s", output_path)

    return str(output_path)


def save_insight_outputs(
    df: pd.DataFrame,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Generate all Lab 10 insight reports and charts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = prepare_insight_dataset(df)

    category_df = category_summary(prepared_df)
    document_df = document_type_summary(
        prepared_df,
        document_col="analysis_document_type",
    )
    yearly_df = yearly_trends(prepared_df)
    insight_df = generate_insight_questions(prepared_df)

    insight_path = output_dir / "analytical_questions_report.csv"
    category_path = output_dir / "insight_category_summary.csv"
    document_path = output_dir / "insight_document_type_summary.csv"
    yearly_path = output_dir / "insight_yearly_summary.csv"

    insight_df.to_csv(insight_path, index=False)
    category_df.to_csv(category_path, index=False)
    document_df.to_csv(document_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)

    chart_paths = {
        "top_categories_chart": save_top_categories_chart(
            category_df,
            str(output_dir / "top_categories_estimated_value.png"),
        ),
        "document_type_chart": save_document_type_chart(
            document_df,
            str(output_dir / "document_type_counts.png"),
        ),
        "rating_value_scatter": save_rating_value_scatter(
            prepared_df,
            str(output_dir / "rating_vs_estimated_value.png"),
        ),
        "top_articles_chart": save_top_articles_chart(
            prepared_df,
            str(output_dir / "top_articles_estimated_value.png"),
        ),
    }

    paths = {
        "analytical_questions_report": str(insight_path),
        "insight_category_summary": str(category_path),
        "insight_document_type_summary": str(document_path),
        "insight_yearly_summary": str(yearly_path),
    }

    paths.update(chart_paths)

    logger.info("Saved all insight outputs to %s", output_dir)

    return paths