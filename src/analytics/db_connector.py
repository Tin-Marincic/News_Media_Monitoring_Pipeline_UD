"""
src/analytics/db_connector.py

MySQL connector for Lab 10 - News Media Monitoring Pipeline.

This module:
- connects to MySQL using PyMySQL
- creates the news analytics database
- creates/recreates the news_article_metrics table
- prepares cleaned news data for SQL storage
- inserts records using parameterized queries
- queries the table back into pandas using pd.read_sql()
"""

import logging
import os
from typing import Optional

import pandas as pd
import pymysql
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "news_analytics")

TABLE_NAME = "news_article_metrics"

def get_connection(database: Optional[str] = MYSQL_DATABASE):
    """
    Open a PyMySQL connection.

    If database=None, connects without selecting a database.
    This is useful when creating the database for the first time.
    """
    connection_kwargs = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "charset": "utf8mb4",
        "autocommit": False,
    }

    if database:
        connection_kwargs["database"] = database

    logger.info(
        "Opening MySQL connection host=%s port=%s database=%s",
        MYSQL_HOST,
        MYSQL_PORT,
        database,
    )

    return pymysql.connect(**connection_kwargs)


def create_database(database: str = MYSQL_DATABASE) -> None:
    """
    Create the MySQL database if it does not already exist.
    """
    connection = get_connection(database=None)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE DATABASE IF NOT EXISTS `{database}`
                CHARACTER SET utf8mb4
                COLLATE utf8mb4_unicode_ci
                """
            )

        connection.commit()
        logger.info("Database ensured: %s", database)

    finally:
        connection.close()


def create_article_metrics_table(
    connection,
    table_name: str = TABLE_NAME,
) -> None:
    """
    Create a table for cleaned news/article metrics.

    This adapts the Lab 10 MySQL requirement to the News Media Monitoring domain.
    """
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        record_id BIGINT PRIMARY KEY,
        title TEXT NOT NULL,
        primary_category VARCHAR(255),
        document_type VARCHAR(255),
        source_name VARCHAR(255),
        published_date DATETIME NULL,
        published_year INT NULL,
        rating_score FLOAT NULL,
        popularity FLOAT NULL,
        content_length INT NULL,
        title_length INT NULL,
        engagement_score FLOAT NULL,
        estimated_value FLOAT NULL
    )
    """

    with connection.cursor() as cursor:
        cursor.execute(create_sql)

    connection.commit()
    logger.info("MySQL table ensured: %s", table_name)


def reset_article_metrics_table(
    connection,
    table_name: str = TABLE_NAME,
) -> None:
    """
    Drop and recreate the article metrics table.

    This is useful when an old table exists with wrong schema or bad rows.
    """
    with connection.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")

    connection.commit()
    logger.info("Dropped old MySQL table if it existed: %s", table_name)

    create_article_metrics_table(connection, table_name=table_name)


def prepare_article_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the cleaned news DataFrame for MySQL insertion.

    Since this is a news project, we create analytics-style metrics:
    - primary_category from category/genres/document_type
    - engagement_score from rating_score, popularity, and content_length
    - estimated_value as a news-domain equivalent of revenue-like value
    """
    logger.info("Preparing article metrics for MySQL insertion")

    metrics_df = df.copy()

    # Remove accidental duplicated CSV header rows if they exist in the dataset.
    if "record_id" in metrics_df.columns:
        header_record_mask = metrics_df["record_id"].astype(str).str.lower().eq("record_id")
        header_title_mask = (
            metrics_df["title"].astype(str).str.lower().eq("title")
            if "title" in metrics_df.columns
            else False
        )

        if hasattr(header_title_mask, "__len__"):
            header_like_mask = header_record_mask | header_title_mask
        else:
            header_like_mask = header_record_mask

        removed_headers = int(header_like_mask.sum())
        if removed_headers > 0:
            metrics_df = metrics_df[~header_like_mask].copy()
            logger.info("Removed %d accidental header-like rows", removed_headers)

    if "record_id" not in metrics_df.columns:
        metrics_df["record_id"] = range(1, len(metrics_df) + 1)

    if "title" not in metrics_df.columns:
        metrics_df["title"] = "Untitled Record"

    if "category" in metrics_df.columns:
        metrics_df["primary_category"] = metrics_df["category"]
    elif "genres" in metrics_df.columns:
        metrics_df["primary_category"] = metrics_df["genres"]
    elif "document_type" in metrics_df.columns:
        metrics_df["primary_category"] = metrics_df["document_type"]
    else:
        metrics_df["primary_category"] = "unknown"

    if "document_type" not in metrics_df.columns:
        metrics_df["document_type"] = "unknown"

    if "source_name" not in metrics_df.columns:
        metrics_df["source_name"] = "unknown"

    if "published_date" not in metrics_df.columns:
        metrics_df["published_date"] = pd.NaT

    if "published_year" not in metrics_df.columns:
        metrics_df["published_year"] = pd.to_datetime(
            metrics_df["published_date"],
            errors="coerce",
        ).dt.year

    numeric_defaults = {
        "rating_score": 0.0,
        "popularity": 0.0,
        "content_length": 0,
        "title_length": 0,
    }

    for col, default in numeric_defaults.items():
        if col not in metrics_df.columns:
            metrics_df[col] = default

        metrics_df[col] = pd.to_numeric(
            metrics_df[col],
            errors="coerce",
        ).fillna(default)

    metrics_df["record_id"] = pd.to_numeric(
        metrics_df["record_id"],
        errors="coerce",
    )

    metrics_df["published_date"] = pd.to_datetime(
        metrics_df["published_date"],
        errors="coerce",
    )

    metrics_df["published_year"] = pd.to_numeric(
        metrics_df["published_year"],
        errors="coerce",
    )

    metrics_df["title"] = metrics_df["title"].fillna("Untitled Record").astype(str)
    metrics_df["primary_category"] = metrics_df["primary_category"].fillna("unknown").astype(str)
    metrics_df["document_type"] = metrics_df["document_type"].fillna("unknown").astype(str)
    metrics_df["source_name"] = metrics_df["source_name"].fillna("unknown").astype(str)

    # Domain-specific derived metrics for Lab 10 analysis.
    metrics_df["engagement_score"] = (
        metrics_df["rating_score"].fillna(0) * 10
        + metrics_df["popularity"].fillna(0) * 0.05
        + metrics_df["content_length"].fillna(0) * 0.01
    )

    metrics_df["estimated_value"] = (
        metrics_df["engagement_score"].fillna(0)
        * (metrics_df["content_length"].fillna(0) + 1)
    )

    output_cols = [
        "record_id",
        "title",
        "primary_category",
        "document_type",
        "source_name",
        "published_date",
        "published_year",
        "rating_score",
        "popularity",
        "content_length",
        "title_length",
        "engagement_score",
        "estimated_value",
    ]

    metrics_df = metrics_df[output_cols].copy()

    # Remove invalid/header-like rows before converting record_id to integer.
    header_like_mask = (
        metrics_df["title"].str.lower().eq("title")
        | metrics_df["primary_category"].str.lower().eq("primary_category")
        | metrics_df["document_type"].str.lower().eq("document_type")
        | metrics_df["source_name"].str.lower().eq("source_name")
    )

    valid_mask = (
        metrics_df["record_id"].notna()
        & metrics_df["title"].str.strip().ne("")
        & ~header_like_mask
    )

    metrics_df = metrics_df[valid_mask].copy()

    metrics_df["record_id"] = metrics_df["record_id"].astype("int64")

    metrics_df = metrics_df.drop_duplicates(subset=["record_id"], keep="first")

    logger.info("Prepared %d article metric rows for MySQL", len(metrics_df))

    return metrics_df


def populate_article_metrics(
    connection,
    df: pd.DataFrame,
    table_name: str = TABLE_NAME,
    clear_existing: bool = True,
) -> int:
    """
    Insert cleaned article metrics into MySQL using parameterized queries.
    """
    metrics_df = prepare_article_metrics(df)

    create_article_metrics_table(connection, table_name=table_name)

    if clear_existing:
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM `{table_name}`")
        connection.commit()
        logger.info("Cleared existing rows from %s", table_name)

    insert_sql = f"""
    INSERT INTO `{table_name}` (
        record_id,
        title,
        primary_category,
        document_type,
        source_name,
        published_date,
        published_year,
        rating_score,
        popularity,
        content_length,
        title_length,
        engagement_score,
        estimated_value
    )
    VALUES (
        %(record_id)s,
        %(title)s,
        %(primary_category)s,
        %(document_type)s,
        %(source_name)s,
        %(published_date)s,
        %(published_year)s,
        %(rating_score)s,
        %(popularity)s,
        %(content_length)s,
        %(title_length)s,
        %(engagement_score)s,
        %(estimated_value)s
    )
    ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        primary_category = VALUES(primary_category),
        document_type = VALUES(document_type),
        source_name = VALUES(source_name),
        published_date = VALUES(published_date),
        published_year = VALUES(published_year),
        rating_score = VALUES(rating_score),
        popularity = VALUES(popularity),
        content_length = VALUES(content_length),
        title_length = VALUES(title_length),
        engagement_score = VALUES(engagement_score),
        estimated_value = VALUES(estimated_value)
    """

    records = []

    for _, row in metrics_df.iterrows():
        published_date = row["published_date"]

        if pd.isna(published_date):
            published_date_value = None
        else:
            published_date_value = published_date.to_pydatetime()

        published_year = row["published_year"]
        published_year_value = None if pd.isna(published_year) else int(published_year)

        records.append({
            "record_id": int(row["record_id"]),
            "title": str(row["title"]),
            "primary_category": str(row["primary_category"]),
            "document_type": str(row["document_type"]),
            "source_name": str(row["source_name"]),
            "published_date": published_date_value,
            "published_year": published_year_value,
            "rating_score": float(row["rating_score"]) if pd.notna(row["rating_score"]) else None,
            "popularity": float(row["popularity"]) if pd.notna(row["popularity"]) else None,
            "content_length": int(row["content_length"]) if pd.notna(row["content_length"]) else None,
            "title_length": int(row["title_length"]) if pd.notna(row["title_length"]) else None,
            "engagement_score": float(row["engagement_score"]) if pd.notna(row["engagement_score"]) else None,
            "estimated_value": float(row["estimated_value"]) if pd.notna(row["estimated_value"]) else None,
        })

    if not records:
        logger.warning("No valid records available for MySQL insertion")
        return 0

    with connection.cursor() as cursor:
        cursor.executemany(insert_sql, records)

    connection.commit()

    logger.info("Inserted/updated %d rows into %s", len(records), table_name)

    return len(records)


def query_article_metrics(
    connection,
    table_name: str = TABLE_NAME,
    min_estimated_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Query article metrics back from MySQL into a pandas DataFrame.

    Demonstrates SQL filtering through WHERE clauses.
    """
    sql = f"SELECT * FROM `{table_name}`"
    params = None

    if min_estimated_value is not None:
        sql += " WHERE estimated_value >= %s"
        params = [min_estimated_value]

    sql += " ORDER BY estimated_value DESC"

    df = pd.read_sql(sql, connection, params=params)

    logger.info("Queried %d rows from %s", len(df), table_name)

    return df


def setup_mysql_from_cleaned_data(
    cleaned_csv_path: str = "data/processed/cleaned/cleaned_data.csv",
    database: str = MYSQL_DATABASE,
    table_name: str = TABLE_NAME,
    reset_table: bool = True,
) -> pd.DataFrame:
    """
    Convenience function:
    - create database
    - connect
    - create or reset table
    - populate table from cleaned CSV
    - query rows back into pandas

    reset_table=True is intentional for Lab 10 so old dirty tables do not
    affect notebook/pipeline outputs.
    """
    logger.info("Setting up MySQL from cleaned CSV: %s", cleaned_csv_path)

    create_database(database)

    cleaned_df = pd.read_csv(cleaned_csv_path)

    connection = get_connection(database=database)

    try:
        if reset_table:
            reset_article_metrics_table(connection, table_name=table_name)
        else:
            create_article_metrics_table(connection, table_name=table_name)

        inserted = populate_article_metrics(
            connection,
            cleaned_df,
            table_name=table_name,
            clear_existing=True,
        )

        logger.info("Inserted %d cleaned rows into MySQL", inserted)

        queried_df = query_article_metrics(
            connection,
            table_name=table_name,
        )

        return queried_df

    finally:
        connection.close()
        logger.info("MySQL connection closed")