"""
src/analytics/mongo_pipeline.py

MongoDB aggregation pipeline utilities for Lab 10 - News Media Monitoring Pipeline.

This module demonstrates MongoDB server-side aggregation using:
- $match
- $group
- $sort
- $project

The movie lab example groups movies by genre/rating.
For this news project, we group news/media records by category,
document_type, and source_name.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "news_pipeline")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "raw_articles")


def get_mongo_client(uri: str = MONGO_URI) -> MongoClient:
    """
    Create a MongoDB client.
    """
    logger.info("Opening MongoDB client: %s", uri)
    return MongoClient(uri)


def get_collection(
    database_name: str = MONGO_DATABASE,
    collection_name: str = MONGO_COLLECTION,
):
    """
    Return the configured MongoDB collection.
    """
    client = get_mongo_client()
    db = client[database_name]
    collection = db[collection_name]

    logger.info(
        "Using MongoDB collection: %s.%s",
        database_name,
        collection_name,
    )

    return client, collection

def build_category_aggregation_pipeline(
    min_content_length: int = 0,
) -> list:
    """
    Build a MongoDB aggregation pipeline grouped by category/document_type.

    This version is adapted to the News Media Monitoring Pipeline, where many
    MongoDB records do not have a title/content_text field but do have
    document_type, source, name, year, wins, losses, etc.

    Required Lab 10 stages included:
    - $match
    - $project
    - $group
    - $sort
    - $project
    """
    pipeline = [
        {
            "$match": {
                "$or": [
                    {"category": {"$exists": True}},
                    {"document_type": {"$exists": True}},
                    {"source_name": {"$exists": True}},
                    {"source": {"$exists": True}},
                    {"source_path": {"$exists": True}},
                    {"name": {"$exists": True}},
                ]
            }
        },
        {
            "$project": {
                "category_group": {
                    "$ifNull": [
                        "$category",
                        {
                            "$ifNull": [
                                "$document_type",
                                "unknown",
                            ]
                        },
                    ]
                },
                "document_type": {
                    "$ifNull": [
                        "$document_type",
                        "unknown",
                    ]
                },
                "source_group": {
                    "$ifNull": [
                        "$source_name",
                        {
                            "$ifNull": [
                                "$source",
                                {
                                    "$ifNull": [
                                        "$source_path",
                                        "unknown",
                                    ]
                                },
                            ]
                        },
                    ]
                },
                "title_available": {
                    "$cond": [
                        {
                            "$or": [
                                {"$ne": [{"$ifNull": ["$title", ""]}, ""]},
                                {"$ne": [{"$ifNull": ["$name", ""]}, ""]},
                            ]
                        },
                        1,
                        0,
                    ]
                },
                "text_available": {
                    "$cond": [
                        {
                            "$or": [
                                {"$ne": [{"$ifNull": ["$text", ""]}, ""]},
                                {"$ne": [{"$ifNull": ["$raw_text", ""]}, ""]},
                                {"$ne": [{"$ifNull": ["$processed_text", ""]}, ""]},
                                {"$ne": [{"$ifNull": ["$preview_text", ""]}, ""]},
                            ]
                        },
                        1,
                        0,
                    ]
                },
            }
        },
        {
            "$match": {
                "category_group": {
                    "$ne": None
                }
            }
        },
        {
            "$group": {
                "_id": "$category_group",
                "record_count": {
                    "$sum": 1
                },
                "titles_available": {
                    "$sum": "$title_available"
                },
                "text_records_available": {
                    "$sum": "$text_available"
                },
                "unique_document_types": {
                    "$addToSet": "$document_type"
                },
                "sample_source": {
                    "$first": "$source_group"
                },
            }
        },
        {
            "$sort": {
                "record_count": -1
            }
        },
        {
            "$project": {
                "_id": 0,
                "category": "$_id",
                "record_count": 1,
                "titles_available": 1,
                "text_records_available": 1,
                "unique_document_type_count": {
                    "$size": "$unique_document_types"
                },
                "sample_source": 1,
            }
        },
    ]

    return pipeline


def build_document_type_pipeline() -> list:
    """
    Build aggregation grouped by document_type.
    """
    return [
        {
            "$match": {
                "document_type": {"$exists": True}
            }
        },
        {
            "$group": {
                "_id": "$document_type",
                "record_count": {"$sum": 1},
                "titles_available": {
                    "$sum": {
                        "$cond": [
                            {"$ifNull": ["$title", False]},
                            1,
                            0,
                        ]
                    }
                },
            }
        },
        {
            "$sort": {
                "record_count": -1
            }
        },
        {
            "$project": {
                "_id": 0,
                "document_type": "$_id",
                "record_count": 1,
                "titles_available": 1,
            }
        },
    ]


def build_source_pipeline() -> list:
    """
    Build aggregation grouped by source_name.
    """
    return [
        {
            "$match": {
                "$or": [
                    {"source_name": {"$exists": True}},
                    {"source": {"$exists": True}},
                    {"source_path": {"$exists": True}},
                ]
            }
        },
        {
            "$project": {
                "source_group": {
                    "$ifNull": [
                        "$source_name",
                        {
                            "$ifNull": [
                                "$source",
                                {
                                    "$ifNull": [
                                        "$source_path",
                                        "unknown",
                                    ]
                                },
                            ]
                        },
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": "$source_group",
                "record_count": {"$sum": 1},
            }
        },
        {
            "$sort": {
                "record_count": -1
            }
        },
        {
            "$project": {
                "_id": 0,
                "source_name": "$_id",
                "record_count": 1,
            }
        },
    ]


def run_mongo_aggregation(
    pipeline: list,
    database_name: str = MONGO_DATABASE,
    collection_name: str = MONGO_COLLECTION,
) -> pd.DataFrame:
    """
    Execute a MongoDB aggregation pipeline and return a DataFrame.
    """
    client, collection = get_collection(
        database_name=database_name,
        collection_name=collection_name,
    )

    try:
        results = list(collection.aggregate(pipeline))
        df = pd.DataFrame(results)

        logger.info(
            "Mongo aggregation returned %d rows from %s.%s",
            len(df),
            database_name,
            collection_name,
        )

        return df

    finally:
        client.close()
        logger.info("MongoDB client closed")


def run_category_aggregation(
    min_content_length: int = 0,
) -> pd.DataFrame:
    """
    Run the category aggregation pipeline.
    """
    pipeline = build_category_aggregation_pipeline(
        min_content_length=min_content_length,
    )

    return run_mongo_aggregation(pipeline)


def run_document_type_aggregation() -> pd.DataFrame:
    """
    Run the document type aggregation pipeline.
    """
    pipeline = build_document_type_pipeline()
    return run_mongo_aggregation(pipeline)


def run_source_aggregation() -> pd.DataFrame:
    """
    Run the source aggregation pipeline.
    """
    pipeline = build_source_pipeline()
    return run_mongo_aggregation(pipeline)


def save_mongo_aggregation_outputs(
    category_df: pd.DataFrame,
    document_type_df: pd.DataFrame,
    source_df: pd.DataFrame,
    output_dir: str = "data/processed/analytics/lab10",
) -> dict:
    """
    Save MongoDB aggregation outputs to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_path = output_dir / "mongo_category_aggregation.csv"
    document_type_path = output_dir / "mongo_document_type_aggregation.csv"
    source_path = output_dir / "mongo_source_aggregation.csv"

    category_df.to_csv(category_path, index=False)
    document_type_df.to_csv(document_type_path, index=False)
    source_df.to_csv(source_path, index=False)

    logger.info("Saved Mongo category aggregation to %s", category_path)
    logger.info("Saved Mongo document type aggregation to %s", document_type_path)
    logger.info("Saved Mongo source aggregation to %s", source_path)

    return {
        "mongo_category_aggregation": str(category_path),
        "mongo_document_type_aggregation": str(document_type_path),
        "mongo_source_aggregation": str(source_path),
    }