import os
import sys
from pathlib import Path

import pandas as pd
from pymongo import MongoClient, ASCENDING, TEXT


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CSV_PATH = Path(os.getenv("DASHBOARD_CSV_PATH", "data/processed/cleaned/cleaned_data.csv"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", os.getenv("MONGO_DB", "news_dashboard"))
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "news_records")


def clean_value(value):
    if isinstance(value, (list, dict)):
        return value

    try:
        if pd.isna(value):
            return None
    except ValueError:
        return value

    if hasattr(value, "item"):
        return value.item()

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    clean_df = df.copy()

    for col in clean_df.columns:
        clean_df[col] = clean_df[col].map(clean_value)

    return clean_df.to_dict(orient="records")


def seed_mongodb() -> dict:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    records = dataframe_to_records(df)

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    try:
        client.admin.command("ping")

        db = client[DB_NAME]
        db.drop_collection(COLLECTION_NAME)

        collection = db[COLLECTION_NAME]

        if records:
            collection.insert_many(records)

        collection.create_index([("record_id", ASCENDING)])
        collection.create_index([("category", ASCENDING)])
        collection.create_index([("document_type", ASCENDING)])
        collection.create_index([("published_year", ASCENDING)])
        collection.create_index([("rating_score", ASCENDING)])
        collection.create_index([("popularity", ASCENDING)])
        collection.create_index([("title", ASCENDING)])

        collection.create_index(
            [
                ("title", TEXT),
                ("content_text", TEXT),
                ("overview", TEXT),
                ("category", TEXT),
                ("document_type", TEXT),
            ],
            name="news_text_search_index",
            default_language="none",
            language_override="text_search_language",
        )

        result = {
            "csv_path": str(CSV_PATH),
            "mongo_uri": MONGO_URI,
            "db_name": DB_NAME,
            "collection_name": COLLECTION_NAME,
            "inserted_records": collection.count_documents({}),
            "columns": len(df.columns),
        }

        return result

    finally:
        client.close()


if __name__ == "__main__":
    result = seed_mongodb()

    print("MongoDB seeding complete")
    print("CSV path:", result["csv_path"])
    print("Database:", result["db_name"])
    print("Collection:", result["collection_name"])
    print("Inserted records:", result["inserted_records"])
    print("Columns:", result["columns"])