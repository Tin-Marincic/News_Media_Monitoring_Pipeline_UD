import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


DEFAULT_CSV_PATH = Path("data/processed/cleaned/cleaned_data.csv")
DEFAULT_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME = os.getenv("MONGO_DB_NAME", "news_dashboard")
DEFAULT_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "news_records")


def get_mongo_client(uri: str = DEFAULT_MONGO_URI, timeout_ms: int = 3000) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def normalize_dashboard_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    required_text_columns = {
        "title": "Untitled",
        "category": "unknown",
        "document_type": "unknown",
        "language": "unknown",
        "source_name": "unknown",
        "content_text": "",
        "overview": "",
    }

    for col, default in required_text_columns.items():
        if col not in data.columns:
            data[col] = default

        data[col] = (
            data[col]
            .fillna(default)
            .astype(str)
            .str.strip()
            .replace("", default)
        )

    required_numeric_columns = [
        "record_id",
        "published_year",
        "rating_score",
        "popularity",
        "content_length",
        "title_length",
        "vote_average",
        "vote_count",
    ]

    for col in required_numeric_columns:
        if col not in data.columns:
            data[col] = pd.NA

        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "record_id" not in data.columns or data["record_id"].isna().all():
        data["record_id"] = range(1, len(data) + 1)

    data["record_id"] = data["record_id"].fillna(pd.Series(range(1, len(data) + 1), index=data.index))
    data["published_year"] = data["published_year"].fillna(0)
    data["rating_score"] = data["rating_score"].fillna(0.0)
    data["popularity"] = data["popularity"].fillna(0.0)
    data["content_length"] = data["content_length"].fillna(0.0)
    data["title_length"] = data["title_length"].fillna(data["title"].str.len())

    data["published_year"] = data["published_year"].astype(int)
    data["record_id"] = data["record_id"].astype(int)

    return data


def load_from_csv(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    return normalize_dashboard_dataframe(df)


def load_from_mongodb(
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> pd.DataFrame:
    client = get_mongo_client(uri)

    try:
        client.admin.command("ping")
        collection = client[db_name][collection_name]
        records = list(collection.find({}, {"_id": 0}))

        if not records:
            raise ValueError("MongoDB collection is empty")

        df = pd.DataFrame(records)
        return normalize_dashboard_dataframe(df)

    finally:
        client.close()


def load_dashboard_data(
    prefer_mongo: bool = True,
    csv_path: Path | str = DEFAULT_CSV_PATH,
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> pd.DataFrame:
    if prefer_mongo:
        try:
            return load_from_mongodb(
                uri=uri,
                db_name=db_name,
                collection_name=collection_name,
            )
        except (PyMongoError, ServerSelectionTimeoutError, ValueError, OSError):
            return load_from_csv(csv_path)

    return load_from_csv(csv_path)


def get_available_categories(df: pd.DataFrame | None = None) -> list[str]:
    data = load_dashboard_data() if df is None else df

    categories = (
        data["category"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .unique()
        .tolist()
    )

    return sorted(categories)


def get_available_document_types(df: pd.DataFrame | None = None) -> list[str]:
    data = load_dashboard_data() if df is None else df

    document_types = (
        data["document_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .unique()
        .tolist()
    )

    return sorted(document_types)


def get_year_range(df: pd.DataFrame | None = None) -> tuple[int, int]:
    data = load_dashboard_data() if df is None else df

    years = data["published_year"]
    years = years[(years.notna()) & (years > 0)]

    if years.empty:
        return 2026, 2026

    return int(years.min()), int(years.max())


def filter_news_data(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    document_types: list[str] | None = None,
    year_range: list[int] | tuple[int, int] | None = None,
    search_text: str | None = None,
) -> pd.DataFrame:
    data = normalize_dashboard_dataframe(df)

    if categories:
        data = data[data["category"].isin(categories)]

    if document_types:
        data = data[data["document_type"].isin(document_types)]

    if year_range and len(year_range) == 2:
        start_year, end_year = int(year_range[0]), int(year_range[1])
        valid_year_mask = data["published_year"].between(start_year, end_year)
        unknown_year_mask = data["published_year"] == 0
        data = data[valid_year_mask | unknown_year_mask]

    if search_text:
        query = str(search_text).strip().lower()

        if query:
            searchable = (
                data["title"].fillna("").astype(str)
                + " "
                + data["content_text"].fillna("").astype(str)
                + " "
                + data["overview"].fillna("").astype(str)
                + " "
                + data["category"].fillna("").astype(str)
                + " "
                + data["document_type"].fillna("").astype(str)
            ).str.lower()

            data = data[searchable.str.contains(query, na=False, regex=False)]

    return data.reset_index(drop=True)


def get_summary_metrics(df: pd.DataFrame) -> dict:
    data = normalize_dashboard_dataframe(df)

    total_records = int(len(data))
    category_count = int(data["category"].nunique())
    document_type_count = int(data["document_type"].nunique())
    avg_rating = float(data["rating_score"].mean()) if total_records else 0.0
    avg_popularity = float(data["popularity"].mean()) if total_records else 0.0
    avg_content_length = float(data["content_length"].mean()) if total_records else 0.0

    return {
        "total_records": total_records,
        "category_count": category_count,
        "document_type_count": document_type_count,
        "avg_rating": round(avg_rating, 2),
        "avg_popularity": round(avg_popularity, 2),
        "avg_content_length": round(avg_content_length, 2),
    }


def get_top_records(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    data = normalize_dashboard_dataframe(df)

    return (
        data.sort_values(["popularity", "rating_score"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_collection_config() -> dict:
    return {
        "mongo_uri": DEFAULT_MONGO_URI,
        "db_name": DEFAULT_DB_NAME,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "csv_path": str(DEFAULT_CSV_PATH),
    }