import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from pymongo import MongoClient

logger = logging.getLogger(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_pipeline"
COLLECTION = "raw_articles"

NA_VALUES = ["", "None", "none", "null", "NULL", "N/A", "n/a", "NaN", "nan"]


def _safe_text(value: Any) -> str:
    """
    Convert nested/unusual values safely to text for CSV export.
    """
    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        return str(value)

    return str(value)


def flatten_mongo_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MongoDB documents where collected content is stored inside 'data'.

    Works for the News Media Monitoring Pipeline:
    - NewsAPI JSON articles
    - parsed PDF/Word/Excel records
    - OCR records
    - scraped records
    - image/audio/video metadata if stored in the same collection
    """
    logger.info("Flattening MongoDB documents for news analytics")

    if df.empty:
        logger.warning("MongoDB DataFrame is empty; nothing to flatten")
        return df

    records = []

    for _, row in df.iterrows():
        record = {}

        data = row.get("data")

        if isinstance(data, dict):
            for key, value in data.items():
                if key == "source":
                    if isinstance(value, dict):
                        record["source_id"] = value.get("id")
                        record["source_name"] = value.get("name")
                    else:
                        record["source_name"] = value
                else:
                    record[key] = value
        else:
            record["content_text"] = _safe_text(data)

        # Preserve top-level Mongo metadata separately.
        record["source_path"] = row.get("source")
        record["fetched_at"] = row.get("fetched_at")
        record["version"] = row.get("version")

        metadata_columns = [
            "document_type",
            "file_name",
            "page_number",
            "extraction_timestamp",
            "extraction_library",
            "type",
        ]

        for col in metadata_columns:
            if col in df.columns and col not in record:
                record[col] = row.get(col)

        records.append(record)

    flat_df = pd.DataFrame(records)

    logger.info("Flattened MongoDB data: shape=%s", flat_df.shape)

    return flat_df


def normalise_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise fields for the News Media Monitoring Pipeline.

    Creates consistent analytics columns:
    - record_id
    - title
    - content_text
    - category
    - document_type
    - published_date
    - published_year
    - language
    - rating_score
    - mentions
    - source_name
    - source_path
    - content_length
    - title_length

    Also creates compatibility aliases for Lab 8 wording:
    - overview
    - genres
    - popularity
    - release_date
    - release_year
    - original_language
    - vote_average
    - vote_count
    """
    logger.info("Normalising news analytics columns")

    df = df.copy()

    if df.empty:
        logger.warning("Cannot normalise news columns because DataFrame is empty")
        return df

    # ------------------------------------------------------------
    # Record ID
    # ------------------------------------------------------------
    if "record_id" not in df.columns:
        df.insert(0, "record_id", range(1, len(df) + 1))

    # ------------------------------------------------------------
    # Document type
    # ------------------------------------------------------------
    if "document_type" not in df.columns:
        if "type" in df.columns:
            df["document_type"] = df["type"]
        else:
            df["document_type"] = "unknown"

    df["document_type"] = df["document_type"].fillna("unknown").astype(str)

    # Improve document_type when it is missing/unknown using available clues.
    if "file_name" in df.columns:
        file_name_text = df["file_name"].fillna("").astype(str)

        json_mask = (
            df["document_type"].str.lower().eq("unknown")
            & file_name_text.str.endswith(".json")
        )
        pdf_mask = (
            df["document_type"].str.lower().eq("unknown")
            & file_name_text.str.endswith(".pdf")
        )
        word_mask = (
            df["document_type"].str.lower().eq("unknown")
            & file_name_text.str.endswith(".docx")
        )
        excel_mask = (
            df["document_type"].str.lower().eq("unknown")
            & file_name_text.str.endswith(".xlsx")
        )

        df.loc[json_mask, "document_type"] = "json"
        df.loc[pdf_mask, "document_type"] = "pdf"
        df.loc[word_mask, "document_type"] = "word"
        df.loc[excel_mask, "document_type"] = "excel"

    if "source_path" in df.columns:
        source_path_text = df["source_path"].fillna("").astype(str)

        api_mask = (
            df["document_type"].str.lower().eq("unknown")
            & source_path_text.str.contains("news_page", case=False, na=False)
        )

        hockey_mask = (
            df["document_type"].str.lower().eq("unknown")
            & source_path_text.str.contains("scrapethissite.com/pages/forms", case=False, na=False)
        )

        movie_mask = (
            df["document_type"].str.lower().eq("unknown")
            & source_path_text.str.contains("oscars|ajax", case=False, na=False)
        )

        df.loc[api_mask, "document_type"] = "json"
        df.loc[hockey_mask, "document_type"] = "scraped_html"
        df.loc[movie_mask, "document_type"] = "scraped_json_api"

    if "url" in df.columns:
        api_article_mask = (
            df["document_type"].str.lower().eq("unknown")
            & df["url"].notna()
        )

        df.loc[api_article_mask, "document_type"] = "news_api"

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    if "title" not in df.columns:
        if "name" in df.columns:
            df["title"] = df["name"]
        elif "file_name" in df.columns:
            df["title"] = df["file_name"]
        else:
            df["title"] = "Untitled record"

    df["title"] = df["title"].fillna("Untitled record").astype(str)

    empty_title_mask = df["title"].str.strip().str.len() == 0

    if "file_name" in df.columns:
        df.loc[empty_title_mask, "title"] = df.loc[empty_title_mask, "file_name"].fillna("Untitled record").astype(str)
    else:
        df.loc[empty_title_mask, "title"] = "Untitled record"

    # ------------------------------------------------------------
    # Source name
    # ------------------------------------------------------------
    if "source_name" not in df.columns:
        if "source" in df.columns:
            df["source_name"] = df["source"]
        elif "source_path" in df.columns:
            df["source_name"] = df["source_path"]
        else:
            df["source_name"] = "unknown"

    df["source_name"] = df["source_name"].fillna("unknown").astype(str)

    # ------------------------------------------------------------
    # Main text/content field
    # ------------------------------------------------------------
    text_candidates = [
        "description",
        "text",
        "processed_text",
        "raw_text",
        "preview_text",
        "content",
        "transcript",
        "content_text",
    ]

    available_text_cols = [col for col in text_candidates if col in df.columns]

    if available_text_cols:
        df["content_text"] = ""

        for col in available_text_cols:
            current_text = df["content_text"].fillna("").astype(str)
            new_text = df[col].fillna("").astype(str)

            df["content_text"] = current_text.where(
                current_text.str.strip().str.len() > 0,
                new_text,
            )
    else:
        df["content_text"] = df["title"]

    df["content_text"] = df["content_text"].fillna("").astype(str)

    empty_content_mask = df["content_text"].str.strip().str.len() == 0
    df.loc[empty_content_mask, "content_text"] = df.loc[empty_content_mask, "title"]

    # ------------------------------------------------------------
    # Category
    # ------------------------------------------------------------
    if "category" not in df.columns:
        df["category"] = df["document_type"]

    df["category"] = df["category"].fillna("unknown").astype(str)

    # If category is unknown, use document_type as useful analytical category.
    unknown_category_mask = df["category"].str.lower().eq("unknown")
    df.loc[unknown_category_mask, "category"] = df.loc[unknown_category_mask, "document_type"]

    # ------------------------------------------------------------
    # Language
    # ------------------------------------------------------------
    if "language" not in df.columns:
        if "original_language" in df.columns:
            df["language"] = df["original_language"]
        else:
            df["language"] = "unknown"

    df["language"] = df["language"].fillna("unknown").astype(str)

    # ------------------------------------------------------------
    # Date
    # ------------------------------------------------------------
    if "published_date" not in df.columns:
        if "publishedAt" in df.columns:
            df["published_date"] = df["publishedAt"]
        elif "extraction_timestamp" in df.columns:
            df["published_date"] = df["extraction_timestamp"]
        elif "fetched_at" in df.columns:
            df["published_date"] = df["fetched_at"]
        else:
            df["published_date"] = pd.NaT

    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["published_year"] = df["published_date"].dt.year

    # ------------------------------------------------------------
    # Mentions / popularity-like numeric feature
    # ------------------------------------------------------------
    if "mentions" in df.columns:
        df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce")
    else:
        df["mentions"] = pd.NA

    # ------------------------------------------------------------
    # Text-length features
    # ------------------------------------------------------------
    df["content_length"] = df["content_text"].fillna("").astype(str).str.len()
    df["title_length"] = df["title"].fillna("").astype(str).str.len()

    # ------------------------------------------------------------
    # Rating score for Lab 8 chunked mean requirement
    # ------------------------------------------------------------
    if "rating_score" not in df.columns:
        df["rating_score"] = pd.NA

    # 1. Use sentiment_score if available, scaled from 0-1 to 0-10.
    if "sentiment_score" in df.columns:
        sentiment = pd.to_numeric(df["sentiment_score"], errors="coerce")
        df.loc[sentiment.notna(), "rating_score"] = sentiment[sentiment.notna()] * 10

    # 2. Use mentions if available, min-max scaled to 0-10.
    if df["rating_score"].isna().any() and "mentions" in df.columns:
        mentions = pd.to_numeric(df["mentions"], errors="coerce")
        valid_mentions = mentions.dropna()

        if not valid_mentions.empty and valid_mentions.max() != valid_mentions.min():
            mentions_scaled = (
                (mentions - valid_mentions.min())
                / (valid_mentions.max() - valid_mentions.min())
                * 10
            )

            fill_mask = df["rating_score"].isna() & mentions_scaled.notna()
            df.loc[fill_mask, "rating_score"] = mentions_scaled[fill_mask]

    # 3. Fallback: content-length score, clipped to 0-10.
    if df["rating_score"].isna().any():
        content_score = (df["content_length"] / 1000 * 10).clip(0, 10)
        fill_mask = df["rating_score"].isna()
        df.loc[fill_mask, "rating_score"] = content_score[fill_mask]

    df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")

    # ------------------------------------------------------------
    # URL
    # ------------------------------------------------------------
    if "url" not in df.columns:
        df["url"] = pd.NA

    # ------------------------------------------------------------
    # News-friendly aliases for the old lab wording.
    # These help reuse Lab 8 functions while still analyzing news data.
    # ------------------------------------------------------------
    df["overview"] = df["content_text"]
    df["genres"] = df["category"]
    df["popularity"] = df["mentions"].fillna(df["content_length"])
    df["release_date"] = df["published_date"]
    df["release_year"] = df["published_year"]
    df["original_language"] = df["language"]
    df["vote_average"] = df["rating_score"]
    df["vote_count"] = df["mentions"].fillna(1)

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------
    numeric_columns = [
        "record_id",
        "mentions",
        "content_length",
        "title_length",
        "rating_score",
        "published_year",
        "popularity",
        "release_year",
        "vote_average",
        "vote_count",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "News columns normalised: shape=%s columns=%s",
        df.shape,
        list(df.columns),
    )

    return df


# Backward-compatible alias in case another file still imports this name.
def normalise_movie_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.warning("normalise_movie_columns() called; redirecting to normalise_news_columns()")
    return normalise_news_columns(df)


def load_from_mongodb(
    uri: str = MONGO_URI,
    db: str = DB_NAME,
    collection: str = COLLECTION,
    document_type: Optional[str] = None,
    limit: int = 0,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Load integrated News Media Monitoring Pipeline data from MongoDB.

    By default, loads all records from news_pipeline.raw_articles.
    Pass document_type only if you want to filter a specific source type.
    """
    logger.info("Connecting to MongoDB: db=%s collection=%s", db, collection)

    client = MongoClient(uri)

    try:
        coll = client[db][collection]

        query = {}

        if document_type:
            query["document_type"] = document_type

        cursor = coll.find(query, {"_id": 0})

        if limit:
            cursor = cursor.limit(limit)

        raw_df = pd.DataFrame(list(cursor))

        logger.info(
            "Loaded %d raw rows from MongoDB using query=%s",
            len(raw_df),
            query,
        )

        flat_df = flatten_mongo_documents(raw_df)

        if normalise:
            flat_df = normalise_news_columns(flat_df)

        logger.info("MongoDB load complete: shape=%s", flat_df.shape)

        return flat_df

    finally:
        client.close()
        logger.info("MongoDB connection closed")


def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Export a DataFrame to CSV and log the action.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, encoding="utf-8")

    logger.info("Saved DataFrame to CSV: rows=%d path=%s", len(df), path)


def load_from_csv(
    path: str,
    dtype: Optional[dict] = None,
    parse_dates: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load CSV safely.
    """
    logger.info("Loading CSV: %s", path)

    df = pd.read_csv(
        path,
        dtype=dtype,
        parse_dates=parse_dates,
        na_values=NA_VALUES,
        encoding="utf-8",
    )

    date_columns = [
        "published_date",
        "release_date",
        "fetched_at",
        "extraction_timestamp",
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("CSV loaded: shape=%s", df.shape)

    return df


def chunked_stats(
    path: str,
    chunk_size: int = 200,
    rating_column: str = "rating_score",
    language_column: str = "language",
) -> dict:
    """
    Load a large CSV in chunks and compute:
    1. Global mean of the rating_score column
    2. Per-language mean rating_score
    3. Per-language record count
    """
    logger.info(
        "Starting chunked stats: path=%s chunk_size=%d rating_column=%s language_column=%s",
        path,
        chunk_size,
        rating_column,
        language_column,
    )

    total_rows = 0
    rating_sum = 0.0
    rating_count = 0
    language_accumulator = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunk_size, na_values=NA_VALUES),
        start=1,
    ):
        logger.info("Processing chunk %d with %d rows", chunk_number, len(chunk))

        total_rows += len(chunk)

        if rating_column not in chunk.columns:
            raise ValueError(
                f"Missing rating column '{rating_column}'. "
                f"Available columns: {list(chunk.columns)}"
            )

        chunk[rating_column] = pd.to_numeric(chunk[rating_column], errors="coerce")

        valid_ratings = chunk.dropna(subset=[rating_column])

        rating_sum += valid_ratings[rating_column].sum()
        rating_count += valid_ratings[rating_column].count()

        if language_column not in chunk.columns:
            chunk[language_column] = "unknown"

        clean_language_chunk = chunk.dropna(subset=[language_column, rating_column])

        grouped = clean_language_chunk.groupby(language_column)[rating_column].agg(["sum", "count"])

        for language, row in grouped.iterrows():
            if language not in language_accumulator:
                language_accumulator[language] = {"sum": 0.0, "count": 0}

            language_accumulator[language]["sum"] += row["sum"]
            language_accumulator[language]["count"] += row["count"]

    global_mean = float(rating_sum / rating_count) if rating_count else 0.0

    language_rows = []

    for language, values in language_accumulator.items():
        if values["count"] > 0:
            language_rows.append(
                {
                    "language": language,
                    "mean_rating_score": values["sum"] / values["count"],
                    "record_count": values["count"],
                }
            )

    language_df = pd.DataFrame(language_rows)

    if not language_df.empty:
        language_df = language_df.sort_values(
            "record_count",
            ascending=False,
        ).reset_index(drop=True)

    logger.info(
        "Chunked stats complete: total_rows=%d rating_count=%d global_mean=%.4f",
        total_rows,
        rating_count,
        global_mean,
    )

    return {
        "global_mean": global_mean,
        "total_rows": total_rows,
        "rating_count": int(rating_count),
        "language_df": language_df,
    }


def process_chunks_per_language(
    path: str,
    chunk_size: int = 200,
    rating_column: str = "rating_score",
    language_column: str = "language",
) -> pd.DataFrame:
    """
    Process chunks per language and combine accumulators.
    """
    logger.info("Processing chunks per language")

    stats = chunked_stats(
        path=path,
        chunk_size=chunk_size,
        rating_column=rating_column,
        language_column=language_column,
    )

    language_df = stats["language_df"]

    logger.info("Per-language chunk processing complete: rows=%d", len(language_df))

    return language_df


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise DataFrame memory usage by:
    - downcasting integer columns
    - downcasting float columns
    - converting low-cardinality text columns to category
    """
    logger.info("Starting dtype optimisation")

    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["int", "int64", "int32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer", errors="coerce")
        logger.debug("Downcast integer column: %s -> %s", col, df_opt[col].dtype)

    for col in df_opt.select_dtypes(include=["float", "float64", "float32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float", errors="coerce")
        logger.debug("Downcast float column: %s -> %s", col, df_opt[col].dtype)

    object_columns = df_opt.select_dtypes(include=["object"]).columns

    long_text_columns = [
        "content_text",
        "overview",
        "description",
        "text",
        "processed_text",
        "raw_text",
        "preview_text",
        "url",
        "tables",
    ]

    for col in object_columns:
        if len(df_opt) == 0:
            continue

        if col in long_text_columns:
            continue

        cardinality_ratio = df_opt[col].nunique(dropna=True) / len(df_opt)

        if cardinality_ratio < 0.50:
            df_opt[col] = df_opt[col].astype("category")
            logger.debug(
                "Converted object column to category: %s cardinality=%.2f%%",
                col,
                cardinality_ratio * 100,
            )

    logger.info("Dtype optimisation complete")

    return df_opt


def memory_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Compare memory usage before and after dtype optimisation.
    This result is logged to pipeline.log.
    """
    before_mb = df_before.memory_usage(deep=True).sum() / 1024**2
    after_mb = df_after.memory_usage(deep=True).sum() / 1024**2

    reduction_pct = (1 - after_mb / before_mb) * 100 if before_mb else 0

    logger.info("Memory usage before optimisation: %.4f MB", before_mb)
    logger.info("Memory usage after optimisation: %.4f MB", after_mb)
    logger.info("Memory reduction after dtype optimisation: %.2f%%", reduction_pct)

    return {
        "before_mb": before_mb,
        "after_mb": after_mb,
        "reduction_pct": reduction_pct,
    }