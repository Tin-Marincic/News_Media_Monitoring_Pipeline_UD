"""
src/embeddings/chroma_store.py

ChromaDB vector store utilities for the News Media Monitoring Pipeline.

This module:
- creates a persistent ChromaDB database in data/embeddings/chroma_db/
- creates/loads a collection named "data"
- stores cleaned news/document records as vector-searchable documents
- stores metadata for filtering by category, document type, year, language, rating
- supports semantic query and multi-query search
"""

import logging
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.embeddings.embedder import build_news_text

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("data/embeddings/chroma_db")
COLLECTION_NAME = "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_client(path: Path = CHROMA_PATH):
    """
    Return a persistent ChromaDB client.

    Data is stored in data/embeddings/chroma_db/ and survives restarts.
    """
    path.mkdir(parents=True, exist_ok=True)

    logger.info("Opening persistent ChromaDB client at %s", path)

    return chromadb.PersistentClient(path=str(path))


def get_embedding_function():
    """
    Return the embedding function used by ChromaDB.

    It must match the model used in embedder.py.
    """
    return SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)


def get_collection(client=None, reset: bool = False):
    """
    Get or create the ChromaDB collection.

    Collection name: data
    Similarity space: cosine
    """
    if client is None:
        client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
            logger.info("Deleted existing ChromaDB collection: %s", COLLECTION_NAME)
        except Exception:
            logger.info("No existing ChromaDB collection to delete")

    embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Loaded ChromaDB collection '%s'", COLLECTION_NAME)

    return collection


def _safe_metadata_value(value, default):
    """
    Convert metadata values into ChromaDB-safe scalar values.

    Chroma metadata values must be str, int, float, bool, or None.
    We avoid None/NaN by replacing with defaults.
    """
    if pd.isna(value):
        return default

    return value


def build_metadata(row: pd.Series) -> dict:
    """
    Build ChromaDB metadata for one news/document record.
    """
    title = str(_safe_metadata_value(row.get("title"), "Untitled"))[:500]
    category = str(_safe_metadata_value(row.get("category"), "unknown"))[:100]
    document_type = str(_safe_metadata_value(row.get("document_type"), "unknown"))[:100]
    source_name = str(_safe_metadata_value(row.get("source_name"), "unknown"))[:200]
    language = str(_safe_metadata_value(row.get("language"), "unknown"))[:20]

    year_value = _safe_metadata_value(row.get("published_year"), 0)
    rating_value = _safe_metadata_value(row.get("rating_score"), 0.0)
    popularity_value = _safe_metadata_value(row.get("popularity"), 0.0)
    content_length_value = _safe_metadata_value(row.get("content_length"), 0)

    try:
        year = int(float(year_value))
    except Exception:
        year = 0

    try:
        rating_score = float(rating_value)
    except Exception:
        rating_score = 0.0

    try:
        popularity = float(popularity_value)
    except Exception:
        popularity = 0.0

    try:
        content_length = int(float(content_length_value))
    except Exception:
        content_length = 0

    return {
        "title": title,
        "category": category,
        "document_type": document_type,
        "source_name": source_name,
        "language": language,
        "year": year,
        "rating_score": rating_score,
        "popularity": popularity,
        "content_length": content_length,
    }


def build_record_id(row: pd.Series, fallback_index: int) -> str:
    """
    Create a stable ChromaDB document ID.
    """
    record_id = row.get("record_id")

    if pd.notna(record_id):
        try:
            return f"record_{int(float(record_id))}"
        except Exception:
            return f"record_{str(record_id)}"

    return f"row_{fallback_index}"


def add_news_to_collection(
    df: pd.DataFrame,
    collection,
    batch_size: int = 100,
) -> int:
    """
    Add cleaned news/document records to the ChromaDB collection.

    Existing IDs are skipped so repeated runs do not duplicate records.
    """
    existing_ids = set(collection.get()["ids"])

    print(f"Collection already has {len(existing_ids)} records")
    logger.info("Collection already has %d records", len(existing_ids))

    documents = []
    metadatas = []
    ids = []

    added_count = 0
    skipped_count = 0

    for fallback_index, (_, row) in enumerate(df.iterrows()):
        doc_id = build_record_id(row, fallback_index)

        if doc_id in existing_ids:
            skipped_count += 1
            continue

        text = build_news_text(row)

        if not text or text == "Unknown news document":
            skipped_count += 1
            continue

        metadata = build_metadata(row)

        documents.append(text)
        metadatas.append(metadata)
        ids.append(doc_id)

        if len(documents) >= batch_size:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            added_count += len(documents)

            documents, metadatas, ids = [], [], []

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        added_count += len(documents)

    total = collection.count()

    print(f"Added {added_count} records")
    print(f"Skipped {skipped_count} records")
    print(f"Collection now contains {total} records")

    logger.info(
        "add_news_to_collection complete: added=%d skipped=%d total=%d",
        added_count,
        skipped_count,
        total,
    )

    return total


def add_movies_to_collection(df, collection, batch_size=100):
    """
    Backward-compatible alias for the professor's function name.

    In this project, it adds news/document records instead of movies.
    """
    return add_news_to_collection(df, collection, batch_size=batch_size)


def query_collection(
    collection,
    query_text: str,
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Query ChromaDB by semantic text similarity.

    Optional where filter examples:
    {"category": {"$eq": "business"}}
    {"year": {"$gte": 2026}}
    {"category": {"$in": ["business", "politics"]}}
    {"$and": [{"category": {"$eq": "business"}}, {"rating_score": {"$gte": 5}}]}
    """
    logger.info(
        "Querying ChromaDB: query='%s' n_results=%d where=%s",
        query_text,
        n_results,
        where,
    )

    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return result


def query_collection_multi(
    collection,
    query_texts: list[str],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Query ChromaDB with multiple queries in one API call.
    """
    logger.info(
        "Running multi-query ChromaDB search: queries=%d n_results=%d where=%s",
        len(query_texts),
        n_results,
        where,
    )

    result = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return result


def chroma_results_to_dataframe(result: dict, query_index: int = 0) -> pd.DataFrame:
    """
    Convert ChromaDB query result into a readable DataFrame.
    """
    ids = result.get("ids", [[]])[query_index]
    documents = result.get("documents", [[]])[query_index]
    metadatas = result.get("metadatas", [[]])[query_index]
    distances = result.get("distances", [[]])[query_index]

    rows = []

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        row = {
            "rank": rank,
            "id": doc_id,
            "distance": distance,
            "similarity_score": 1 - distance if distance is not None else None,
            "document": document,
        }

        if isinstance(metadata, dict):
            row.update(metadata)

        rows.append(row)

    return pd.DataFrame(rows)


def get_collection_count(collection) -> int:
    """
    Return collection size.
    """
    return collection.count()