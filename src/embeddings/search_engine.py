"""
src/embeddings/search_engine.py

High-level search functions for the News Media Monitoring Pipeline.

This module provides:
- semantic search using ChromaDB
- keyword search using exact text matching
- side-by-side comparison of keyword vs semantic search
"""

import logging
import re

import pandas as pd

from src.embeddings.chroma_store import (
    get_chroma_client,
    get_collection,
    query_collection,
    chroma_results_to_dataframe,
)

logger = logging.getLogger(__name__)


def semantic_search(
    query: str,
    n_results: int = 10,
    filters: dict | None = None,
    collection=None,
) -> pd.DataFrame:
    """
    Search news/documents by meaning using ChromaDB.

    Args:
        query: natural language search string
        n_results: number of results to return
        filters: optional ChromaDB metadata filter
        collection: existing ChromaDB collection, created if None

    Returns:
        DataFrame with semantic search results.
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_collection(client)

    collection_count = collection.count()

    if collection_count == 0:
        logger.warning("semantic_search called on an empty ChromaDB collection")
        return pd.DataFrame()

    n_results = min(n_results, collection_count)

    result = query_collection(
        collection=collection,
        query_text=query,
        n_results=n_results,
        where=filters,
    )

    df = chroma_results_to_dataframe(result)

    if df.empty:
        return df

    df["search_type"] = "semantic"

    columns = [
        "rank",
        "title",
        "category",
        "document_type",
        "year",
        "language",
        "rating_score",
        "similarity_score",
        "document",
        "search_type",
        "id",
    ]

    existing_columns = [col for col in columns if col in df.columns]

    return df[existing_columns]


def keyword_search(
    query: str,
    df: pd.DataFrame,
    text_cols: list | None = None,
    n_results: int = 10,
) -> pd.DataFrame:
    """
    Simple keyword search using exact word/string matching.

    Searches across title, content_text, overview, description, category,
    document_type, and source_name if available.
    """
    if df.empty:
        return pd.DataFrame()

    if text_cols is None:
        possible_cols = [
            "title",
            "content_text",
            "overview",
            "description",
            "text",
            "processed_text",
            "raw_text",
            "category",
            "document_type",
            "source_name",
        ]

        text_cols = [col for col in possible_cols if col in df.columns]

    if not text_cols:
        logger.warning("keyword_search skipped because no searchable text columns were found")
        return pd.DataFrame()

    query_clean = str(query).strip().lower()

    if not query_clean:
        return pd.DataFrame()

    query_terms = [
        term for term in re.split(r"\s+", query_clean)
        if term
    ]

    combined_text = pd.Series("", index=df.index, dtype="object")

    for col in text_cols:
        combined_text = combined_text + " " + df[col].fillna("").astype(str).str.lower()

    phrase_mask = combined_text.str.contains(
        re.escape(query_clean),
        case=False,
        na=False,
        regex=True,
    )

    term_score = pd.Series(0, index=df.index, dtype="int64")

    for term in query_terms:
        term_score += combined_text.str.contains(
            re.escape(term),
            case=False,
            na=False,
            regex=True,
        ).astype(int)

    mask = phrase_mask | (term_score > 0)

    results = df.loc[mask].copy()

    if results.empty:
        return pd.DataFrame()

    results["keyword_score"] = term_score.loc[results.index]
    results["phrase_match"] = phrase_mask.loc[results.index]
    results["search_type"] = "keyword"

    sort_columns = ["phrase_match", "keyword_score"]

    if "rating_score" in results.columns:
        sort_columns.append("rating_score")

    results = results.sort_values(
        sort_columns,
        ascending=[False, False] + [False] * (len(sort_columns) - 2),
    )

    output_columns = [
        "record_id",
        "title",
        "category",
        "document_type",
        "published_year",
        "language",
        "rating_score",
        "keyword_score",
        "phrase_match",
        "search_type",
    ]

    existing_columns = [col for col in output_columns if col in results.columns]

    return results[existing_columns].head(n_results).reset_index(drop=True)


def compare_search(
    query: str,
    df: pd.DataFrame,
    collection=None,
    n_results: int = 5,
    filters: dict | None = None,
) -> dict:
    """
    Run keyword and semantic search on the same query.

    Returns:
        {
            "keyword": keyword_results_df,
            "semantic": semantic_results_df
        }
    """
    keyword_results = keyword_search(
        query=query,
        df=df,
        n_results=n_results,
    )

    semantic_results = semantic_search(
        query=query,
        n_results=n_results,
        filters=filters,
        collection=collection,
    )

    print(f"--- Query: '{query}' ---")
    print()

    print(f"Keyword search found {len(keyword_results)} results:")
    if keyword_results.empty:
        print("  No keyword results found.")
    else:
        for _, row in keyword_results.iterrows():
            title = row.get("title", "Untitled")
            category = row.get("category", "unknown")
            year = row.get("published_year", "unknown")
            print(f"  {title} ({year}) - {category}")

    print()

    print(f"Semantic search found {len(semantic_results)} results:")
    if semantic_results.empty:
        print("  No semantic results found.")
    else:
        for _, row in semantic_results.iterrows():
            title = row.get("title", "Untitled")
            category = row.get("category", "unknown")
            year = row.get("year", "unknown")
            similarity = row.get("similarity_score", 0)
            print(f"  [{similarity:.3f}] {title} ({year}) - {category}")

    logger.info(
        "compare_search complete for query='%s': keyword=%d semantic=%d",
        query,
        len(keyword_results),
        len(semantic_results),
    )

    return {
        "keyword": keyword_results,
        "semantic": semantic_results,
    }


def result_overlap(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    id_col_a: str = "record_id",
    id_col_b: str = "id",
) -> dict:
    """
    Calculate overlap between two ranked result sets.

    Useful for comparing synonym queries or keyword vs semantic search.
    """
    if results_a.empty or results_b.empty:
        return {
            "overlap_count": 0,
            "overlap_ratio": 0.0,
            "set_a_size": len(results_a),
            "set_b_size": len(results_b),
        }

    if id_col_a in results_a.columns:
        set_a = set(results_a[id_col_a].astype(str))
    elif "title" in results_a.columns:
        set_a = set(results_a["title"].astype(str))
    else:
        set_a = set(results_a.index.astype(str))

    if id_col_b in results_b.columns:
        set_b = set(results_b[id_col_b].astype(str))
    elif "title" in results_b.columns:
        set_b = set(results_b["title"].astype(str))
    else:
        set_b = set(results_b.index.astype(str))

    overlap = set_a.intersection(set_b)
    denominator = min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 1

    return {
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / denominator,
        "set_a_size": len(set_a),
        "set_b_size": len(set_b),
    }


def compare_synonym_queries(
    query_a: str,
    query_b: str,
    df: pd.DataFrame,
    collection=None,
    n_results: int = 10,
) -> pd.DataFrame:
    """
    Compare how consistent keyword and semantic search are for two similar queries.

    Example:
        "artificial intelligence market"
        "AI technology business"
    """
    keyword_a = keyword_search(query_a, df, n_results=n_results)
    keyword_b = keyword_search(query_b, df, n_results=n_results)

    semantic_a = semantic_search(query_a, n_results=n_results, collection=collection)
    semantic_b = semantic_search(query_b, n_results=n_results, collection=collection)

    keyword_overlap = result_overlap(
        keyword_a,
        keyword_b,
        id_col_a="record_id",
        id_col_b="record_id",
    )

    semantic_overlap = result_overlap(
        semantic_a,
        semantic_b,
        id_col_a="id",
        id_col_b="id",
    )

    rows = [
        {
            "method": "keyword",
            "query_a": query_a,
            "query_b": query_b,
            **keyword_overlap,
        },
        {
            "method": "semantic",
            "query_a": query_a,
            "query_b": query_b,
            **semantic_overlap,
        },
    ]

    return pd.DataFrame(rows)