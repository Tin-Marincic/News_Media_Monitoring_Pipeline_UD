"""
src/embeddings/hybrid_search.py

Hybrid search for the News Media Monitoring Pipeline.

Combines keyword search and semantic search using Reciprocal Rank Fusion (RRF).

RRF formula:
    score = 1 / (k + rank)

where k is usually 60.
"""

import logging
import pandas as pd

from src.embeddings.search_engine import keyword_search, semantic_search

logger = logging.getLogger(__name__)


def _get_result_key(row: pd.Series) -> str:
    """
    Create a stable key for merging ranked results.

    Priority:
    1. record_id
    2. id
    3. title
    """
    if "record_id" in row and pd.notna(row["record_id"]):
        return f"record_{row['record_id']}"

    if "id" in row and pd.notna(row["id"]):
        return str(row["id"])

    if "title" in row and pd.notna(row["title"]):
        return str(row["title"])

    return str(row.name)


def reciprocal_rank_fusion(
    keyword_results: pd.DataFrame,
    semantic_results: pd.DataFrame,
    k: int = 60,
) -> pd.DataFrame:
    """
    Combine keyword and semantic ranked lists using Reciprocal Rank Fusion.

    Args:
        keyword_results: DataFrame returned by keyword_search()
        semantic_results: DataFrame returned by semantic_search()
        k: RRF constant, default 60

    Returns:
        DataFrame sorted by combined RRF score.
    """
    scores = {}
    metadata = {}

    if keyword_results is not None and not keyword_results.empty:
        for rank, (_, row) in enumerate(keyword_results.iterrows(), start=1):
            key = _get_result_key(row)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            row_dict = row.to_dict()
            row_dict["keyword_rank"] = rank

            metadata[key] = row_dict

    if semantic_results is not None and not semantic_results.empty:
        for rank, (_, row) in enumerate(semantic_results.iterrows(), start=1):
            key = _get_result_key(row)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            row_dict = row.to_dict()
            row_dict["semantic_rank"] = rank

            if key in metadata:
                metadata[key].update({
                    "semantic_rank": rank,
                    "semantic_similarity_score": row.get("similarity_score"),
                    "semantic_document": row.get("document"),
                })

                if "search_type" in metadata[key]:
                    metadata[key]["search_type"] = "hybrid"
            else:
                metadata[key] = row_dict

    rows = []

    for key, score in sorted(scores.items(), key=lambda item: -item[1]):
        row = metadata[key].copy()

        row["result_key"] = key
        row["rrf_score"] = round(score, 6)
        row["search_type"] = "hybrid"

        rows.append(row)

    result = pd.DataFrame(rows)

    logger.info(
        "reciprocal_rank_fusion complete: keyword=%d semantic=%d combined=%d",
        0 if keyword_results is None else len(keyword_results),
        0 if semantic_results is None else len(semantic_results),
        len(result),
    )

    return result


def hybrid_search(
    query: str,
    df: pd.DataFrame,
    collection,
    n_results: int = 10,
    k: int = 60,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    Run keyword and semantic search, then combine results with RRF.

    Args:
        query: search string
        df: cleaned news DataFrame for keyword search
        collection: ChromaDB collection for semantic search
        n_results: final number of hybrid results
        k: RRF constant
        filters: optional ChromaDB metadata filter for semantic search

    Returns:
        Hybrid-ranked DataFrame.
    """
    n_candidates = max(n_results * 3, n_results)

    keyword_results = keyword_search(
        query=query,
        df=df,
        n_results=n_candidates,
    )

    semantic_results = semantic_search(
        query=query,
        n_results=n_candidates,
        filters=filters,
        collection=collection,
    )

    combined = reciprocal_rank_fusion(
        keyword_results=keyword_results,
        semantic_results=semantic_results,
        k=k,
    )

    if combined.empty:
        return combined

    preferred_columns = [
        "title",
        "category",
        "document_type",
        "published_year",
        "year",
        "language",
        "rating_score",
        "keyword_score",
        "phrase_match",
        "similarity_score",
        "semantic_similarity_score",
        "keyword_rank",
        "semantic_rank",
        "rrf_score",
        "search_type",
        "record_id",
        "id",
        "result_key",
    ]

    existing_columns = [col for col in preferred_columns if col in combined.columns]
    remaining_columns = [col for col in combined.columns if col not in existing_columns]

    combined = combined[existing_columns + remaining_columns]

    return combined.head(n_results).reset_index(drop=True)


def compare_all_search_methods(
    query: str,
    df: pd.DataFrame,
    collection,
    n_results: int = 5,
    filters: dict | None = None,
) -> dict:
    """
    Run keyword, semantic, and hybrid search for the same query.

    Returns:
        {
            "keyword": DataFrame,
            "semantic": DataFrame,
            "hybrid": DataFrame
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

    hybrid_results = hybrid_search(
        query=query,
        df=df,
        collection=collection,
        n_results=n_results,
        k=60,
        filters=filters,
    )

    print(f"--- Query: '{query}' ---")
    print()

    print("Keyword results:")
    if keyword_results.empty:
        print("  No keyword results found.")
    else:
        for _, row in keyword_results.iterrows():
            print(f"  {row.get('title', 'Untitled')} - {row.get('category', 'unknown')}")

    print()
    print("Semantic results:")
    if semantic_results.empty:
        print("  No semantic results found.")
    else:
        for _, row in semantic_results.iterrows():
            print(
                f"  [{row.get('similarity_score', 0):.3f}] "
                f"{row.get('title', 'Untitled')} - {row.get('category', 'unknown')}"
            )

    print()
    print("Hybrid results:")
    if hybrid_results.empty:
        print("  No hybrid results found.")
    else:
        for _, row in hybrid_results.iterrows():
            print(
                f"  [{row.get('rrf_score', 0):.6f}] "
                f"{row.get('title', 'Untitled')} - {row.get('category', 'unknown')}"
            )

    return {
        "keyword": keyword_results,
        "semantic": semantic_results,
        "hybrid": hybrid_results,
    }