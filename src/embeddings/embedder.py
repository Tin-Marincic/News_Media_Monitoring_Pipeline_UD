from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """
    Return the sentence-transformer model, loading it only on first call.
    """
    global _model

    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        print(f"Loading model: {MODEL_NAME}")

        _model = SentenceTransformer(MODEL_NAME)

        logger.info("Embedding model loaded successfully")
        print("Model loaded successfully")

    return _model


def _safe_text(value) -> str:
    """
    Convert missing values safely to text.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def build_news_text(row):
    """
    Create a single searchable text string from a news/document row.

    This is the News Media Monitoring equivalent of combining:
    movie title + overview + genres.

    We combine:
    - title
    - content_text / overview / description
    - category / genres
    - document_type
    - source_name
    - language
    """
    parts = []

    title = _safe_text(row.get("title", ""))
    content_text = _safe_text(row.get("content_text", ""))
    overview = _safe_text(row.get("overview", ""))
    description = _safe_text(row.get("description", ""))
    category = _safe_text(row.get("category", ""))
    genres = _safe_text(row.get("genres", ""))
    document_type = _safe_text(row.get("document_type", ""))
    source_name = _safe_text(row.get("source_name", ""))
    language = _safe_text(row.get("language", ""))

    if title:
        parts.append(f"Title: {title}")

    if content_text:
        parts.append(f"Content: {content_text}")
    elif overview:
        parts.append(f"Content: {overview}")
    elif description:
        parts.append(f"Description: {description}")

    if category:
        parts.append(f"Category: {category}")

    if genres and genres != category:
        parts.append(f"Genres: {genres}")

    if document_type:
        parts.append(f"Document type: {document_type}")

    if source_name:
        parts.append(f"Source: {source_name}")

    if language:
        parts.append(f"Language: {language}")

    return " | ".join(parts) if parts else "Unknown news document"


def build_document_text(row):
    """
    Generic alias used by the rest of the Lab 11 code.
    """
    return build_news_text(row)


def build_movie_text(row):
    """
    Backward-compatible alias for the professor's movie-based function name.

    In this project, it builds news/document text instead of movie text.
    """
    return build_news_text(row)


def build_texts_from_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Build searchable text strings for every row in a DataFrame.
    """
    texts = df.apply(build_news_text, axis=1).tolist()

    logger.info("Built %d searchable document texts", len(texts))

    return texts


def embed_texts(texts, batch_size=64, normalize=True):
    """
    Generate embeddings for a list of text strings.

    Returns:
        numpy array of shape (len(texts), 384) when using all-MiniLM-L6-v2
    """
    texts = list(texts)

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
    )

    logger.info("Generated embeddings with shape %s", embeddings.shape)

    return embeddings


def embed_single(text, normalize=True):
    """
    Generate an embedding for a single text string.
    Useful for embedding a search query.
    """
    model = get_model()

    embedding = model.encode(
        text,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embedding


def embed_dataframe(
    df: pd.DataFrame,
    batch_size=64,
    normalize=True,
) -> tuple[list[str], np.ndarray]:
    """
    Build searchable text from a DataFrame and generate embeddings.
    """
    texts = build_texts_from_dataframe(df)

    embeddings = embed_texts(
        texts,
        batch_size=batch_size,
        normalize=normalize,
    )

    return texts, embeddings


def cosine_similarity(vec_a, vec_b) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    vec_a = np.asarray(vec_a)
    vec_b = np.asarray(vec_b)

    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / denominator)


def dot_product(vec_a, vec_b) -> float:
    """
    Compute dot product between two vectors.
    """
    return float(np.dot(vec_a, vec_b))


def euclidean_distance(vec_a, vec_b) -> float:
    """
    Compute Euclidean distance between two vectors.
    Lower distance means more similar.
    """
    return float(np.linalg.norm(np.asarray(vec_a) - np.asarray(vec_b)))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    """
    embeddings = np.asarray(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1

    normalized = embeddings / norms

    return np.matmul(normalized, normalized.T)


def rank_texts_by_similarity(
    query: str,
    texts: list[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Rank texts by cosine similarity to a natural-language query.
    """
    query_embedding = embed_single(query, normalize=True)

    scores = np.dot(embeddings, query_embedding)

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    rows = []

    for rank, idx in enumerate(ranked_indices, start=1):
        rows.append({
            "rank": rank,
            "text_index": int(idx),
            "similarity_score": float(scores[idx]),
            "text": texts[idx],
        })

    return pd.DataFrame(rows)


def compare_similarity_measures(text_a: str, text_b: str, text_c: str) -> pd.DataFrame:
    """
    Compare cosine similarity, dot product, and Euclidean distance.

    Usually:
    - text_a and text_b should be related
    - text_a and text_c should be less related
    """
    texts = [text_a, text_b, text_c]
    embeddings = embed_texts(texts, batch_size=3, normalize=True)

    pairs = [
        ("A vs B", embeddings[0], embeddings[1]),
        ("A vs C", embeddings[0], embeddings[2]),
    ]

    rows = []

    for pair_name, vec_1, vec_2 in pairs:
        rows.append({
            "pair": pair_name,
            "cosine_similarity": cosine_similarity(vec_1, vec_2),
            "dot_product": dot_product(vec_1, vec_2),
            "euclidean_distance": euclidean_distance(vec_1, vec_2),
        })

    return pd.DataFrame(rows)