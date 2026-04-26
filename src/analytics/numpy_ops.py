import numpy as np
import logging

logger = logging.getLogger(__name__)


def demonstrate_array_creation() -> dict:
    """
    Demonstrate 5 NumPy array creation methods for the
    News Media Monitoring Pipeline.
    """
    logger.info("Demonstrating NumPy array creation methods for news analytics")

    article_mentions = np.array([34, 28, 19, 23, 14, 17])
    missing_sentiment_placeholder = np.zeros(6)
    category_weights = np.ones((3, 4))   # 3 categories × 4 analytical features
    years = np.arange(2020, 2027)        # 2020 to 2026
    score_buffer = np.empty((2, 3))

    logger.info("Array creation complete")

    return {
        "article_mentions": article_mentions,
        "missing_sentiment_placeholder": missing_sentiment_placeholder,
        "category_weights": category_weights,
        "years": years,
        "score_buffer": score_buffer,
    }


def print_array_info(arrays: dict) -> None:
    """
    Print key ndarray attributes required by the lab:
    shape, dtype, ndim, size, and itemsize.
    """
    logger.info("Printing NumPy array information")

    for name, arr in arrays.items():
        print(
            f"{name:<32s} "
            f"shape={str(arr.shape):<12} "
            f"dtype={arr.dtype}  "
            f"ndim={arr.ndim}  "
            f"size={arr.size}  "
            f"itemsize={arr.itemsize}"
        )


def vectorized_operations(rating_score: np.ndarray, mentions: np.ndarray) -> dict:
    """
    Perform vectorized arithmetic for news analytics.

    No Python loops are used for mathematical operations.
    """
    logger.info("Running vectorized NumPy operations for news analytics")

    normalised = rating_score * 10
    weighted = rating_score * np.log1p(mentions)

    high_rated = rating_score > 7.5
    high_impact = (rating_score > 7.0) & (mentions > 20)

    stats = {
        "mean": float(rating_score.mean()),
        "std": float(rating_score.std()),
        "max": float(rating_score.max()),
        "min_mentions": int(mentions.min()),
        "total_mentions": int(mentions.sum()),
    }

    logger.info("Vectorized operations complete: mean=%.2f", stats["mean"])

    return {
        "normalised": normalised,
        "weighted": np.round(weighted, 2),
        "high_rated": high_rated,
        "high_impact": high_impact,
        "stats": stats,
    }


def axis_reductions(matrix: np.ndarray) -> dict:
    """
    Demonstrate reductions across rows and columns.
    """
    logger.info("Running axis-based reductions")

    col_means = matrix.mean(axis=0)
    row_means = matrix.mean(axis=1)
    col_stds = matrix.std(axis=0)

    return {
        "col_means": col_means,
        "row_means": row_means,
        "col_stds": col_stds,
    }


def broadcasting_example(rating_score: np.ndarray) -> np.ndarray:
    """
    Demonstrate broadcasting by min-max normalizing a rating_score array.
    """
    logger.info("Running broadcasting normalization example")

    min_v, max_v = rating_score.min(), rating_score.max()

    if max_v == min_v:
        return np.zeros_like(rating_score, dtype=float)

    return (rating_score - min_v) / (max_v - min_v)