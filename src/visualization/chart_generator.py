"""
src/visualization/chart_generator.py

Orchestrator module for Lab 12 visualization generation.

Loads the cleaned News Media Monitoring dataset and generates:
- 8 static charts as PNG + PDF
- 5 interactive charts as self-contained HTML files
"""

from pathlib import Path
import logging

import pandas as pd

from src.visualization.static_charts import generate_all_static_charts
from src.visualization.interactive_charts import generate_all_interactive_charts

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/processed/cleaned/cleaned_data.csv")
DEFAULT_STATIC_DIR = Path("outputs/visualizations/static")
DEFAULT_INTERACTIVE_DIR = Path("outputs/visualizations/interactive")


def load_visualization_data(data_path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset used for Lab 12 visualizations.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned visualization dataset not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)

    logger.info(
        "Loaded visualization dataset from %s with shape %s",
        data_path,
        df.shape,
    )

    return df


def generate_visualizations(
    data_path=DEFAULT_DATA_PATH,
    static_dir=DEFAULT_STATIC_DIR,
    interactive_dir=DEFAULT_INTERACTIVE_DIR,
) -> dict:
    """
    Generate all static and interactive visualizations.

    Returns a dictionary containing paths to all generated chart files.
    """
    logger.info("=== Lab 12 Visualization Generation Started ===")

    df = load_visualization_data(data_path)

    static_dir = Path(static_dir)
    interactive_dir = Path(interactive_dir)

    static_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir.mkdir(parents=True, exist_ok=True)

    print("Generating static charts...")
    static_paths = generate_all_static_charts(df, output_dir=static_dir)

    print("Generating interactive charts...")
    interactive_paths = generate_all_interactive_charts(df, output_dir=interactive_dir)

    result = {
        "data_path": str(data_path),
        "dataset_shape": df.shape,
        "static_chart_count": len(static_paths),
        "static_file_count": sum(len(item) for item in static_paths),
        "interactive_chart_count": len(interactive_paths),
        "static_paths": static_paths,
        "interactive_paths": interactive_paths,
    }

    logger.info(
        "Generated %d static charts / %d static files and %d interactive charts",
        result["static_chart_count"],
        result["static_file_count"],
        result["interactive_chart_count"],
    )

    logger.info("=== Lab 12 Visualization Generation Complete ===")

    return result