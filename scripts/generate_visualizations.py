"""
CLI entry point for Lab 12 visualizations.

Run from project root:

    python scripts/generate_visualizations.py

Optional custom data path:

    python scripts/generate_visualizations.py --data data/processed/cleaned/cleaned_data.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.chart_generator import generate_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Lab 12 static and interactive visualizations."
    )

    parser.add_argument(
        "--data",
        default="data/processed/cleaned/cleaned_data.csv",
        help="Path to cleaned CSV dataset.",
    )

    parser.add_argument(
        "--static-dir",
        default="outputs/visualizations/static",
        help="Output directory for PNG/PDF static charts.",
    )

    parser.add_argument(
        "--interactive-dir",
        default="outputs/visualizations/interactive",
        help="Output directory for interactive HTML charts.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = generate_visualizations(
        data_path=args.data,
        static_dir=args.static_dir,
        interactive_dir=args.interactive_dir,
    )

    print("\nVisualization generation complete.")
    print("Dataset shape:", result["dataset_shape"])
    print("Static charts:", result["static_chart_count"])
    print("Static files:", result["static_file_count"])
    print("Interactive charts:", result["interactive_chart_count"])

    print("\nStatic outputs:")
    for item in result["static_paths"]:
        print("  PNG:", item["png"])
        print("  PDF:", item["pdf"])

    print("\nInteractive outputs:")
    for path in result["interactive_paths"]:
        print(" ", path)


if __name__ == "__main__":
    main()