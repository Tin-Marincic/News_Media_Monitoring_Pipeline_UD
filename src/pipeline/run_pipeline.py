import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.logger import logging
from src.api.client import fetch_news
from src.parsing.parsers import parse_json_files
from src.storage.s3 import upload_file_to_s3

def run_pipeline():
    try:
        logging.info("Pipeline started")

        # Step 1: Fetch API data and save raw JSON pages
        articles = fetch_news(query="technology", pages=3, page_size=5)
        logging.info(f"Fetched {len(articles)} total articles from API")

        # Step 2: Parse saved JSON files and store parsed data to MongoDB
        parsed_articles = parse_json_files()
        logging.info(f"Parsed and stored {len(parsed_articles)} articles to MongoDB")

        # Step 3: Upload raw JSON files to S3
        raw_api_dir = Path("data/raw/api")
        for file_path in raw_api_dir.glob("*.json"):
            upload_file_to_s3(str(file_path), file_path.name)

        logging.info("Pipeline finished successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()