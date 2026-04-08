import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.logger import logging
from src.api.client import fetch_news
from src.parsing.parsers import (
    parse_json_files,
    extract_text_from_pdf,
    extract_text_from_two_column_pdf,
    extract_text_from_word,
    extract_text_from_two_column_word,
    extract_word_runs,
    extract_data_from_excel,
    extract_summary_from_excel,
    read_file_with_encoding
)
from src.storage.mongo import save_to_mongo
from src.storage.s3 import upload_file_to_s3

from src.scraping.scraper import scrape_hockey_teams, scrape_hockey_teams_multi_page
from src.scraping.dynamic_scraper import scrape_ajax_movies_api
from src.ocr.ocr_utils import ocr_image, ocr_scanned_pdf


def run_pipeline():
    try:
        logging.info("Pipeline started")

        # Step 1: Fetch API data and save raw JSON pages
        articles = fetch_news(query="technology", pages=3, page_size=5)
        logging.info(f"Fetched {len(articles)} total articles from API")

        # Step 2: Parse saved JSON files and store parsed data to MongoDB
        parsed_articles = parse_json_files()
        logging.info(f"Parsed and stored {len(parsed_articles)} JSON articles to MongoDB")

        # Step 3: Process normal PDF
        normal_pdf = "data/raw/pdf/news_normal.pdf"
        if Path(normal_pdf).exists():
            pdf_pages = extract_text_from_pdf(normal_pdf)
            for page in pdf_pages:
                save_to_mongo(
                    {"text": page["text"], "tables": page["tables"]},
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["document_type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": page["extraction_library"]
                    }
                )
            logging.info(f"Processed normal PDF: {normal_pdf}")

        # Step 4: Process two-column PDF
        two_column_pdf = "data/raw/pdf/news_two_column.pdf"
        if Path(two_column_pdf).exists():
            pdf_pages = extract_text_from_two_column_pdf(two_column_pdf)
            for page in pdf_pages:
                save_to_mongo(
                    {"text": page["text"], "tables": page["tables"]},
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["document_type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": page["extraction_library"]
                    }
                )
            logging.info(f"Processed two-column PDF: {two_column_pdf}")

        # Step 5: Process normal Word
        normal_word = "data/raw/word/news_normal.docx"
        if Path(normal_word).exists():
            word_data = extract_text_from_word(normal_word)
            save_to_mongo(
                {"text": word_data["text"], "tables": word_data["tables"]},
                word_data["source"],
                {
                    "file_name": word_data["file_name"],
                    "document_type": word_data["document_type"],
                    "extraction_timestamp": word_data["extraction_timestamp"],
                    "extraction_library": word_data["extraction_library"]
                }
            )
            logging.info(f"Processed normal Word file: {normal_word}")

        # Step 6: Process two-column Word
        two_column_word = "data/raw/word/news_two_column.docx"
        if Path(two_column_word).exists():
            word_data = extract_text_from_two_column_word(two_column_word)
            save_to_mongo(
                {"text": word_data["text"], "tables": word_data["tables"]},
                word_data["source"],
                {
                    "file_name": word_data["file_name"],
                    "document_type": word_data["document_type"],
                    "extraction_timestamp": word_data["extraction_timestamp"],
                    "extraction_library": word_data["extraction_library"]
                }
            )
            logging.info(f"Processed two-column Word file: {two_column_word}")

        # Step 7: Process Word runs
        if Path(normal_word).exists():
            word_runs = extract_word_runs(normal_word)
            for run in word_runs:
                save_to_mongo(
                    run,
                    normal_word,
                    {
                        "file_name": Path(normal_word).name,
                        "document_type": "word_run",
                        "extraction_library": "python-docx"
                    }
                )
            logging.info(f"Processed Word runs for file: {normal_word}")

        # Step 8: Process Excel
        excel_path = "data/raw/excel/news_data.xlsx"
        if Path(excel_path).exists():
            excel_articles = extract_data_from_excel(excel_path)
            for article in excel_articles:
                save_to_mongo(
                    article,
                    excel_path,
                    {
                        "file_name": "news_data.xlsx",
                        "document_type": "excel",
                        "extraction_library": "openpyxl"
                    }
                )

            excel_summary = extract_summary_from_excel(excel_path)
            save_to_mongo(
                excel_summary,
                excel_path,
                {
                    "file_name": "news_data.xlsx",
                    "document_type": "excel_summary",
                    "extraction_library": "openpyxl"
                }
            )
            logging.info(f"Processed Excel file: {excel_path}")

        # Step 9: Encoding test
        encoding_file = "data/raw/api/news_page_1.json"
        if Path(encoding_file).exists():
            encoding_text = read_file_with_encoding(encoding_file)
            save_to_mongo(
                {"preview_text": encoding_text[:300]},
                encoding_file,
                {
                    "file_name": "news_page_1.json",
                    "document_type": "encoding_test",
                    "extraction_library": "chardet"
                }
            )
            logging.info(f"Encoding test processed for: {encoding_file}")

        # Step 10: Single-page web scraping
        hockey_url = "https://www.scrapethissite.com/pages/forms/"
        single_scraped = scrape_hockey_teams(hockey_url)
        for record in single_scraped:
            save_to_mongo(
                {
                    "name": record["name"],
                    "year": record["year"],
                    "wins": record["wins"],
                    "losses": record["losses"]
                },
                record["source"],
                {
                    "file_name": "hockey_results.json",
                    "document_type": "scraped_html",
                    "extraction_library": "requests_bs4"
                }
            )
        logging.info(f"Processed single-page scraping: {len(single_scraped)} records")

        # Step 11: Multi-page web scraping
        multi_scraped = scrape_hockey_teams_multi_page(hockey_url, start_page=1, end_page=4)
        for record in multi_scraped:
            save_to_mongo(
                {
                    "name": record["name"],
                    "year": record["year"],
                    "wins": record["wins"],
                    "losses": record["losses"]
                },
                record["source"],
                {
                    "file_name": "hockey_multi_page_results.json",
                    "document_type": "scraped_html_paginated",
                    "page_number": record.get("page"),
                    "extraction_library": "requests_bs4"
                }
            )
        logging.info(f"Processed multi-page scraping: {len(multi_scraped)} records")

        # Step 12: Dynamic JSON API scraping
        ajax_scraped = scrape_ajax_movies_api()
        for record in ajax_scraped:
            save_to_mongo(
                {
                    "title": record["title"],
                    "nominations": record["nominations"],
                    "awards": record["awards"],
                    "best_picture": record["best_picture"],
                    "year": record["year"]
                },
                record["source"],
                {
                    "file_name": "ajax_movies_api_results.json",
                    "document_type": "scraped_json_api",
                    "extraction_timestamp": record["extraction_timestamp"],
                    "extraction_library": "requests"
                }
            )
        logging.info(f"Processed dynamic JSON scraping: {len(ajax_scraped)} records")

        # Step 13: OCR on scanned image
        image_path = "data/raw/images/test_scan.png"
        if Path(image_path).exists():
            image_ocr = ocr_image(image_path)
            save_to_mongo(
                {
                    "raw_text": image_ocr["raw_text"],
                    "processed_text": image_ocr["processed_text"]
                },
                image_ocr["source"],
                {
                    "file_name": image_ocr["file_name"],
                    "document_type": image_ocr["type"],
                    "extraction_timestamp": image_ocr["extraction_timestamp"],
                    "extraction_library": "pytesseract"
                }
            )
            logging.info("Processed OCR image")

        # Step 14: OCR on scanned PDF
        scanned_pdf = "data/raw/scanned/test_scan.pdf"
        if Path(scanned_pdf).exists():
            pdf_ocr_results = ocr_scanned_pdf(scanned_pdf)
            for page in pdf_ocr_results:
                save_to_mongo(
                    {
                        "raw_text": page["raw_text"],
                        "processed_text": page["processed_text"]
                    },
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": "pytesseract_pdf2image"
                    }
                )
            logging.info(f"Processed OCR scanned PDF: {len(pdf_ocr_results)} pages")

        # Step 15: Upload raw JSON files to S3
        raw_api_dir = Path("data/raw/api")
        for file_path in raw_api_dir.glob("*.json"):
            upload_file_to_s3(str(file_path), file_path.name)

        logging.info("Pipeline finished successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()