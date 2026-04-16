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
    read_file_with_encoding,
)
from src.storage.mongo import (
    save_to_mongo,
    save_batch_results_to_mongo,
    save_transcript,
)

from src.scraping.scraper import scrape_hockey_teams, scrape_hockey_teams_multi_page
from src.scraping.dynamic_scraper import scrape_ajax_movies_api
from src.ocr.ocr_utils import ocr_image, ocr_scanned_pdf

from src.image_processing.downloader import load_articles_from_json, download_article_images
from src.image_processing.batch import batch_process_images

from src.audio_processing.loader import load_audio
from src.audio_processing.processor import trim_audio, apply_fades, export_audio
from src.audio_processing.transcriber import (
    transcribe_audio,
    transcribe_long_audio,
    save_transcript_json,
    save_transcript_txt,
    save_transcript_srt,
)
from src.video_processing.loader import inspect_video, extract_audio_from_video
from src.video_processing.frame_extractor import extract_keyframes


def save_standard_transcript_outputs(result: dict, base_output_path: str) -> tuple[str, str, str]:
    """
    Save transcript as JSON, TXT, and SRT using a shared base path.
    Example base_output_path: data/processed/transcripts/news3
    """
    json_path = f"{base_output_path}.json"
    txt_path = f"{base_output_path}.txt"
    srt_path = f"{base_output_path}.srt"

    save_transcript_json(result, json_path)
    save_transcript_txt(result, txt_path)
    save_transcript_srt(result, srt_path)

    return json_path, txt_path, srt_path


def run_audio_video_stage():
    logging.info("=== Audio/Video Processing Stage Started ===")

    raw_audio_dir = Path("data/raw/audio")
    raw_video_dir = Path("data/raw/video")
    processed_audio_dir = Path("data/processed/audio")
    processed_frames_dir = Path("data/processed/frames")
    processed_transcripts_dir = Path("data/processed/transcripts")

    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    processed_transcripts_dir.mkdir(parents=True, exist_ok=True)

    # AUDIO STAGE
    if raw_audio_dir.exists():
        for audio_file in raw_audio_dir.glob("*.mp3"):
            try:
                logging.info(f"Processing audio file: {audio_file.name}")

                audio = load_audio(str(audio_file))
                logging.info(
                    f"Loaded audio {audio_file.name}: duration={len(audio)/1000:.2f}s, "
                    f"channels={audio.channels}, frame_rate={audio.frame_rate}"
                )

                trimmed = trim_audio(audio, 0, min(30000, len(audio)))
                faded = apply_fades(trimmed, fade_in_ms=1000, fade_out_ms=2000)

                processed_clip_path = processed_audio_dir / f"{audio_file.stem}_clip.mp3"
                export_audio(faded, str(processed_clip_path), fmt="mp3", bitrate="192k")
                logging.info(f"Saved processed audio clip: {processed_clip_path}")

                # Use chunked transcription for longer audio, standard transcription for shorter
                if len(audio) > 5 * 60 * 1000:
                    logging.info(f"Using chunked transcription for long audio: {audio_file.name}")
                    chunk_output_dir = processed_transcripts_dir / f"{audio_file.stem}_chunks"

                    result = transcribe_long_audio(
                        audio_path=str(audio_file),
                        output_dir=str(chunk_output_dir),
                        model_size="base",
                        chunk_minutes=5.0,
                    )

                    json_path = str(chunk_output_dir / "combined_transcript.json")
                    txt_path = str(chunk_output_dir / "combined_transcript.txt")
                    srt_path = str(chunk_output_dir / "combined_transcript.srt")
                else:
                    logging.info(f"Using standard transcription for audio: {audio_file.name}")
                    result = transcribe_audio(
                        audio_path=str(audio_file),
                        model_size="base",
                        word_timestamps=True,
                    )

                    base_output = processed_transcripts_dir / audio_file.stem
                    json_path, txt_path, srt_path = save_standard_transcript_outputs(
                        result,
                        str(base_output),
                    )

                save_transcript(
                    transcript_result=result,
                    source_path=str(audio_file),
                    source_type="audio",
                    json_path=json_path,
                    txt_path=txt_path,
                    srt_path=srt_path,
                )

                logging.info(f"Finished audio stage for: {audio_file.name}")

            except Exception as e:
                logging.error(f"Audio processing failed for {audio_file.name}: {e}")

    # VIDEO STAGE
    if raw_video_dir.exists():
        for video_file in raw_video_dir.glob("*.mp4"):
            try:
                logging.info(f"Processing video file: {video_file.name}")

                video_info = inspect_video(str(video_file))
                logging.info(
                    f"Video info for {video_file.name}: "
                    f"duration={video_info['duration_s']}s, "
                    f"fps={video_info['fps']}, "
                    f"resolution={video_info['resolution']}"
                )

                video_frames_dir = processed_frames_dir / video_file.stem
                frames = extract_keyframes(
                    str(video_file),
                    str(video_frames_dir),
                    interval_seconds=10.0,
                )
                logging.info(f"Extracted {len(frames)} keyframes for {video_file.name}")

                extracted_audio_path = processed_audio_dir / f"{video_file.stem}_from_video.mp3"
                extract_audio_from_video(str(video_file), str(extracted_audio_path))
                logging.info(f"Extracted audio from video: {extracted_audio_path}")

                result = transcribe_audio(
                    str(extracted_audio_path),
                    model_size="base",
                    word_timestamps=True,
                )

                base_output = processed_transcripts_dir / f"{video_file.stem}_video_audio"
                json_path, txt_path, srt_path = save_standard_transcript_outputs(
                    result,
                    str(base_output),
                )

                save_transcript(
                    transcript_result=result,
                    source_path=str(extracted_audio_path),
                    source_type="video_audio",
                    json_path=json_path,
                    txt_path=txt_path,
                    srt_path=srt_path,
                )

                logging.info(f"Finished video stage for: {video_file.name}")

            except Exception as e:
                logging.error(f"Video processing failed for {video_file.name}: {e}")

    logging.info("=== Audio/Video Processing Stage Complete ===")


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
                        "extraction_library": page["extraction_library"],
                    },
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
                        "extraction_library": page["extraction_library"],
                    },
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
                    "extraction_library": word_data["extraction_library"],
                },
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
                    "extraction_library": word_data["extraction_library"],
                },
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
                        "extraction_library": "python-docx",
                    },
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
                        "extraction_library": "openpyxl",
                    },
                )

            excel_summary = extract_summary_from_excel(excel_path)
            save_to_mongo(
                excel_summary,
                excel_path,
                {
                    "file_name": "news_data.xlsx",
                    "document_type": "excel_summary",
                    "extraction_library": "openpyxl",
                },
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
                    "extraction_library": "chardet",
                },
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
                    "losses": record["losses"],
                },
                record["source"],
                {
                    "file_name": "hockey_results.json",
                    "document_type": "scraped_html",
                    "extraction_library": "requests_bs4",
                },
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
                    "losses": record["losses"],
                },
                record["source"],
                {
                    "file_name": "hockey_multi_page_results.json",
                    "document_type": "scraped_html_paginated",
                    "page_number": record.get("page"),
                    "extraction_library": "requests_bs4",
                },
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
                    "year": record["year"],
                },
                record["source"],
                {
                    "file_name": "ajax_movies_api_results.json",
                    "document_type": "scraped_json_api",
                    "extraction_timestamp": record["extraction_timestamp"],
                    "extraction_library": "requests",
                },
            )
        logging.info(f"Processed dynamic JSON scraping: {len(ajax_scraped)} records")

        # Step 13: OCR on scanned image
        image_path = "data/raw/images/test_scan.png"
        if Path(image_path).exists():
            image_ocr = ocr_image(image_path)
            save_to_mongo(
                {
                    "raw_text": image_ocr["raw_text"],
                    "processed_text": image_ocr["processed_text"],
                },
                image_ocr["source"],
                {
                    "file_name": image_ocr["file_name"],
                    "document_type": image_ocr["type"],
                    "extraction_timestamp": image_ocr["extraction_timestamp"],
                    "extraction_library": "pytesseract",
                },
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
                        "processed_text": page["processed_text"],
                    },
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": "pytesseract_pdf2image",
                    },
                )
            logging.info(f"Processed OCR scanned PDF: {len(pdf_ocr_results)} pages")

        # Step 15: Load article metadata from saved JSON and download article images
        image_articles = load_articles_from_json("data/raw/api")
        downloaded_images = download_article_images(
            image_articles,
            dest_dir="data/raw/images",
            limit=10,
        )
        logging.info(f"Downloaded {len(downloaded_images)} article images")

        # Step 16: Batch process images
        image_results, image_errors = batch_process_images(
            input_dir="data/raw/images",
            output_dir="data/processed",
            max_width=500,
            thumb_size=(128, 128),
            convert_webp=True,
            extract_metadata=True,
            upload_to_drive=True,
        )
        logging.info(f"Batch processed {len(image_results)} images with {len(image_errors)} errors")

        # Step 17: Save image metadata to MongoDB
        save_batch_results_to_mongo(image_results)
        logging.info(f"Saved {len(image_results)} image metadata records to MongoDB")

        # Step 18: Audio/Video stage
        run_audio_video_stage()

        logging.info("Pipeline finished successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()