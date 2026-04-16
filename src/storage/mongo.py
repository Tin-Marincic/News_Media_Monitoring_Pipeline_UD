from pymongo import MongoClient
from datetime import datetime
from pathlib import Path
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["news_pipeline"]

collection = db["raw_articles"]
image_metadata_collection = db["image_metadata"]
transcripts_collection = db["transcripts"]


def get_db():
    return db


def save_to_mongo(data, source, extra_metadata=None):
    document = {
        "data": data,
        "source": source,
        "fetched_at": datetime.utcnow(),
        "version": 1
    }

    if extra_metadata:
        document.update(extra_metadata)

    collection.insert_one(document)
    logger.info(f"Inserted raw article document from source: {source}")
    print("Inserted document:", document)


def build_scraped_record(data, source_url, page_number=None):
    return {
        "data": data,
        "source": source_url,
        "page_number": page_number,
        "extracted_at": datetime.utcnow(),
        "type": "web_scraping"
    }


def build_ocr_record(text, source_file, page_number=None):
    return {
        "data": {"text": text},
        "source": source_file,
        "page_number": page_number,
        "extracted_at": datetime.utcnow(),
        "type": "ocr"
    }


def save_image_metadata(metadata_list):
    for meta in metadata_list:
        meta["processed_at"] = datetime.utcnow().isoformat()

        image_metadata_collection.update_one(
            {"filename": meta["filename"]},
            {"$set": meta},
            upsert=True
        )

    logger.info(f"Saved {len(metadata_list)} image metadata records to MongoDB")
    print(f"Saved {len(metadata_list)} image records to MongoDB")


def get_image_metadata(article_id=None):
    query = {"article_id": article_id} if article_id else {}
    return list(image_metadata_collection.find(query, {"_id": 0}))


def apply_image_metadata(image_path, article_id=None, source="news_api", image_type="article_image"):
    image_path = Path(image_path)

    with Image.open(image_path) as img:
        width, height = img.size
        file_size = os.path.getsize(image_path)

        metadata = {
            "filename": image_path.name,
            "article_id": article_id,
            "source": source,
            "type": image_type,
            "original_path": str(image_path),
            "format": img.format,
            "mode": img.mode,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3),
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 1)
        }

    return metadata


def process_images_and_save_metadata(image_paths, article_id=None):
    metadata_list = []

    for image_path in image_paths:
        img_metadata = apply_image_metadata(image_path, article_id)
        metadata_list.append(img_metadata)

    save_image_metadata(metadata_list)
    return metadata_list


def save_batch_results_to_mongo(results):
    cleaned = []

    for item in results:
        doc = {
            "filename": item.get("filename"),
            "format": item.get("format"),
            "mode": item.get("mode"),
            "width": item.get("width"),
            "height": item.get("height"),
            "aspect_ratio": item.get("aspect_ratio"),
            "file_size_bytes": item.get("file_size_bytes"),
            "file_size_kb": item.get("file_size_kb"),
            "original_path": item.get("original_path"),
            "resized_path": item.get("resized_path"),
            "thumbnail_path": item.get("thumbnail_path"),
            "webp_path": item.get("webp_path"),
            "exif": item.get("exif"),
            "source": "news_api",
            "type": "article_image"
        }
        cleaned.append(doc)

    save_image_metadata(cleaned)


def save_transcript(
    transcript_result: dict,
    source_path: str,
    source_type: str = "audio",
    json_path: str = None,
    txt_path: str = None,
    srt_path: str = None
) -> str:
    doc = {
        "source_file": transcript_result.get("source_file", ""),
        "source_path": source_path,
        "source_type": source_type,   # "audio" or "video_audio"
        "model": transcript_result.get("model", "unknown"),
        "language": transcript_result.get("language", ""),
        "language_probability": transcript_result.get("language_probability", 0),
        "duration_s": transcript_result.get("duration_s", 0),
        "total_chunks": transcript_result.get("total_chunks"),
        "full_text": transcript_result.get("full_text", ""),
        "segments": transcript_result.get("segments", []),
        "transcript_json_path": json_path,
        "transcript_txt_path": txt_path,
        "transcript_srt_path": srt_path,
        "transcribed_at": datetime.utcnow().isoformat(),
        "type": "transcript"
    }

    result = transcripts_collection.insert_one(doc)

    logger.info(
        f"Saved transcript to MongoDB: {doc['source_file']} "
        f"(id={result.inserted_id})"
    )

    return str(result.inserted_id)


def get_transcripts(source_type=None):
    query = {"source_type": source_type} if source_type else {}
    return list(transcripts_collection.find(query, {"_id": 0}))