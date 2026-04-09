from pymongo import MongoClient
from datetime import datetime, timezone


client = MongoClient("mongodb://localhost:27017/")
db = client["news_pipeline"]
collection = db["raw_articles"]


def save_to_mongo(data, source, extra_metadata=None):
    try:
        document = {
            "data": data,
            "source": source,
            "fetched_at": datetime.now(timezone.utc),
            "version": 1
        }

        if extra_metadata:
            document.update(extra_metadata)

        result = collection.insert_one(document)
        print(f"Inserted document with id: {result.inserted_id}")

    except Exception as e:
        print(f"MongoDB insert failed: {e}")


def build_document_record(data, source_file, document_type, page_number=None, extraction_library=None):
    return {
        "data": data,
        "source": source_file,
        "page_number": page_number,
        "extracted_at": datetime.now(timezone.utc),
        "type": document_type,
        "extraction_library": extraction_library
    }