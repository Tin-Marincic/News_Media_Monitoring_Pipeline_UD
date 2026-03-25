from pymongo import MongoClient
from datetime import datetime


client = MongoClient("mongodb://localhost:27017/")
db = client["news_pipeline"]
collection = db["raw_articles"]

def save_to_mongo(data, source):
    try:
        document = {
            "source_file": source,
            "ingested_at": datetime.utcnow(),
            "data": data
        }
        result = collection.insert_one(document)
        print(f"Inserted document with id: {result.inserted_id}")
    except Exception as e:
        print(f"MongoDB insert failed: {e}")