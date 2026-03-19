import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.mongo import save_to_mongo

def parse_json_files():
    json_dir = Path("data/raw/api")
    parsed_articles = []

    if not json_dir.exists():
        print(f"JSON directory not found: {json_dir}")
        return parsed_articles

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return parsed_articles

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])

        for article in articles:
            parsed_article = {
                "source": article.get("source", {}).get("name"),
                "author": article.get("author"),
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt"),
            }
            parsed_articles.append(parsed_article)
            save_to_mongo(parsed_article, file_path.name)

        print(f"Parsed {len(articles)} articles from {file_path.name}")

    print(f"Total parsed JSON articles: {len(parsed_articles)}")
    return parsed_articles


def parse_csv_file(file_path):
    parsed_rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_rows.append(row)
            save_to_mongo(row, Path(file_path).name)

    print(f"Parsed {len(parsed_rows)} rows from CSV: {file_path}")
    return parsed_rows


def parse_xml_file(file_path):
    parsed_items = []

    tree = ET.parse(file_path)
    root = tree.getroot()

    for article in root.findall("article"):
        parsed_article = {
            "id": article.findtext("id"),
            "title": article.findtext("title"),
            "category": article.findtext("category"),
        }
        parsed_items.append(parsed_article)
        save_to_mongo(parsed_article, Path(file_path).name)

    print(f"Parsed {len(parsed_items)} items from XML: {file_path}")
    return parsed_items


if __name__ == "__main__":
    json_data = parse_json_files()
    print("First 2 parsed JSON articles:")
    print(json_data[:2])

    csv_data = parse_csv_file("data/raw/csv/sample.csv")
    print("CSV data:")
    print(csv_data)

    xml_data = parse_xml_file("data/raw/xml/sample.xml")
    print("XML data:")
    print(xml_data)