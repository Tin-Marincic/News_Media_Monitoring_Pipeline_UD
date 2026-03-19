import os
import time
import json
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"


def save_json(data, filename):
    output_dir = Path("data/raw/api")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved {file_path}")


def fetch_single_page(query="technology", page=1, page_size=5, max_retries=3):
    params = {
        "q": query,
        "page": page,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
    }

    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)

            if response.status_code == 429:
                print("Rate limit reached. Waiting 3 seconds...")
                time.sleep(3)
                retries += 1
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            print(f"Request failed on page {page}: {e}")
            retries += 1
            time.sleep(2)

    return None


def fetch_news(query="technology", pages=3, page_size=5):
    all_articles = []

    for page in range(1, pages + 1):
        print(f"Fetching page {page}...")
        data = fetch_single_page(query=query, page=page, page_size=page_size)

        if data:
            save_json(data, f"news_page_{page}.json")
            articles = data.get("articles", [])
            all_articles.extend(articles)

    return all_articles


if __name__ == "__main__":
    if not API_KEY:
        print("NEWS_API_KEY not found in .env")
    else:
        articles = fetch_news(pages=3)
        print(f"Fetched {len(articles)} articles.")

        if articles:
            print("Here are some article titles:")
            for article in articles[:5]:
                print(article.get("title", "No title"))