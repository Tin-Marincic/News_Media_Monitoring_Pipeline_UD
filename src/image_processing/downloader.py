import os
import json
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

RAW_API_DIR = Path("data/raw/api")
RAW_IMAGES_DIR = Path("data/raw/images")


def safe_filename(name: str) -> str:
    """Clean filename so it works on Windows/macOS/Linux."""
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name.strip()


def get_filename_from_url(url: str, fallback: str = "image.jpg") -> str:
    """Extract filename from image URL."""
    try:
        path = urlparse(url).path
        filename = os.path.basename(path)

        if not filename:
            return fallback

        # If filename has no extension, add jpg
        if "." not in filename:
            filename += ".jpg"

        return safe_filename(filename)
    except Exception:
        return fallback


def load_articles_from_json(folder="data/raw/api"):
    """Load all articles from saved NewsAPI JSON files."""
    articles = []
    json_files = sorted(Path(folder).glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {folder}")
        return articles

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_articles = data.get("articles", [])
            articles.extend(file_articles)
            logger.info(f"Loaded {len(file_articles)} articles from {json_file.name}")

        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    logger.info(f"Total loaded articles: {len(articles)}")
    return articles


def download_image(image_url, dest_dir="data/raw/images", filename=None):
    """Download a single image from a full URL."""
    if not image_url:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = get_filename_from_url(image_url)

    dest = dest_dir / filename

    if dest.exists():
        logger.debug(f"Already exists: {dest}")
        return str(dest)

    try:
        resp = requests.get(image_url, stream=True, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "image" not in content_type:
            logger.warning(f"Skipped non-image URL: {image_url}")
            return None

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {dest}")
        return str(dest)

    except Exception as e:
        logger.error(f"Error downloading {image_url}: {e}")
        return None


def download_article_images(articles, dest_dir="data/raw/images", limit=5):
    """Download article images from NewsAPI article list."""
    downloaded = []
    seen_urls = set()

    for index, article in enumerate(articles):
        image_url = article.get("urlToImage")

        if not image_url or image_url in seen_urls:
            continue

        seen_urls.add(image_url)

        title = article.get("title", f"article_{index}")
        filename = get_filename_from_url(image_url, fallback=f"article_{index}.jpg")

        local = download_image(image_url, dest_dir, filename)

        print(f"Saving images to: {os.path.abspath(dest_dir)}")

        if local:
            downloaded.append({
                "article_id": index,
                "title": title,
                "image_url": image_url,
                "local_path": local,
                "source": article.get("source", {}).get("name"),
                "published_at": article.get("publishedAt")
            })

        if len(downloaded) >= limit:
            break

        time.sleep(0.1)

    logger.info(f"Downloaded {len(downloaded)} images.")
    return downloaded