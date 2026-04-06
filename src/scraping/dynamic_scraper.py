import json
from pathlib import Path
from datetime import datetime

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.scraping.robots_utils import is_allowed, USER_AGENT, polite_request_delay


HEADERS = {
    "User-Agent": USER_AGENT
}


def save_scraped_json(data, filename: str):
    output_dir = Path("data/raw/scraped")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def scrape_ajax_movies_api(start_year: int = 2010, end_year: int = 2015):
    """
    Scrape the AJAX page through its JSON endpoint.
    """
    base_page = "https://www.scrapethissite.com/pages/ajax-javascript/"
    api_url = "https://www.scrapethissite.com/pages/ajax-javascript/?ajax=true&year={year}"

    if not is_allowed(base_page):
        raise PermissionError(f"Blocked by robots.txt: {base_page}")

    all_results = []

    for year in range(start_year, end_year + 1):
        polite_request_delay(base_page)

        current_url = api_url.format(year=year)
        response = requests.get(current_url, headers=HEADERS, timeout=20)
        response.raise_for_status()

        data = response.json()

        for item in data:
            record = {
                "title": item.get("title", ""),
                "nominations": item.get("nominations", ""),
                "awards": item.get("awards", ""),
                "best_picture": item.get("best_picture", False),
                "year": year,
                "source": current_url,
                "type": "scraped_json_api",
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            all_results.append(record)

        print(f"Year {year}: scraped {len(data)} records")

    save_scraped_json(all_results, "ajax_movies_api_results.json")
    return all_results


def scrape_with_selenium_fallback(url: str):
    """
    Fallback for pages that require real browser rendering.
    """
    if not is_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    polite_request_delay(url)

    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"user-agent={USER_AGENT}")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        html = driver.page_source

        output_dir = Path("data/raw/html")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dynamic_page_selenium.html").write_text(html, encoding="utf-8")

        return {
            "source": url,
            "type": "scraped_selenium_fallback",
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "html_saved": "data/raw/html/dynamic_page_selenium.html"
        }

    finally:
        driver.quit()


if __name__ == "__main__":
    data = scrape_ajax_movies_api()
    print(f"Total API records scraped: {len(data)}")