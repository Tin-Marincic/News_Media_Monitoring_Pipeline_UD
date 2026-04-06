import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.scraping.robots_utils import is_allowed, USER_AGENT, polite_request_delay


HEADERS = {
    "User-Agent": USER_AGENT
}


def save_raw_html(html: str, filename: str):
    output_dir = Path("data/raw/html")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(html, encoding="utf-8")


def save_scraped_json(data, filename: str):
    output_dir = Path("data/raw/scraped")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_hockey_rows(soup, source_url: str, page_number: int = None):
    results = []
    rows = soup.select("tr.team")

    for row in rows:
        record = {
            "name": row.select_one("td.name").get_text(strip=True) if row.select_one("td.name") else "",
            "year": row.select_one("td.year").get_text(strip=True) if row.select_one("td.year") else "",
            "wins": row.select_one("td.wins").get_text(strip=True) if row.select_one("td.wins") else "",
            "losses": row.select_one("td.losses").get_text(strip=True) if row.select_one("td.losses") else "",
            "source": source_url
        }

        if page_number is not None:
            record["page"] = page_number

        results.append(record)

    return results


def scrape_hockey_teams(url: str):
    if not is_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    polite_request_delay(url)

    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    save_raw_html(response.text, "hockey_page.html")

    soup = BeautifulSoup(response.text, "lxml")
    results = parse_hockey_rows(soup, url)

    save_scraped_json(results, "hockey_results.json")
    return results


def scrape_hockey_teams_multi_page(base_url: str, start_page: int = 1, end_page: int = 4):
    all_results = []

    for page in range(start_page, end_page + 1):
        url = f"{base_url}?page_num={page}"

        if not is_allowed(url):
            print(f"Blocked by robots.txt: {url}")
            continue

        polite_request_delay(url)

        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()

        save_raw_html(response.text, f"hockey_page_{page}.html")

        soup = BeautifulSoup(response.text, "lxml")
        page_results = parse_hockey_rows(soup, url, page)
        all_results.extend(page_results)

        print(f"Page {page}: scraped {len(page_results)} records")

    save_scraped_json(all_results, "hockey_multi_page_results.json")
    return all_results


if __name__ == "__main__":
    base_url = "https://www.scrapethissite.com/pages/forms/"

    single_data = scrape_hockey_teams(base_url)
    print(f"Single-page scrape: {len(single_data)} records")

    multi_data = scrape_hockey_teams_multi_page(base_url, start_page=1, end_page=4)
    print(f"Multi-page scrape: {len(multi_data)} records")