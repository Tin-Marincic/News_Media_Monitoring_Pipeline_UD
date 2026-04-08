from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import time
import requests

USER_AGENT = "ResearchBot/1.0"
DEFAULT_DELAY = 1.5

_robots_cache = {}


def get_robots_parser(url: str):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    if base in _robots_cache:
        return _robots_cache[base]

    robots_url = f"{base}/robots.txt"

    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=15
        )
        response.raise_for_status()

        rp = RobotFileParser()
        rp.parse(response.text.splitlines())

        _robots_cache[base] = rp
        return rp

    except Exception as e:
        print(f"Could not read robots.txt from {robots_url}: {e}")
        _robots_cache[base] = None
        return None


def is_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    rp = get_robots_parser(url)

    if rp is None:
        return False

    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def get_crawl_delay(url: str, user_agent: str = USER_AGENT) -> float:
    rp = get_robots_parser(url)

    if rp is None:
        return DEFAULT_DELAY

    try:
        delay = rp.crawl_delay(user_agent)
        return delay if delay is not None else DEFAULT_DELAY
    except Exception:
        return DEFAULT_DELAY


def polite_request_delay(url: str, user_agent: str = USER_AGENT):
    delay = get_crawl_delay(url, user_agent)
    time.sleep(delay)