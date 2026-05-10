"""
api.py — Data source: fetch blog posts from a REST API.

Primary endpoint: JSONPlaceholder
Fallback endpoint: DummyJSON

Returns JSON arrays of post objects: { "id": int, "title": str, "body": str }
"""

import requests
from typing import List, Dict, Any
from utils import logger, retry


class APIError(Exception):
    """Raised when every API endpoint has been exhausted."""


@retry(max_attempts=3, delay=1.0)
def fetch_posts(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch blog posts with automatic fallback.

    Args:
        limit: Number of posts to fetch.

    Raises:
        APIError: If both endpoints fail.
    """
    # Configure endpoints and their respective limit parameters
    endpoints = [
        ("JSONPlaceholder", f"https://jsonplaceholder.typicode.com/posts?_limit={limit}"),
        ("DummyJSON",       f"https://dummyjson.com/posts?limit={limit}"),
    ]

    for name, url in endpoints:
        try:
            logger.info(f"Fetching {limit} posts from {name}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Normalize DummyJSON response format
            if isinstance(data, dict) and "posts" in data:
                data = data["posts"]

            logger.info(f"Fetched {len(data)} posts from {name}.")
            return data

        except requests.RequestException as exc:
            logger.warning(f"{name} unavailable: {exc}")

    raise APIError("All API endpoints failed. Cannot fetch posts.")
