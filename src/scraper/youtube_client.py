"""Reusable YouTube Data API v3 client setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from googleapiclient.discovery import build


YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_KEY_ENV_VAR = "YOUTUBE_API_KEY"


def load_youtube_api_key(env_path: str | Path | None = None) -> str:
    """Load the YouTube API key from environment variables or a .env file."""
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv(YOUTUBE_API_KEY_ENV_VAR)

    if not api_key:
        raise ValueError(
            f"{YOUTUBE_API_KEY_ENV_VAR} is missing. Add it to your .env file "
            "before running the data collection pipeline."
        )

    return api_key


def get_youtube_client(api_key: str | None = None) -> Any:
    """Return an authenticated YouTube Data API v3 client."""
    developer_key = api_key or load_youtube_api_key()
    return build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=developer_key,
        cache_discovery=False,
    )

