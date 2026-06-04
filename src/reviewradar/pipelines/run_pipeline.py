"""End-to-end pipeline runner."""

from __future__ import annotations

from reviewradar.config.settings import get_settings
from reviewradar.utils.logging import configure_logging


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


if __name__ == "__main__":
    run()