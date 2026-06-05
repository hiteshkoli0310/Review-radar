"""Centralized runtime settings and path handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class Settings:
    youtube_api_key: str
    youtube_api_service_name: str
    youtube_api_version: str
    log_level: str
    streamlit_port: int
    paths: Paths


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_paths() -> Paths:
    root = get_project_root()
    data_dir = root / os.getenv("REVIEWRADAR_DATA_DIR", "data")
    return Paths(
        project_root=root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
        models_dir=data_dir / "models",
        reports_dir=root / "reports",
        logs_dir=root / "logs",
    )


def get_settings() -> Settings:
    paths = build_paths()
    return Settings(
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
        youtube_api_service_name=os.getenv("YOUTUBE_API_SERVICE_NAME", "youtube"),
        youtube_api_version=os.getenv("YOUTUBE_API_VERSION", "v3"),
        log_level=os.getenv("REVIEWRADAR_LOG_LEVEL", "INFO"),
        streamlit_port=int(os.getenv("REVIEWRADAR_STREAMLIT_PORT", "8501")),
        paths=paths,
    )
