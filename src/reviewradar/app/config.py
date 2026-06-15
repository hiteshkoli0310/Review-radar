"""App-specific paths, model config, and constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data
MASTER_CSV = PROJECT_ROOT / "data" / "exports" / "reviewradar_master_raw.csv"
ANNOTATION_CSV = PROJECT_ROOT / "data" / "annotation" / "manual_review_sample.csv"
ANNOTATION_GUIDELINES = PROJECT_ROOT / "data" / "annotation" / "annotation_guidelines.md"

# Reports
SENTIMENT_EVAL_JSON = PROJECT_ROOT / "data" / "reports" / "sentiment_evaluation.json"
ASPECT_EVAL_JSON = PROJECT_ROOT / "data" / "reports" / "aspect_classification_oversampled.json"
INSIGHT_DIR = PROJECT_ROOT / "data" / "reports" / "insights"
INSIGHT_OVERSAMPLED_DIR = PROJECT_ROOT / "data" / "reports" / "insights_oversampled"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Models
SENTIMENT_MODEL = PROJECT_ROOT / "models" / "distilbert_sentiment"
ASPECT_MODEL = PROJECT_ROOT / "models" / "distilbert_aspect_oversampled"

# Pipeline
PIPELINE_CONFIG = {
    "video_search_limit": 20,
    "comments_per_video": 100,
}

# Dashboard
DEFAULT_PRODUCT = "Iphone 17"
PAGE_TITLE = "ReviewRadar — Consumer Intelligence Platform"
