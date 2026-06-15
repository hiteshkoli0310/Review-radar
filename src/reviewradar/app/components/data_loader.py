"""Data loading helpers — load existing reports, CSV without recomputation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.app.config import (
    MASTER_CSV,
    ANNOTATION_CSV,
    SENTIMENT_EVAL_JSON,
    ASPECT_EVAL_JSON,
    INSIGHT_DIR,
    INSIGHT_OVERSAMPLED_DIR,
    CHARTS_DIR,
)


# ── Master CSV ───────────────────────────────────────────────────────────────

@pd.api.extensions.register_dataframe_accessor("reviewradar")
class ReviewRadarAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def products(self) -> list[str]:
        return sorted(self._obj["product_query"].dropna().unique())


_master: pd.DataFrame | None = None


def load_master() -> pd.DataFrame:
    global _master
    if _master is None:
        _master = pd.read_csv(MASTER_CSV)
    return _master


def invalidate_master_cache() -> None:
    global _master
    _master = None


def load_annotation() -> pd.DataFrame:
    return pd.read_csv(ANNOTATION_CSV)


def get_product_count(master: pd.DataFrame | None = None) -> int:
    df = master if master is not None else load_master()
    return df["product_query"].nunique()


def get_total_comments(master: pd.DataFrame | None = None) -> int:
    df = master if master is not None else load_master()
    return len(df)


def get_annotated_count() -> int:
    return len(pd.read_csv(ANNOTATION_CSV))


def get_product_names() -> list[str]:
    return load_master().reviewradar.products


# ── Product-specific data ────────────────────────────────────────────────────

def get_product_df(product: str) -> pd.DataFrame:
    master = load_master()
    return master[master["product_query"] == product].copy()


# ── Evaluation JSONs ─────────────────────────────────────────────────────────

_sentiment_eval: dict[str, Any] | None = None
_aspect_eval: dict[str, Any] | None = None


def load_sentiment_eval() -> dict[str, Any]:
    global _sentiment_eval
    if _sentiment_eval is None:
        with open(SENTIMENT_EVAL_JSON) as f:
            _sentiment_eval = json.load(f)
    return _sentiment_eval


def load_aspect_eval() -> dict[str, Any]:
    global _aspect_eval
    if _aspect_eval is None:
        with open(ASPECT_EVAL_JSON) as f:
            _aspect_eval = json.load(f)
    return _aspect_eval


# ── Insight report parsing ───────────────────────────────────────────────────

def _parse_insight_markdown(slug: str, insight_dir: Path, product: str) -> dict[str, Any] | None:
    """Parse a human-readable markdown report into a structured dict."""
    md_path = insight_dir / f"{slug}_insights.md"
    if not md_path.exists():
        return None

    text = md_path.read_text(encoding="utf-8")
    result: dict[str, Any] = {"product": product, "raw_path": str(md_path)}

    # Total comments
    m = re.search(r"\*\*Total comments analyzed:\s*(\d+)", text)
    if m:
        result["total_comments"] = int(m.group(1))

    # Sentiment distribution
    sent = {}
    for label in ("Positive", "Neutral", "Negative"):
        pat = re.compile(rf"\|\s*{label}\s*\|\s*(\d+)\s*\|\s*([\d.]+)%")
        m2 = pat.search(text)
        if m2:
            sent[label.lower()] = {"count": int(m2.group(1)), "pct": float(m2.group(2))}
    result["sentiment_distribution"] = sent

    # Competitor mentions
    competitors = []
    if "## Competitor Mentions" in text:
        section = text.split("## Competitor Mentions")[1].split("##")[0]
        for line in section.split("\n"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2 and parts[0][0].isalpha():
                try:
                    competitors.append({
                        "competitor": parts[0],
                        "mentions": int(parts[1]),
                    })
                except (ValueError, IndexError):
                    pass
    result["competitor_mentions"] = competitors

    return result


def parse_insight_report(product: str, use_oversampled: bool = False) -> dict[str, Any] | None:
    """Return a structured insight dict for a product.

    Parses the human-readable markdown report.
    """
    insight_dir = INSIGHT_OVERSAMPLED_DIR if use_oversampled else INSIGHT_DIR
    slug = product.lower().replace(" ", "_")

    result = _parse_insight_markdown(slug, insight_dir, product)
    if result is not None:
        return result

    fallback_dir = INSIGHT_OVERSAMPLED_DIR if not use_oversampled else INSIGHT_DIR
    result = _parse_insight_markdown(slug, fallback_dir, product)
    if result is not None:
        return result

    return None


# ── Summary report ───────────────────────────────────────────────────────────

def load_insight_summary() -> str:
    path = INSIGHT_DIR / "product_insights_summary.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_chart_path(name: str) -> Path | None:
    path = CHARTS_DIR / name
    return path if path.exists() else None
