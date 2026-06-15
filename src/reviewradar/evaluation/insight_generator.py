"""Generate product-level consumer insight reports from sentiment analysis."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer as SentimentScorer


logger = logging.getLogger(__name__)

COMPETITOR_KEYWORDS: dict[str, list[str]] = {
    "Steam Deck": [
        "rog ally", "asus rog", "ally x", "nintendo switch", "switch 2",
        "lenovo legion go", "legion go", "msi claw", "playstation", "ps5",
        "xbox", "onexplayer", "ayaneo", "gpd win",
    ],
    "Nintendo Switch": [
        "steam deck", "playstation", "ps5", "ps4", "xbox", "playstation 5",
        "rog ally", "lenovo legion go", "mobile", "android", "iphone",
    ],
    "Iphone 17": [
        "samsung", "galaxy", "pixel", "google pixel", "oneplus", "realme",
        "xiaomi", "vivo", "oppo", "iphone 16", "iphone 15", "16 pro",
        "15 pro", "14 pro",
    ],
}

_DEFAULT_KEYWORDS = [
    "samsung", "nintendo", "playstation", "xbox", "ps5", "ps4",
    "rog ally", "legion go", "steam deck", "switch", "pixel",
    "oneplus", "xiaomi", "realme", "vivo", "oppo",
]


def _extract_competitors(text: str, product: str) -> list[str]:
    text_lower = str(text).lower()
    found = set()
    keywords = COMPETITOR_KEYWORDS.get(product, _DEFAULT_KEYWORDS)
    for kw in keywords:
        if kw.lower() in text_lower:
            found.add(kw.title() if kw[0].isalpha() else kw)
    for kw in _DEFAULT_KEYWORDS:
        if kw.lower() in text_lower and kw.title() not in found:
            found.add(kw.title() if kw[0].isalpha() else kw)
    return list(found)


def generate_product_insights(
    df: pd.DataFrame,
    sentiment_scorer: SentimentScorer,
    product_column: str = "product_query",
    text_column: str = "cleaned_comment_text",
    output_dir: str = "data/reports/insights",
) -> dict[str, Any]:
    """Generate per-product consumer insight reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_insights: dict[str, Any] = {}

    for product in df[product_column].unique():
        product_df = df[df[product_column] == product].copy()
        texts = product_df[text_column].fillna("").tolist()
        print(f"  {product}: {len(texts)} comments")

        # Predict sentiments
        sentiments = sentiment_scorer.predict_batch(texts)
        product_df["predicted_sentiment"] = sentiments

        # ── Overall sentiment distribution (lowercase keys) ──────────
        sent_counts = Counter(sentiments)
        total_sent = sum(sent_counts.values())
        sentiment_dist = {
            s.lower(): {"count": c, "pct": round(c / total_sent * 100, 1)}
            for s, c in sent_counts.most_common()
        }

        # ── Competitor mentions ──────────────────────────────────────
        competitor_mentions: dict[str, int] = {}
        for text in texts:
            competitors = _extract_competitors(text, product)
            for comp in competitors:
                competitor_mentions[comp] = competitor_mentions.get(comp, 0) + 1
        competitor_list = sorted(
            [{"competitor": k, "mentions": v} for k, v in competitor_mentions.items()],
            key=lambda x: x["mentions"], reverse=True,
        )

        # ── Representative comments per sentiment ────────────────────
        sample_comments: dict[str, list[str]] = {}
        for sent in ["Positive", "Neutral", "Negative"]:
            subset = product_df[product_df["predicted_sentiment"] == sent][text_column].dropna()
            sample_comments[sent.lower()] = subset.head(5).tolist()

        insights = {
            "product": product,
            "total_comments": len(product_df),
            "sentiment_distribution": sentiment_dist,
            "competitor_mentions": competitor_list[:10],
            "sample_comments": sample_comments,
        }

        all_insights[product] = insights
        _write_product_report(product, insights, output_path)

    _write_summary_report(all_insights, output_path)
    return all_insights


def _write_product_report(product: str, insights: dict[str, Any], output_dir: Path) -> None:
    lines = [f"# {product} — Consumer Insights\n"]
    lines.append(f"**Total comments analyzed:** {insights['total_comments']}\n")

    # Sentiment overview
    lines.append("## Sentiment Overview\n")
    sd = insights["sentiment_distribution"]
    lines.append("| Sentiment | Count | % |")
    lines.append("|---|---|---|")
    for sent_label in ["Positive", "Neutral", "Negative"]:
        info = sd.get(sent_label.lower(), {"count": 0, "pct": 0})
        lines.append(f"| {sent_label} | {info['count']} | {info['pct']}% |")

    # Competitors
    if insights["competitor_mentions"]:
        lines.append("\n## Competitor Mentions\n")
        lines.append("| Competitor | Mentions |")
        lines.append("|---|---|")
        for item in insights["competitor_mentions"][:10]:
            lines.append(f"| {item['competitor']} | {item['mentions']} |")

    (output_dir / f"{product.lower().replace(' ', '_')}_insights.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"  Wrote {output_dir / f'{product.lower().replace(' ', '_')}_insights.md'}")


def _write_summary_report(all_insights: dict[str, Any], output_dir: Path) -> None:
    lines = ["# Product Insights Summary\n"]
    lines.append(f"Generated for {len(all_insights)} products using DistilBERT sentiment classifier.\n")

    for product, insights in all_insights.items():
        lines.append(f"## {product}\n")
        sd = insights["sentiment_distribution"]
        pos = sd.get("positive", {}).get("pct", 0)
        neg = sd.get("negative", {}).get("pct", 0)
        lines.append(f"- **Comments:** {insights['total_comments']}")
        lines.append(f"- **Sentiment:** {pos}% Positive, {neg}% Negative")
        if insights["competitor_mentions"]:
            top_comp = insights["competitor_mentions"][0]
            lines.append(f"- **Top competitor:** {top_comp['competitor']} ({top_comp['mentions']} mentions)")
        lines.append(f"- **Full report:** `{product.lower().replace(' ', '_')}_insights.md`\n")

    summary_path = output_dir / "product_insights_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {summary_path}")
