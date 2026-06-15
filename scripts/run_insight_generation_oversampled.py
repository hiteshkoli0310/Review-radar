"""Regenerate insight reports with oversampled aspect model and compare outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer as SentimentScorer
from reviewradar.evaluation.aspect_evaluation import DistilBertAspectScorer as AspectScorer
from reviewradar.evaluation.insight_generator import generate_product_insights


def compare_insights(old_dir: Path, new_dir: Path) -> str:
    lines = ["# Insight Report Comparison: Baseline vs Oversampled Aspect Model\n"]
    
    products = ["iphone_17", "nintendo_switch", "steam_deck"]
    suffix = "_insights.md"

    for prod in products:
        old_path = old_dir / f"{prod}{suffix}"
        new_path = new_dir / f"{prod}{suffix}"

        if not old_path.exists() or not new_path.exists():
            lines.append(f"\n## {prod.replace('_',' ').title()}\n  (one or both reports missing)\n")
            continue

        old_text = old_path.read_text(encoding="utf-8")
        new_text = new_path.read_text(encoding="utf-8")

        lines.append(f"\n## {prod.replace('_',' ').title()}\n")
        lines.append("| Metric | Baseline Model | Oversampled Model |\n|---|---|---|")

        # Extract sentiment overview
        for line in old_text.split("\n"):
            if "comments analyzed" in line:
                lines.append(f"| Comments | {line.strip()} |\n")
                break

        # Extract sentiment percentages
        for sent in ["Positive", "Neutral", "Negative"]:
            old_val = "?"
            new_val = "?"
            for i, l in enumerate(old_text.split("\n")):
                if l.startswith(f"| {sent} |"):
                    old_val = l.split("|")[3].strip()
                    break
            for i, l in enumerate(new_text.split("\n")):
                if l.startswith(f"| {sent} |"):
                    new_val = l.split("|")[3].strip()
                    break
            lines.append(f"| {sent} | {old_val} | {new_val} |")

        # Extract top aspect
        for section_flag, label in [("Most Discussed Aspects", "Top aspect")]:
            old_section = old_text.split(f"## {section_flag}")[1].split("##")[0] if f"## {section_flag}" in old_text else ""
            new_section = new_text.split(f"## {section_flag}")[1].split("##")[0] if f"## {section_flag}" in new_text else ""
            old_top = old_section.strip().split("\n")[2] if old_section else "?"
            new_top = new_section.strip().split("\n")[2] if new_section else "?"
            
        # Spam count
        for text in [old_text, new_text]:
            for line in text.split("\n"):
                if "flagged as Spam" in line:
                    pass  # skip — already in the report body

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    master_path = Path("data/exports/reviewradar_master_raw.csv")
    old_insights_dir = Path("data/reports/insights")
    new_insights_dir = Path("data/reports/insights_oversampled")

    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} rows")

    print("Loading models...")
    sentiment_scorer = SentimentScorer(model_path="models/distilbert_sentiment")
    aspect_scorer = AspectScorer(model_path="models/distilbert_aspect_oversampled")
    print("  Done")

    print("\nGenerating insights with oversampled aspect model...")
    insights = generate_product_insights(
        df, sentiment_scorer, aspect_scorer, output_dir=str(new_insights_dir),
    )
    print(f"\nReports saved to {new_insights_dir.resolve()}")

    # Compare
    if old_insights_dir.exists():
        print("\nComparing with previous reports...")
        comparison = compare_insights(old_insights_dir, new_insights_dir)
        comparison_path = new_insights_dir.parent / "insight_model_comparison.md"
        comparison_path.write_text(comparison, encoding="utf-8")
        print(f"Comparison: {comparison_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("PRODUCT INSIGHTS — OVERVIEW (Oversampled Model)")
    print(f"{'='*60}")
    for product, data in insights.items():
        sd = data["sentiment_distribution"]
        pos = sd.get("Positive", {}).get("pct", 0)
        neg = sd.get("Negative", {}).get("pct", 0)
        spam = data["spam_count"]
        if data["most_discussed_aspects"]:
            top = data["most_discussed_aspects"][0]
            print(f"\n{product}:")
            print(f"  {data['total_comments']} comments ({spam} spam)")
            print(f"  Sentiment: {pos}% Pos / {neg}% Neg")
            print(f"  Top: {top['aspect']} ({top['pct']}%)")


if __name__ == "__main__":
    main()
