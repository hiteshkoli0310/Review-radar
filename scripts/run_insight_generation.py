"""Phase 3: Generate consumer insight reports from trained models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer as SentimentScorer
from reviewradar.evaluation.aspect_evaluation import DistilBertAspectScorer as AspectScorer
from reviewradar.evaluation.insight_generator import generate_product_insights


def main() -> None:
    master_path = Path("data/exports/reviewradar_master_raw.csv")
    output_dir = Path("data/reports/insights")

    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} rows from master dataset")

    print("Loading sentiment model...")
    sentiment_scorer = SentimentScorer(model_path="models/distilbert_sentiment")
    print("  Done")

    print("Loading aspect model...")
    aspect_scorer = AspectScorer(model_path="models/distilbert_aspect")
    print("  Done")

    print("\nGenerating insights...")
    insights = generate_product_insights(
        df, sentiment_scorer, aspect_scorer, output_dir=str(output_dir)
    )
    print(f"\nInsights generated for {len(insights)} products")
    print(f"Reports in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
