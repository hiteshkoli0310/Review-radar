"""Run sentiment evaluation comparing all approaches including fine-tuned DistilBERT."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewradar.evaluation.sentiment_evaluation import (
    DistilBertScorer,
    RobertaScorer,
    RuleBasedScorer,
    VaderScorer,
    evaluate_sentiment,
    build_evaluation_report,
    _parse_ground_truth,
)


def main() -> None:
    annotation_path = Path("data/annotation/manual_review_sample.csv")
    output_dir = Path("data/reports")

    annotated = pd.read_csv(annotation_path)
    print(f"Loaded {len(annotated)} annotated comments")

    ground_truth = annotated["sentiment_label"].apply(_parse_ground_truth)
    texts = annotated["cleaned_comment_text"].fillna("").tolist()

    # Stratified 85/15 split (give DistilBERT more data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, ground_truth.tolist(), test_size=0.15, random_state=42, stratify=ground_truth
    )
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Further split train into train/val for DistilBERT
    train_texts_sub, val_texts, train_labels_sub, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    print(f"DistilBERT train: {len(train_texts_sub)}, val: {len(val_texts)}")

    # Train DistilBERT
    print("\nTraining DistilBERT...")
    distilbert = DistilBertScorer()
    distilbert.train(
        texts=train_texts_sub,
        labels=train_labels_sub,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir="models/distilbert_sentiment",
        num_epochs=10,
        batch_size=16,
        lr=3e-5,
    )
    print("DistilBERT training complete\n")

    test_df = pd.DataFrame({"cleaned_comment_text": test_texts, "sentiment_label": test_labels})

    scorers = {
        "Rule-Based": RuleBasedScorer(),
        "VADER": VaderScorer(),
        "RoBERTa": RobertaScorer(),
        "DistilBERT": distilbert,
    }

    results = evaluate_sentiment(test_df, scorers)

    print(f"\n--- Results on {len(test_df)} held-out test samples ---")
    for name in ["Rule-Based", "VADER", "RoBERTa", "DistilBERT"]:
        print(f"{name:12s} accuracy: {results[name]['accuracy']:.2%}")

    build_evaluation_report(results, output_dir)
    print(f"\nReport saved to {output_dir / 'sentiment_evaluation_report.md'}")


if __name__ == "__main__":
    main()
