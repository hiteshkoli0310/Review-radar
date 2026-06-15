"""Phase 2: Train and evaluate DistilBERT aspect classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewradar.evaluation.aspect_evaluation import (
    DistilBertAspectScorer,
    evaluate_aspect,
    build_aspect_evaluation_report,
)


def main() -> None:
    annotation_path = Path("data/annotation/manual_review_sample.csv")
    output_dir = Path("data/reports")

    df = pd.read_csv(annotation_path)
    print(f"Loaded {len(df)} annotated comments")

    # Filter out empty aspect labels
    df = df[df["aspect_label"].notna() & (df["aspect_label"].str.strip() != "")].copy()
    print(f"After filtering empty aspects: {len(df)}")

    ground_truth = df["aspect_label"].str.strip()
    texts = df["cleaned_comment_text"].fillna("").tolist()

    # Stratified 85/15 split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, ground_truth.tolist(), test_size=0.15, random_state=42, stratify=ground_truth
    )
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Further split train into train/val
    train_texts_sub, val_texts, train_labels_sub, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    print(f"DistilBERT train: {len(train_texts_sub)}, val: {len(val_texts)}")

    # Train
    print("\nTraining DistilBERT aspect classifier...")
    scorer = DistilBertAspectScorer()
    scorer.train(
        texts=train_texts_sub,
        labels=train_labels_sub,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir="models/distilbert_aspect",
        num_epochs=15,
        batch_size=16,
        lr=3e-5,
    )
    print("Training complete\n")

    # Evaluate
    test_df = pd.DataFrame({"cleaned_comment_text": test_texts, "aspect_label": test_labels})
    results = evaluate_aspect(test_df, scorer)
    print(f"Test accuracy: {results['accuracy']:.2%}")
    print(f"Macro F1:      {results['macro_f1']:.2%}")

    build_aspect_evaluation_report(results, output_dir)
    print(f"\nReport saved to {output_dir / 'aspect_classification_report.md'}")


if __name__ == "__main__":
    main()
