"""Aspect classifier with Random Oversampling — train, evaluate, report, compare."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewradar.evaluation.aspect_evaluation import (
    DistilBertAspectScorer,
    ASPECT_LABELS,
    evaluate_aspect,
    _format_report_markdown,
)


# ── Known baseline (from first training run) ──────────────────────────────────
_BASELINE_RESULTS: dict[str, Any] = {
    "accuracy": 0.3556,
    "macro_precision": 0.2240,
    "macro_recall": 0.3462,
    "macro_f1": 0.2478,
    "per_label": {
        "battery": {"label": "Battery", "precision": 0.3333, "recall": 1.0, "f1": 0.5, "support": 1},
        "camera": {"label": "Camera", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "competition": {"label": "Competition", "precision": 0.5, "recall": 0.3333, "f1": 0.4, "support": 6},
        "display": {"label": "Display", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "gaming": {"label": "Gaming", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2},
        "hardware": {"label": "Hardware", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 3},
        "other": {"label": "Other", "precision": 0.4, "recall": 0.1429, "f1": 0.2105, "support": 14},
        "performance": {"label": "Performance", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "price": {"label": "Price", "precision": 0.4, "recall": 0.5, "f1": 0.4444, "support": 4},
        "purchase_intent": {"label": "Purchase Intent", "precision": 0.4, "recall": 0.6667, "f1": 0.5, "support": 3},
        "software": {"label": "Software", "precision": 0.3333, "recall": 1.0, "f1": 0.5, "support": 1},
        "spam": {"label": "Spam", "precision": 0.5455, "recall": 0.8571, "f1": 0.6667, "support": 7},
        "support": {"label": "Support", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
    },
    "confusion_matrix": [
        [1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,2,0,0,0,0,0,0,2,1,1,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,2,0,0,0,0,0,0],
        [1,0,0,0,3,2,2,0,1,0,1,3,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,1,0,0,2,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,2,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,6,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0],
    ],
    "confusion_labels": ASPECT_LABELS,
    "predictions": [],
    "ground_truth": [],
}


# ── Oversampling helpers ──────────────────────────────────────────────────────

def _audit_class_distribution(labels: list[str], name: str) -> dict[str, int]:
    counts = Counter(labels)
    print(f"\n{name} distribution ({len(labels)} total):")
    for label in sorted(ASPECT_LABELS):
        cnt = counts.get(label, 0)
        pct = cnt / len(labels) * 100 if len(labels) else 0
        print(f"  {label:20s} {cnt:3d} ({pct:5.1f}%)")
    return dict(counts)


def _random_oversample(
    texts: list[str],
    labels: list[str],
    target_count: int = 50,
    exclude_classes: set[str] | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str], list[bool]]:
    if exclude_classes is None:
        exclude_classes = set()

    rng = random.Random(seed)
    result_texts: list[str] = []
    result_labels: list[str] = []
    is_oversampled_flags: list[bool] = []

    pairs: dict[str, list[tuple[str, str]]] = {}
    for t, l in zip(texts, labels):
        pairs.setdefault(l, []).append((t, l))

    for label in sorted(ASPECT_LABELS):
        examples = pairs.get(label, [])
        if label in exclude_classes or len(examples) >= target_count:
            for t, l in examples:
                result_texts.append(t)
                result_labels.append(l)
                is_oversampled_flags.append(False)
            continue

        for t, l in examples:
            result_texts.append(t)
            result_labels.append(l)
            is_oversampled_flags.append(False)

        needed = target_count - len(examples)
        sampled = rng.choices(examples, k=needed)
        for t, l in sampled:
            result_texts.append(t)
            result_labels.append(l)
            is_oversampled_flags.append(True)

    return result_texts, result_labels, is_oversampled_flags


# ── Report builders ───────────────────────────────────────────────────────────

def _build_oversampling_report(
    original_counts: dict[str, int],
    oversampled_counts: dict[str, int],
    target_count: int,
    total_before: int,
    total_after: int,
) -> str:
    lines = ["# Oversampling Report\n"]
    lines.append(f"Target count per class: **{target_count}** (max across all classes)\n")
    lines.append("All classes were oversampled to the target count (including Other). "
                 "No class weights are used during training since data is balanced.\n")
    lines.append(f"| Aspect | Original | Oversampled | Ratio |")
    lines.append("|---|---|---|---|")
    for label in sorted(ASPECT_LABELS):
        orig = original_counts.get(label, 0)
        over = oversampled_counts.get(label, 0)
        ratio = f"{over / orig:.2f}x" if orig else "N/A"
        note = " (excluded)" if label == "Other" and over == orig else ""
        lines.append(f"| {label} | {orig} | {over} | {ratio}{note} |")
    lines.append(f"| **Total** | **{total_before}** | **{total_after}** | **{total_after/total_before:.2f}x** |")
    return "\n".join(lines)


def _build_comparison_report(
    baseline: dict[str, Any],
    results: dict[str, Any],
) -> str:
    lines = ["# Aspect Classification: Baseline vs Oversampled\n"]
    lines.append(f"| Metric | Baseline | Oversampled | Δ |")
    lines.append("|---|---|---|---|")
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        b = baseline[metric]
        o = results[metric]
        delta = o - b
        lines.append(
            f"| {metric.replace('_', ' ').title()} "
            f"| {b:.2%} | {o:.2%} | {delta:+.2%} |"
        )

    lines.append("\n## Per-Label Comparison\n")
    lines.append("| Aspect | Baseline F1 | Oversampled F1 | Δ F1 | Baseline Recall | Oversampled Recall | Δ Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in sorted(ASPECT_LABELS):
        key = label.lower().replace(" ", "_")
        b = baseline["per_label"][key]
        o = results["per_label"][key]
        f1_delta = o["f1"] - b["f1"]
        rec_delta = o["recall"] - b["recall"]
        lines.append(
            f"| {label} "
            f"| {b['f1']:.2%} | {o['f1']:.2%} | {f1_delta:+.2%} "
            f"| {b['recall']:.2%} | {o['recall']:.2%} | {rec_delta:+.2%} |"
        )

    cm_base = baseline["confusion_matrix"]
    cm_over = results["confusion_matrix"]
    labels = baseline["confusion_labels"]

    def _top_confusions(cm):
        pairs = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i][j] > 0:
                    pairs.append((labels[i], labels[j], cm[i][j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _worst_recall(results_dict):
        worst = []
        for label in sorted(ASPECT_LABELS):
            key = label.lower().replace(" ", "_")
            rec = results_dict["per_label"][key]["recall"]
            if rec < 0.50:
                b_rec = baseline["per_label"][key]["recall"]
                worst.append((label, b_rec, rec))
        return worst

    lines.append("\n## Top-5 Confused Pairs (Baseline)\n")
    lines.append("| True → Predicted | Count |")
    lines.append("|---|---|")
    for t, p, c in _top_confusions(cm_base)[:5]:
        lines.append(f"| {t} → {p} | {c} |")

    lines.append("\n## Top-5 Confused Pairs (Oversampled)\n")
    lines.append("| True → Predicted | Count |")
    lines.append("|---|---|")
    for t, p, c in _top_confusions(cm_over)[:5]:
        lines.append(f"| {t} → {p} | {c} |")

    lines.append("\n## Classes with Recall < 50%\n")
    lines.append("| Aspect | Baseline Recall | Oversampled Recall | Δ |")
    lines.append("|---|---|---|---|")
    for label, b_rec, o_rec in _worst_recall(results):
        lines.append(f"| {label} | {b_rec:.0%} | {o_rec:.0%} | {o_rec-b_rec:+.0%} |")

    lines.append("")
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    annotation_path = Path("data/annotation/manual_review_sample.csv")
    output_dir = Path("data/reports")
    oversample_dir = Path("data/oversampled")
    oversample_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # ── Step 1: Load and split ──────────────────────────────────────────
    df = pd.read_csv(annotation_path)
    df = df[df["aspect_label"].notna() & (df["aspect_label"].str.strip() != "")].copy()
    print(f"Loaded {len(df)} annotated comments")

    ground_truth = df["aspect_label"].str.strip()
    texts = df["cleaned_comment_text"].fillna("").tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, ground_truth.tolist(), test_size=0.15, random_state=42, stratify=ground_truth
    )
    train_texts_sub, val_texts, train_labels_sub, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    print(f"Split — Train: {len(train_texts_sub)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # ── Step 2: Audit original training distribution ────────────────────
    original_counts = _audit_class_distribution(train_labels_sub, "Training (before oversampling)")

    # ── Step 3: Determine target and oversample ALL classes ────────────
    all_counts = Counter(train_labels_sub)
    target_count = max(all_counts.values())  # 69 (Other)
    print(f"\nTarget count: {target_count} (max across all classes)")

    train_texts_over, train_labels_over, is_over = _random_oversample(
        train_texts_sub, train_labels_sub,
        target_count=target_count,
        exclude_classes=set(),  # oversample ALL classes including Other
        seed=42,
    )

    over_counts = Counter(train_labels_over)
    total_before = len(train_texts_sub)
    total_after = len(train_texts_over)
    n_synth = total_after - total_before
    print(f"Training: {total_before} → {total_after} (+{n_synth} synthetic)")

    # ── Step 4: Save oversampled dataset with flag ──────────────────────
    over_df = pd.DataFrame({
        "cleaned_comment_text": train_texts_over,
        "aspect_label": train_labels_over,
        "is_oversampled": is_over,
    })
    over_path = oversample_dir / "aspect_oversampled_train.csv"
    over_df.to_csv(over_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {over_path}")

    # ── Step 5: Generate oversampling report ────────────────────────────
    over_rep = _build_oversampling_report(
        original_counts, over_counts, target_count,
        total_before, total_after,
    )
    (output_dir / "aspect_oversampling_report.md").write_text(over_rep, encoding="utf-8")
    print("Wrote oversampling report")

    # ── Step 6: Train with oversampled data ─────────────────────────────
    print("\nTraining DistilBERT with oversampled training data (no class weights)...")
    scorer_over = DistilBertAspectScorer()
    scorer_over.train(
        texts=train_texts_over,
        labels=train_labels_over,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir="models/distilbert_aspect_oversampled",
        num_epochs=15,
        batch_size=16,
        lr=3e-5,
        use_class_weights=False,
    )

    # ── Step 7: Evaluate on untouched test set ──────────────────────────
    test_df = pd.DataFrame({"cleaned_comment_text": test_texts, "aspect_label": test_labels})
    results_over = evaluate_aspect(test_df, scorer_over)

    # Save oversampled report to separate file
    oversampled_md = _format_report_markdown(results_over)
    (output_dir / "aspect_classification_oversampled_report.md").write_text(oversampled_md, encoding="utf-8")
    with open(output_dir / "aspect_classification_oversampled.json", "w", encoding="utf-8") as f:
        json.dump(results_over, f, indent=2, default=str)

    print(f"\nOversampled — Test accuracy: {results_over['accuracy']:.2%}, Macro F1: {results_over['macro_f1']:.2%}")

    # ── Step 8: Compare with baseline ────────────────────────────────────
    print(f"Baseline     — Test accuracy: {_BASELINE_RESULTS['accuracy']:.2%}, Macro F1: {_BASELINE_RESULTS['macro_f1']:.2%}")

    comparison = _build_comparison_report(_BASELINE_RESULTS, results_over)
    (output_dir / "aspect_oversampling_comparison.md").write_text(comparison, encoding="utf-8")
    print("Wrote comparison report")

    # ── Step 9: Summary ──────────────────────────────────────────────────
    acc_improvement = results_over['accuracy'] - _BASELINE_RESULTS['accuracy']
    f1_improvement = results_over['macro_f1'] - _BASELINE_RESULTS['macro_f1']
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Accuracy:  {_BASELINE_RESULTS['accuracy']:.2%} → {results_over['accuracy']:.2%} ({acc_improvement:+.2%})")
    print(f"  Macro F1:  {_BASELINE_RESULTS['macro_f1']:.2%} → {results_over['macro_f1']:.2%} ({f1_improvement:+.2%})")
    print(f"  Target:    {target_count} samples/class (excluding Other)")
    print(f"  Train:     {total_before} → {total_after} (+{n_synth} synthetic)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
