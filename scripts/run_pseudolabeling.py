"""Pseudo-labeling: expand aspect dataset and retrain classifier."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewradar.evaluation.aspect_evaluation import (
    DistilBertAspectScorer,
    ASPECT_LABELS,
    evaluate_aspect,
    build_aspect_evaluation_report,
)


# ── Known baseline (oversampled model, 46.67%) ───────────────────────────────
_BASELINE: dict = {
    "accuracy": 0.4667, "macro_f1": 0.2174,
    "per_label": {
        "battery":         {"label":"Battery","recall":1.0,"f1":0.4,"support":1},
        "camera":          {"label":"Camera","recall":0.0,"f1":0.0,"support":1},
        "competition":     {"label":"Competition","recall":0.1667,"f1":0.25,"support":6},
        "display":         {"label":"Display","recall":0.0,"f1":0.0,"support":1},
        "gaming":          {"label":"Gaming","recall":0.0,"f1":0.0,"support":2},
        "hardware":        {"label":"Hardware","recall":0.3333,"f1":0.5,"support":3},
        "other":           {"label":"Other","recall":0.8571,"f1":0.5854,"support":14},
        "performance":     {"label":"Performance","recall":0.0,"f1":0.0,"support":1},
        "price":           {"label":"Price","recall":1.0,"f1":0.7273,"support":4},
        "purchase_intent": {"label":"Purchase Intent","recall":0.0,"f1":0.0,"support":3},
        "software":        {"label":"Software","recall":0.0,"f1":0.0,"support":1},
        "spam":            {"label":"Spam","recall":0.2857,"f1":0.3636,"support":7},
        "support":         {"label":"Support","recall":0.0,"f1":0.0,"support":1},
    },
}


def main() -> None:
    annotation_path = Path("data/annotation/manual_review_sample.csv")
    master_path = Path("data/exports/reviewradar_master_raw.csv")
    output_dir = Path("data/reports/pseudolabel")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load annotation seeds ───────────────────────────────────
    ann = pd.read_csv(annotation_path)
    ann = ann[ann["aspect_label"].notna() & (ann["aspect_label"].str.strip() != "")].copy()
    print(f"Annotation: {len(ann)} rows")

    gt = ann["aspect_label"].str.strip().tolist()
    texts = ann["cleaned_comment_text"].fillna("").tolist()
    ann_ids = set(ann["comment_id"])

    # Same fixed split as before
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, gt, test_size=0.15, random_state=42, stratify=gt,
    )
    train_texts_sub, val_texts, train_labels_sub, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels,
    )
    print(f"Seed split — Train: {len(train_texts_sub)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # ── Step 2: Load master, filter unlabeled ───────────────────────────
    master = pd.read_csv(master_path)
    unlabeled = master[~master["comment_id"].isin(ann_ids)].copy()
    print(f"Unlabeled: {len(unlabeled)} rows")

    unlabeled_texts = unlabeled["cleaned_comment_text"].fillna("").tolist()
    unlabeled_ids = unlabeled["comment_id"].tolist()

    # ── Step 3: Predict with scores ─────────────────────────────────────
    print("\nRunning pseudo-labeling on unlabeled comments...")
    scorer = DistilBertAspectScorer(model_path="models/distilbert_aspect_oversampled")
    scored = scorer.predict_with_scores(unlabeled_texts)

    # Build results dataframe
    pl_df = unlabeled[["comment_id", "product_query", "cleaned_comment_text"]].copy()
    pl_df["pseudo_aspect"] = [s["aspect"] for s in scored]
    pl_df["confidence"] = [s["confidence"] for s in scored]

    # Store full score vectors for audit
    score_cols = {a: [] for a in ASPECT_LABELS}
    for s in scored:
        for a in ASPECT_LABELS:
            score_cols[a].append(s["scores"][a])
    for a in ASPECT_LABELS:
        pl_df[f"score_{a.lower().replace(' ', '_')}"] = score_cols[a]

    out_path = output_dir / "pseudolabeled_all.csv"
    pl_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved all pseudo-labels: {out_path}")

    # ── Step 4: Filter high-confidence ──────────────────────────────────
    high_conf = pl_df[pl_df["confidence"] >= 0.90].copy()
    high_conf = high_conf.sort_values("confidence", ascending=False)

    review_path = output_dir / "pseudolabeled_highconf_review.csv"
    review_cols = ["comment_id", "product_query", "cleaned_comment_text",
                   "pseudo_aspect", "confidence"] + [f"score_{a.lower().replace(' ', '_')}" for a in ASPECT_LABELS]
    high_conf[review_cols].to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"High-confidence review file: {review_path} ({len(high_conf)} rows)")

    # Distribution stats
    print("\nPseudo-label distribution (all):")
    for label, cnt in sorted(Counter(pl_df["pseudo_aspect"]).items()):
        pct = cnt / len(pl_df) * 100
        print(f"  {label:20s} {cnt:5d} ({pct:5.1f}%)")

    print("\nHigh-confidence (>=90%) distribution:")
    hc_counts = Counter(high_conf["pseudo_aspect"])
    for label in sorted(ASPECT_LABELS):
        cnt = hc_counts.get(label, 0)
        print(f"  {label:20s} {cnt:5d}")

    # ── Step 5: Merge seed + pseudo-labels for retraining ───────────────
    # Prepare seed training texts/labels (216 samples)
    seed_train_df = pd.DataFrame({
        "cleaned_comment_text": train_texts_sub,
        "aspect_label": train_labels_sub,
        "source": "seed",
        "confidence": 1.0,
    })

    # Only use high-confidence pseudo-labels
    pseudo_train = high_conf[["cleaned_comment_text", "pseudo_aspect"]].rename(
        columns={"pseudo_aspect": "aspect_label"}
    ).copy()
    pseudo_train["source"] = "pseudolabel"
    pseudo_train["confidence"] = high_conf["confidence"].values

    merged = pd.concat([seed_train_df, pseudo_train], ignore_index=True)
    print(f"\nMerged training set: {len(seed_train_df)} seed + {len(pseudo_train)} pseudo = {len(merged)}")

    merged_out = output_dir / "aspect_training_merged.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")
    print(f"Saved merged training: {merged_out}")

    # ── Step 6: Retrain ─────────────────────────────────────────────────
    print("\nTraining DistilBERT with merged seed + pseudo-labels...")
    scorer2 = DistilBertAspectScorer()
    scorer2.train(
        texts=merged["cleaned_comment_text"].tolist(),
        labels=merged["aspect_label"].tolist(),
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir="models/distilbert_aspect_pseudolabel",
        num_epochs=15,
        batch_size=16,
        lr=3e-5,
        use_class_weights=True,
    )

    # ── Step 7: Evaluate on same test set ───────────────────────────────
    test_df = pd.DataFrame({"cleaned_comment_text": test_texts, "aspect_label": test_labels})
    results = evaluate_aspect(test_df, scorer2)

    # Save report
    build_aspect_evaluation_report(results, output_dir)
    with open(output_dir / "aspect_classification_pseudolabel.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Step 8: Compare with baseline ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pseudo-label model — Accuracy: {results['accuracy']:.2%}, Macro F1: {results['macro_f1']:.2%}")
    print(f"Oversampled (base) — Accuracy: {_BASELINE['accuracy']:.2%}, Macro F1: {_BASELINE['macro_f1']:.2%}")
    print(f"Δ                  — Accuracy: {results['accuracy']-_BASELINE['accuracy']:+.2%}, Macro F1: {results['macro_f1']-_BASELINE['macro_f1']:+.2%}")
    print(f"{'='*60}")

    # Per-class recall comparison
    lines = ["# Pseudo-Labeling Evaluation Report\n"]
    lines.append(f"Seed annotations: {len(ann)} | Pseudo-labels (≥90% conf): {len(pseudo_train)}\n")
    lines.append(f"| Metric | Oversampled (46.67%) | Pseudo-Label | Δ |")
    lines.append("|---|---|---|---|")
    for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        b = _BASELINE.get(m, 0)
        o = results.get(m, 0)
        if m == "macro_precision":
            b = 0.2512
        elif m == "macro_recall":
            b = 0.2802
        lines.append(f"| {m.replace('_',' ').title()} | {b:.2%} | {o:.2%} | {o-b:+.2%} |")

    lines.append("\n## Per-Class Recall\n")
    lines.append("| Aspect | Baseline Recall | Pseudo-Label Recall | Δ | Support |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(ASPECT_LABELS):
        key = label.lower().replace(" ", "_")
        b_r = _BASELINE["per_label"][key]["recall"]
        o_r = results["per_label"].get(key, {}).get("recall", 0)
        support = _BASELINE["per_label"][key]["support"]
        lines.append(f"| {label} | {b_r:.1%} | {o_r:.1%} | {o_r-b_r:+.1%} | {support} |")

    # Other-proportion analysis
    all_preds = results["predictions"]
    other_pct = sum(1 for p in all_preds if p == "Other") / len(all_preds) * 100
    lines.append(f"\n## Other-Proportion Analysis\n")
    lines.append(f"Proportion of test predictions classified as **Other**: {other_pct:.1f}%")

    (output_dir / "pseudolabel_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {output_dir / 'pseudolabel_evaluation_report.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
