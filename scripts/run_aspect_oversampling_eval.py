"""Evaluate saved oversampled model and generate comparison reports only."""

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
    _format_report_markdown,
)

# ── Baseline (from first training run on 2026-06-13) ─────────────────────────
BASELINE: dict = {
    "accuracy": 0.3556, "macro_precision": 0.2240, "macro_recall": 0.3462, "macro_f1": 0.2478,
    "per_label": {
        "battery":         {"label":"Battery","precision":0.3333,"recall":1.0,"f1":0.5,"support":1},
        "camera":          {"label":"Camera","precision":0.0,"recall":0.0,"f1":0.0,"support":1},
        "competition":     {"label":"Competition","precision":0.5,"recall":0.3333,"f1":0.4,"support":6},
        "display":         {"label":"Display","precision":0.0,"recall":0.0,"f1":0.0,"support":1},
        "gaming":          {"label":"Gaming","precision":0.0,"recall":0.0,"f1":0.0,"support":2},
        "hardware":        {"label":"Hardware","precision":0.0,"recall":0.0,"f1":0.0,"support":3},
        "other":           {"label":"Other","precision":0.4,"recall":0.1429,"f1":0.2105,"support":14},
        "performance":     {"label":"Performance","precision":0.0,"recall":0.0,"f1":0.0,"support":1},
        "price":           {"label":"Price","precision":0.4,"recall":0.5,"f1":0.4444,"support":4},
        "purchase_intent": {"label":"Purchase Intent","precision":0.4,"recall":0.6667,"f1":0.5,"support":3},
        "software":        {"label":"Software","precision":0.3333,"recall":1.0,"f1":0.5,"support":1},
        "spam":            {"label":"Spam","precision":0.5455,"recall":0.8571,"f1":0.6667,"support":7},
        "support":         {"label":"Support","precision":0.0,"recall":0.0,"f1":0.0,"support":1},
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
}


def build_comparison(baseline: dict, results: dict) -> str:
    lines = ["# Aspect Classification: Baseline vs Oversampled\n"]
    lines.append("| Metric | Baseline | Oversampled | Δ |")
    lines.append("|---|---|---|---|")
    for m in ["accuracy","macro_precision","macro_recall","macro_f1"]:
        b, o = baseline[m], results[m]
        lines.append(f"| {m.replace('_',' ').title()} | {b:.2%} | {o:.2%} | {o-b:+.2%} |")

    lines.append("\n## Per-Label Comparison\n")
    lines.append("| Aspect | Baseline F1 | Oversampled F1 | Δ F1 | Baseline Recall | Oversampled Recall | Δ Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for lab in sorted(ASPECT_LABELS):
        k = lab.lower().replace(" ","_")
        b, o = baseline["per_label"][k], results["per_label"][k]
        lines.append(f"| {lab} | {b['f1']:.2%} | {o['f1']:.2%} | {o['f1']-b['f1']:+.2%} | {b['recall']:.2%} | {o['recall']:.2%} | {o['recall']-b['recall']:+.2%} |")

    cm_b, cm_o = baseline["confusion_matrix"], results["confusion_matrix"]
    def top_conf(cm):
        pairs = []
        for i in range(len(ASPECT_LABELS)):
            for j in range(len(ASPECT_LABELS)):
                if i != j and cm[i][j] > 0:
                    pairs.append((ASPECT_LABELS[i], ASPECT_LABELS[j], cm[i][j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:5]

    lines.append("\n## Top-5 Confused Pairs (Baseline)\n")
    lines.append("| True → Predicted | Count |\n|---|---|")
    for t, p, c in top_conf(cm_b):
        lines.append(f"| {t} → {p} | {c} |")

    lines.append("\n## Top-5 Confused Pairs (Oversampled)\n")
    lines.append("| True → Predicted | Count |\n|---|---|")
    for t, p, c in top_conf(cm_o):
        lines.append(f"| {t} → {p} | {c} |")

    lines.append("\n## Classes with Recall < 50% (Oversampled)\n")
    lines.append("| Aspect | Baseline Recall | Oversampled Recall | Δ |")
    lines.append("|---|---|---|---|")
    for lab in sorted(ASPECT_LABELS):
        k = lab.lower().replace(" ","_")
        b_r = baseline["per_label"][k]["recall"]
        o_r = results["per_label"][k]["recall"]
        if o_r < 0.50:
            lines.append(f"| {lab} | {b_r:.0%} | {o_r:.0%} | {o_r-b_r:+.0%} |")
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("data/reports")
    df = pd.read_csv("data/annotation/manual_review_sample.csv")
    df = df[df["aspect_label"].notna() & (df["aspect_label"].str.strip() != "")].copy()

    gt = df["aspect_label"].str.strip()
    texts = df["cleaned_comment_text"].fillna("").tolist()

    _, test_texts, _, test_labels = train_test_split(
        texts, gt.tolist(), test_size=0.15, random_state=42, stratify=gt
    )
    # same 45 test samples as baseline

    scorer = DistilBertAspectScorer(model_path="models/distilbert_aspect_oversampled")
    test_df = pd.DataFrame({"cleaned_comment_text": test_texts, "aspect_label": test_labels})
    results = evaluate_aspect(test_df, scorer)

    print(f"Oversampled model — Accuracy: {results['accuracy']:.2%}, Macro F1: {results['macro_f1']:.2%}")
    print(f"Baseline         — Accuracy: {BASELINE['accuracy']:.2%}, Macro F1: {BASELINE['macro_f1']:.2%}")
    print(f"Δ                — Accuracy: {results['accuracy']-BASELINE['accuracy']:+.2%}, Macro F1: {results['macro_f1']-BASELINE['macro_f1']:+.2%}")

    # save reports
    (out_dir / "aspect_classification_oversampled_report.md").write_text(
        _format_report_markdown(results), encoding="utf-8")
    (out_dir / "aspect_classification_oversampled.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "aspect_oversampling_comparison.md").write_text(
        build_comparison(BASELINE, results), encoding="utf-8")

    print("Reports saved to data/reports/")


if __name__ == "__main__":
    main()
