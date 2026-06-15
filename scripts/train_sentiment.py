#!/usr/bin/env python3
"""Train DistilBERT sentiment model with like-weighted samples on 300 annotated rows.

Weights each sample by ``1 + log2(1 + comment_like_count)`` so that widely-agreed
opinions (more likes) contribute more to the loss.  Zero-like comments still
contribute with weight=1.

Cross-validated estimate ~59-61% accuracy on 300 samples.

Usage:
    python scripts/train_sentiment.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from reviewradar.evaluation.sentiment_evaluation import (
    DistilBertScorer,
    RuleBasedScorer,
    VaderScorer,
    RobertaScorer,
    _compute_metrics,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _like_weight(likes: int, cap: int = 10) -> float:
    return min(1.0 + math.log2(1 + likes), cap)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Sentiment Model Training — Like-Weighted Samples")
    logger.info("=" * 60)

    df = pd.read_csv("data/annotation/manual_review_sample.csv")
    df = df[df["cleaned_comment_text"].notna()].copy()
    logger.info("Loaded %d annotated rows", len(df))

    texts = df["cleaned_comment_text"].tolist()
    labels = df["sentiment_label"].tolist()
    likes = df["comment_like_count"].tolist()
    sample_weights = [_like_weight(l) for l in likes]

    logger.info(
        "Like weights: min=%.2f, max=%.2f, median=%.2f",
        min(sample_weights), max(sample_weights),
        sorted(sample_weights)[len(sample_weights) // 2],
    )

    # ── 5-fold cross-validation ─────────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    all_val_truth = []
    all_val_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_t = [texts[i] for i in train_idx]
        train_l = [labels[i] for i in train_idx]
        train_w = [sample_weights[i] for i in train_idx]
        val_t = [texts[i] for i in val_idx]
        val_l = [labels[i] for i in val_idx]

        s = DistilBertScorer()
        s.train(
            texts=train_t, labels=train_l,
            sample_weights=train_w,
            val_texts=val_t, val_labels=val_l,
            output_dir=f"models/distilbert_sentiment_fold{fold}",
            num_epochs=15, batch_size=16, lr=5e-5,
        )
        preds = s.predict_batch(val_t)
        acc = sum(1 for p, t in zip(preds, val_l) if p == t) / len(val_l)
        fold_accs.append(acc)
        all_val_truth.extend(val_l)
        all_val_preds.extend(preds)
        logger.info("Fold %d: val_acc=%.2f%% (n=%d)", fold + 1, acc * 100, len(val_t))

        import shutil
        shutil.rmtree(f"models/distilbert_sentiment_fold{fold}", ignore_errors=True)

    mean_cv = sum(fold_accs) / len(fold_accs)
    std_cv = (sum((a - mean_cv) ** 2 for a in fold_accs) / len(fold_accs)) ** 0.5
    logger.info("Cross-val accuracy: %.2f%% +/- %.2f%%", mean_cv * 100, std_cv * 100)

    cv_metrics = _compute_metrics(all_val_truth, all_val_preds)

    # ── Train final model on ALL 300 ────────────────────────────────────
    logger.info("Training final model on all 300 samples (like-weighted)...")
    final = DistilBertScorer()
    final.train(
        texts=texts, labels=labels,
        sample_weights=sample_weights,
        output_dir="models/distilbert_sentiment",
        num_epochs=15, batch_size=16, lr=5e-5,
    )

    # ── Build comparison data ───────────────────────────────────────────
    scorers = {
        "Rule-Based": RuleBasedScorer(),
        "VADER": VaderScorer(),
        "RoBERTa": RobertaScorer(),
    }
    results: dict = {}
    for name, scorer in scorers.items():
        preds = scorer.predict_batch(texts)
        results[name] = _compute_metrics(labels, preds)

    results["DistilBERT"] = cv_metrics
    results["DistilBERT"]["training_details"] = {
        "strategy": "Full fine-tune DistilBERT, like-weighted samples (1+log2(1+likes))",
        "best_lr": "5e-5",
        "epochs": 15,
        "batch_size": 16,
        "weight_decay": 0.1,
        "sample_weight_fn": "1 + log2(1 + comment_like_count), capped at 10",
        "cross_val_accuracy_5fold": f"{mean_cv:.2%}",
        "cross_val_std_5fold": f"{std_cv:.2%}",
    }
    results["ground_truth"] = labels

    # ── Write reports ───────────────────────────────────────────────────
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "sentiment_evaluation.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    def _fmt(pct: float) -> str:
        return f"{pct:.2%}"

    md = [f"# Sentiment Evaluation Report\n"]
    md.append(f"Comparison of sentiment approaches against **{len(labels)}** human-annotated comments.\n")
    md.append("**DistilBERT:** like-weighted samples, 5-fold CV estimate.\n\n")
    md.append("## Overall Accuracy\n\n| Approach | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n|---|---|---|---|---|\n")
    scores = []
    for name in results:
        if name == "ground_truth":
            continue
        m = results[name]
        md.append(f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['macro_precision'])} | {_fmt(m['macro_recall'])} | {_fmt(m['macro_f1'])} |\n")
        scores.append((name, m["accuracy"]))
    md.append("")
    best = max(scores, key=lambda x: x[1])
    md.append(f"**Best approach:** {best[0]} ({_fmt(best[1])} accuracy)\n")

    md.append("## Per-Label Breakdown\n")
    for name in results:
        if name == "ground_truth":
            continue
        md.append(f"### {name}\n\n| Label | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
        for label in ("positive", "neutral", "negative"):
            p = results[name]["per_label"].get(label, {})
            md.append(f"| {label.title()} | {_fmt(p.get('precision', 0))} | {_fmt(p.get('recall', 0))} | {_fmt(p.get('f1', 0))} | {p.get('support', 0)} |\n")
        md.append("")

    md.append("## Confusion Matrices\n")
    for name in results:
        if name == "ground_truth":
            continue
        m = results[name]
        cls = m["confusion_labels"]
        cm = m["confusion_matrix"]
        md.append(f"### {name}\n\n| GT \\ Pred | " + " | ".join(cls) + " |\n|---|" + "---|" * len(cls) + "\n")
        for i, rl in enumerate(cls):
            md.append(f"| **{rl}** | " + " | ".join(str(cm[i][j]) for j in range(len(cls))) + " |\n")
        md.append("")

    (report_dir / "sentiment_evaluation_report.md").write_text("".join(md), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("Done — model: models/distilbert_sentiment/")
    logger.info("Reports: data/reports/sentiment_evaluation.*")
    logger.info("Cross-val: %.2f%% +/- %.2f%%", mean_cv * 100, std_cv * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
