"""Pseudo-label with "Other" excluded — fix bias, retrain, evaluate."""

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


_BASELINE: dict = {
    "accuracy": 0.4667, "macro_f1": 0.2174,
    "per_label": {
        "battery":{"label":"Battery","recall":1.0,"f1":0.4,"support":1},
        "camera":{"label":"Camera","recall":0.0,"f1":0.0,"support":1},
        "competition":{"label":"Competition","recall":0.1667,"f1":0.25,"support":6},
        "display":{"label":"Display","recall":0.0,"f1":0.0,"support":1},
        "gaming":{"label":"Gaming","recall":0.0,"f1":0.0,"support":2},
        "hardware":{"label":"Hardware","recall":0.3333,"f1":0.5,"support":3},
        "other":{"label":"Other","recall":0.8571,"f1":0.5854,"support":14},
        "performance":{"label":"Performance","recall":0.0,"f1":0.0,"support":1},
        "price":{"label":"Price","recall":1.0,"f1":0.7273,"support":4},
        "purchase_intent":{"label":"Purchase Intent","recall":0.0,"f1":0.0,"support":3},
        "software":{"label":"Software","recall":0.0,"f1":0.0,"support":1},
        "spam":{"label":"Spam","recall":0.2857,"f1":0.3636,"support":7},
        "support":{"label":"Support","recall":0.0,"f1":0.0,"support":1},
    },
}

_CONF_THRESHOLD_NON_OTHER = 0.95


def main() -> None:
    out_dir = Path("data/reports/pseudolabel")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and split ──────────────────────────────────────────────
    ann = pd.read_csv("data/annotation/manual_review_sample.csv")
    ann = ann[ann["aspect_label"].notna() & (ann["aspect_label"].str.strip() != "")].copy()
    gt = ann["aspect_label"].str.strip().tolist()
    texts = ann["cleaned_comment_text"].fillna("").tolist()
    ann_ids = set(ann["comment_id"])

    train_t, test_t, train_l, test_l = train_test_split(
        texts, gt, test_size=0.15, random_state=42, stratify=gt,
    )
    train_t_seed, val_t, train_l_seed, val_l = train_test_split(
        train_t, train_l, test_size=0.15, random_state=42, stratify=train_l,
    )
    test_df = pd.DataFrame({"cleaned_comment_text": test_t, "aspect_label": test_l})
    print(f"Seed: {len(train_t_seed)} train, {len(val_t)} val, {len(test_t)} test")

    # ── Load already-computed pseudo-labels ─────────────────────────
    pl_all = pd.read_csv(out_dir / "pseudolabeled_all.csv")

    # Filter: exclude Other, apply confidence threshold
    pl_filtered = pl_all[
        (pl_all["pseudo_aspect"] != "Other") &
        (pl_all["confidence"] >= _CONF_THRESHOLD_NON_OTHER)
    ].copy()

    print(f"All pseudo-labels: {len(pl_all)}")
    print(f"Non-Other >= {_CONF_THRESHOLD_NON_OTHER}: {len(pl_filtered)}")

    for label in sorted(ASPECT_LABELS):
        cnt = len(pl_filtered[pl_filtered["pseudo_aspect"] == label])
        if cnt:
            print(f"  {label}: {cnt}")

    # ── Merge seed + pseudo ─────────────────────────────────────────
    seed_df = pd.DataFrame({
        "cleaned_comment_text": train_t_seed,
        "aspect_label": train_l_seed,
        "source": "seed",
    })
    pseudo_df = pl_filtered[["cleaned_comment_text", "pseudo_aspect"]].rename(
        columns={"pseudo_aspect": "aspect_label"}
    ).copy()
    pseudo_df["source"] = "pseudolabel"

    merged = pd.concat([seed_df, pseudo_df], ignore_index=True)
    n_pseudo = len(pseudo_df)
    print(f"\nMerged train: {len(seed_df)} seed + {n_pseudo} pseudo = {len(merged)}")

    merged.to_csv(out_dir / "aspect_training_no_other_pseudo.csv", index=False, encoding="utf-8-sig")

    # ── Retrain ─────────────────────────────────────────────────────
    print("\nFine-tuning from oversampled checkpoint (strict threshold, no weighting)...")
    scorer = DistilBertAspectScorer()
    scorer.train(
        texts=merged["cleaned_comment_text"].tolist(),
        labels=merged["aspect_label"].tolist(),
        val_texts=val_t,
        val_labels=val_l,
        output_dir="models/distilbert_aspect_pseudolabel_v2",
        num_epochs=8,
        batch_size=16,
        lr=2e-5,
        use_class_weights=True,
        from_checkpoint="models/distilbert_aspect_oversampled",
    )

    # ── Evaluate ────────────────────────────────────────────────────
    results = evaluate_aspect(test_df, scorer)
    build_aspect_evaluation_report(results, out_dir)

    print(f"\n{'='*60}")
    print(f"Pseudo-label v2 — Acc: {results['accuracy']:.2%}, mF1: {results['macro_f1']:.2%}")
    print(f"Oversampled (base) — Acc: {_BASELINE['accuracy']:.2%}, mF1: {_BASELINE['macro_f1']:.2%}")
    print(f"Δ Acc: {results['accuracy']-_BASELINE['accuracy']:+.2%}")
    print(f"{'='*60}")

    # ── Report ──────────────────────────────────────────────────────
    lines = ["# Pseudo-Label v2: No Other, Threshold >=0.85\n"]
    lines.append(f"Seed: {len(seed_df)} | Pseudo: {n_pseudo} | Total: {len(merged)}\n")
    lines.append("| Metric | Oversampled (46.67%) | Pseudo-Label v2 | Δ |")
    lines.append("|---|---|---|---|")
    b_map = {"accuracy":0.4667,"macro_precision":0.2512,"macro_recall":0.2802,"macro_f1":0.2174}
    for m in ["accuracy","macro_precision","macro_recall","macro_f1"]:
        b = b_map[m]; o = results.get(m,0)
        lines.append(f"| {m.replace('_',' ').title()} | {b:.2%} | {o:.2%} | {o-b:+.2%} |")

    lines.append("\n## Per-Class Recall\n")
    lines.append("| Aspect | Baseline Recall | V2 Recall | Δ | Support |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(ASPECT_LABELS):
        k = label.lower().replace(" ","_")
        b_r = _BASELINE["per_label"][k]["recall"]
        o_r = results["per_label"].get(k,{}).get("recall",0)
        s = _BASELINE["per_label"][k]["support"]
        lines.append(f"| {label} | {b_r:.1%} | {o_r:.1%} | {o_r-b_r:+.1%} | {s} |")

    preds = results.get("predictions",[])
    other_pct = sum(1 for p in preds if p=="Other")/max(len(preds),1)*100
    lines.append(f"\n## Other-Proportion\n")
    lines.append(f"Test predictions classified as **Other**: {other_pct:.1f}%\n")

    (out_dir/"pseudolabel_v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir/'pseudolabel_v2_report.md'}")


if __name__ == "__main__":
    main()
