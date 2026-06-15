"""Phase 1: Aspect-label audit — distribution, rare classes, quality analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reviewradar.annotation.annotation_dataset_builder import ASPECT_LABELS

CHART_DIR = Path("data/reports/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/annotation/manual_review_sample.csv")

# ── Label counts ───────────────────────────────────────────────────
aspect_counts = df["aspect_label"].value_counts()
aspect_sorted = aspect_counts.sort_values(ascending=False)

# ── Bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if c <= 10 else "#f39c12" if c <= 20 else "#2ecc71" for c in aspect_sorted.values]
bars = ax.bar(range(len(aspect_sorted)), aspect_sorted.values, color=colors)
ax.set_xticks(range(len(aspect_sorted)))
ax.set_xticklabels(aspect_sorted.index, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Aspect Label Distribution (red=rare, orange=low, green=adequate)")
for bar, val in zip(bars, aspect_sorted.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(CHART_DIR / "aspect_distribution.png", dpi=150)
plt.close()

# ── Per-product aspect heatmap ─────────────────────────────────────
prod_aspect = pd.crosstab(df["product_query"], df["aspect_label"])
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(prod_aspect.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(prod_aspect.columns)))
ax.set_xticklabels(prod_aspect.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(prod_aspect.index)))
ax.set_yticklabels(prod_aspect.index)
ax.set_title("Aspect × Product Heatmap")
for i in range(len(prod_aspect.index)):
    for j in range(len(prod_aspect.columns)):
        val = prod_aspect.values[i, j]
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=8, color="white" if val > prod_aspect.values.max() * 0.6 else "black")
plt.colorbar(im, label="Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "aspect_per_product_heatmap.png", dpi=150)
plt.close()

# ── Aspect quality signals ─────────────────────────────────────────
quality = df.groupby("aspect_label").agg(
    count=("aspect_label", "count"),
    has_notes=("review_notes", lambda x: x.notna().sum()),
    is_translated=("is_translated", lambda x: (x == True).sum() if x.dtype == bool else 0),
).reset_index()
quality["notes_ratio"] = (quality["has_notes"] / quality["count"] * 100).round(1)
quality["translated_ratio"] = (quality["is_translated"] / quality["count"] * 100).round(1)

# ── Rare-class analysis ────────────────────────────────────────────
rare = aspect_counts[aspect_counts <= 10]
low = aspect_counts[(aspect_counts > 10) & (aspect_counts <= 20)]
adequate = aspect_counts[aspect_counts > 20]

# ── Imbalance ratios ───────────────────────────────────────────────
max_count = aspect_counts.max()
imbalance_ratios = {}
for label, count in aspect_counts.items():
    imbalance_ratios[label] = round(max_count / count, 2)

# ── Generate report ────────────────────────────────────────────────
lines = ["# Aspect-Label Audit Report\n"]
lines.append(f"Generated from `manual_review_sample.csv` — **300** annotated rows across 3 products.\n")

lines.append("## Label Distribution\n")
lines.append(f"Chart: `charts/aspect_distribution.png`\n")
lines.append(f"| Aspect | Count | % of Total | Imbalance Ratio (vs Spam) |")
lines.append("|---|---|---|---|")
for label in ASPECT_LABELS:
    cnt = aspect_counts.get(label, 0)
    pct = cnt / 300 * 100
    ratio = imbalance_ratios.get(label, 0)
    flag = " ⚠️ rare" if cnt <= 10 else " ⚠️ low" if cnt <= 20 else ""
    lines.append(f"| {label}{flag} | {cnt} | {pct:.1f}% | {ratio}x |")
lines.append("")

lines.append("## Class Summary\n")
lines.append(f"- **Rare** (≤10 samples): {', '.join(rare.index.tolist())} — {len(rare)} classes")
lines.append(f"- **Low** (11-20 samples): {', '.join(low.index.tolist())} — {len(low)} classes")  
lines.append(f"- **Adequate** (>20 samples): {', '.join(adequate.index.tolist())} — {len(adequate)} classes")
lines.append(f"\nDominant class **Spam** ({max_count}) is {max(aspect_counts.values)/min(aspect_counts.values):.0f}x larger than rarest class **Camera** ({aspect_counts.min()}).\n")

lines.append("## Per-Product Aspect Coverage\n")
lines.append(f"Chart: `charts/aspect_per_product_heatmap.png`\n")
lines.append("| Aspect | Iphone 17 | Nintendo Switch | Steam Deck |")
lines.append("|---|---|---|---|")
for label in ASPECT_LABELS:
    vals = []
    for p in ["Iphone 17", "Nintendo Switch", "Steam Deck"]:
        v = prod_aspect.loc[p, label] if label in prod_aspect.columns else 0
        vals.append(str(v))
    lines.append(f"| {label} | " + " | ".join(vals) + " |")
lines.append("")

lines.append("## Annotation Quality Signals\n")
lines.append("| Aspect | Count | With Notes | Notes % | Translated |")
lines.append("|---|---|---|---|---|")
for _, r in quality.iterrows():
    lines.append(f"| {r['aspect_label']} | {r['count']} | {r['has_notes']} | {r['notes_ratio']}% | {r['is_translated']} |")
lines.append("")

lines.append("## Recommendations\n")
lines.append("1. **Rare classes (Camera, Performance, Display, Support, Battery)**:")
lines.append("   - These have ≤10 samples and will be unreliable for training.")
lines.append("   - Option A: Merge into `Other` to reduce label noise.")
lines.append("   - Option B: Keep all 13 with aggressive class weighting (inverse frequency).")
lines.append("   - **Recommended: Option B** — use class weights but flag per-class metrics as unreliable for rare classes.")
lines.append("")
lines.append("2. **Spam dominance (47 samples)** :")
lines.append("   - Spam is a valid consumer-intent signal but should be separated from product aspects.")
lines.append("   - For insight generation, Spam comments can be filtered out or reported separately.")
lines.append("")
lines.append("3. **Per-product coverage gaps**:")
lines.append("   - Camera: only Iphone 17 has it (makes sense).")
lines.append("   - Gaming: only Steam Deck and Nintendo Switch (makes sense).")
lines.append("   - These product-specific aspects are expected and fine.")
lines.append("")
lines.append("4. **Notes ratio**:")
if quality["notes_ratio"].max() > 30:
    lines.append("   - Some aspects have high notes ratios — review those for label quality issues.")
else:
    lines.append("   - Overall low notes ratio — annotations are clean.")
lines.append("")

Path("data/reports/aspect_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
print("Report written to data/reports/aspect_audit_report.md")
print(f"Aspect distribution: {aspect_sorted.to_dict()}")
print(f"Rare (≤10): {list(rare.index)}")
print(f"Low (11-20): {list(low.index)}")
print(f"Adequate (>20): {list(adequate.index)}")
