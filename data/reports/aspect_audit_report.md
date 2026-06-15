# Aspect-Label Audit Report

Generated from `manual_review_sample.csv` — **300** annotated rows across 3 products.

## Label Distribution

Chart: `charts/aspect_distribution.png`

| Aspect | Count | % of Total | Imbalance Ratio (vs Spam) |
|---|---|---|---|
| Gaming ⚠️ low | 14 | 4.7% | 6.86x |
| Display ⚠️ rare | 7 | 2.3% | 13.71x |
| Battery ⚠️ rare | 8 | 2.7% | 12.0x |
| Camera ⚠️ rare | 4 | 1.3% | 24.0x |
| Performance ⚠️ rare | 5 | 1.7% | 19.2x |
| Price | 24 | 8.0% | 4.0x |
| Competition | 39 | 13.0% | 2.46x |
| Purchase Intent | 21 | 7.0% | 4.57x |
| Software ⚠️ rare | 9 | 3.0% | 10.67x |
| Hardware ⚠️ low | 19 | 6.3% | 5.05x |
| Spam | 47 | 15.7% | 2.04x |
| Support ⚠️ rare | 7 | 2.3% | 13.71x |
| Other | 96 | 32.0% | 1.0x |

## Class Summary

- **Rare** (≤10 samples): Software, Battery, Support, Display, Performance, Camera — 6 classes
- **Low** (11-20 samples): Hardware, Gaming — 2 classes
- **Adequate** (>20 samples): Other, Spam, Competition, Price, Purchase Intent — 5 classes

Dominant class **Spam** (96) is 24x larger than rarest class **Camera** (4).

## Per-Product Aspect Coverage

Chart: `charts/aspect_per_product_heatmap.png`

| Aspect | Iphone 17 | Nintendo Switch | Steam Deck |
|---|---|---|---|
| Gaming | 0 | 9 | 5 |
| Display | 2 | 3 | 2 |
| Battery | 5 | 1 | 2 |
| Camera | 4 | 0 | 0 |
| Performance | 2 | 1 | 2 |
| Price | 3 | 8 | 13 |
| Competition | 15 | 7 | 17 |
| Purchase Intent | 8 | 4 | 9 |
| Software | 5 | 1 | 3 |
| Hardware | 5 | 5 | 9 |
| Spam | 16 | 18 | 13 |
| Support | 4 | 1 | 2 |
| Other | 31 | 42 | 23 |

## Annotation Quality Signals

| Aspect | Count | With Notes | Notes % | Translated |
|---|---|---|---|---|
| Battery | 8 | 1 | 12.5% | 0 |
| Camera | 4 | 1 | 25.0% | 0 |
| Competition | 39 | 16 | 41.0% | 3 |
| Display | 7 | 1 | 14.3% | 0 |
| Gaming | 14 | 6 | 42.9% | 1 |
| Hardware | 19 | 9 | 47.4% | 2 |
| Other | 96 | 70 | 72.9% | 4 |
| Performance | 5 | 2 | 40.0% | 1 |
| Price | 24 | 5 | 20.8% | 1 |
| Purchase Intent | 21 | 6 | 28.6% | 4 |
| Software | 9 | 2 | 22.2% | 1 |
| Spam | 47 | 27 | 57.4% | 8 |
| Support | 7 | 2 | 28.6% | 0 |

## Recommendations

1. **Rare classes (Camera, Performance, Display, Support, Battery)**:
   - These have ≤10 samples and will be unreliable for training.
   - Option A: Merge into `Other` to reduce label noise.
   - Option B: Keep all 13 with aggressive class weighting (inverse frequency).
   - **Recommended: Option B** — use class weights but flag per-class metrics as unreliable for rare classes.

2. **Spam dominance (47 samples)** :
   - Spam is a valid consumer-intent signal but should be separated from product aspects.
   - For insight generation, Spam comments can be filtered out or reported separately.

3. **Per-product coverage gaps**:
   - Camera: only Iphone 17 has it (makes sense).
   - Gaming: only Steam Deck and Nintendo Switch (makes sense).
   - These product-specific aspects are expected and fine.

4. **Notes ratio**:
   - Some aspects have high notes ratios — review those for label quality issues.
