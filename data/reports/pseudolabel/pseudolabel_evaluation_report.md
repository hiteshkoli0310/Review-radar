# Pseudo-Labeling Evaluation Report

Seed annotations: 300 | Pseudo-labels (≥90% conf): 568

| Metric | Oversampled (46.67%) | Pseudo-Label | Δ |
|---|---|---|---|
| Accuracy | 46.67% | 35.56% | -11.11% |
| Macro Precision | 25.12% | 17.18% | -7.94% |
| Macro Recall | 28.02% | 21.34% | -6.68% |
| Macro F1 | 21.74% | 17.15% | -4.59% |

## Per-Class Recall

| Aspect | Baseline Recall | Pseudo-Label Recall | Δ | Support |
|---|---|---|---|---|
| Battery | 100.0% | 100.0% | +0.0% | 1 |
| Camera | 0.0% | 0.0% | +0.0% | 1 |
| Competition | 16.7% | 16.7% | -0.0% | 6 |
| Display | 0.0% | 0.0% | +0.0% | 1 |
| Gaming | 0.0% | 0.0% | +0.0% | 2 |
| Hardware | 33.3% | 0.0% | -33.3% | 3 |
| Other | 85.7% | 71.4% | -14.3% | 14 |
| Performance | 0.0% | 0.0% | +0.0% | 1 |
| Price | 100.0% | 75.0% | -25.0% | 4 |
| Purchase Intent | 0.0% | 0.0% | +0.0% | 3 |
| Software | 0.0% | 0.0% | +0.0% | 1 |
| Spam | 28.6% | 14.3% | -14.3% | 7 |
| Support | 0.0% | 0.0% | +0.0% | 1 |

## Other-Proportion Analysis

Proportion of test predictions classified as **Other**: 55.6%