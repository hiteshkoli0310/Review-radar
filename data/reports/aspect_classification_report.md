# Aspect Classification Report

DistilBERT fine-tuned on **45** held-out test samples.

## Overall Accuracy


| Metric | Value |
|---|---|
| Accuracy | 44.44% |
| Macro Precision | 30.86% |
| Macro Recall | 32.33% |
| Macro F1 | 27.36% |

## Per-Label Breakdown

| Aspect | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Battery ⚠️ | 50.00% | 100.00% | 66.67% | 1 |
| Camera ⚠️ | 50.00% | 100.00% | 66.67% | 1 |
| Competition ⚠️ | 100.00% | 16.67% | 28.57% | 6 |
| Display ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Gaming ⚠️ | 0.00% | 0.00% | 0.00% | 2 |
| Hardware ⚠️ | 0.00% | 0.00% | 0.00% | 3 |
| Other | 34.48% | 71.43% | 46.51% | 14 |
| Performance ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Price ⚠️ | 100.00% | 75.00% | 85.71% | 4 |
| Purchase Intent ⚠️ | 0.00% | 0.00% | 0.00% | 3 |
| Software ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Spam ⚠️ | 66.67% | 57.14% | 61.54% | 7 |
| Support ⚠️ | 0.00% | 0.00% | 0.00% | 1 |

## Confusion Matrix

| GT \ Pred | Battery | Camera | Competition | Display | Gaming | Hardware | Other | Performance | Price | Purchase Intent | Software | Spam | Support |
|---|------|---|---|---|---|---|---|---|---|---|---|---|---|
| **Battery** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Camera** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Competition** | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Display** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Gaming** | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Hardware** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Other** | 1 | 0 | 0 | 0 | 0 | 2 | 10 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Performance** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Price** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| **Purchase Intent** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Software** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Spam** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 4 | 0 |
| **Support** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Notes

- ⚠️ marks classes with ≤10 test samples — metrics are unreliable.
- Class weights were used (inverse frequency) to handle imbalance.
- Model: `distilbert-base-uncased` → 13-class classification head.
- Training: 15 epochs, batch_size=16, lr=3e-5, warmup=10%, weight_decay=0.01.