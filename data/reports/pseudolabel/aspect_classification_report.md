# Aspect Classification Report

DistilBERT fine-tuned on **45** held-out test samples.

## Overall Accuracy


| Metric | Value |
|---|---|
| Accuracy | 40.00% |
| Macro Precision | 17.73% |
| Macro Recall | 24.36% |
| Macro F1 | 18.11% |

## Per-Label Breakdown

| Aspect | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Battery ⚠️ | 33.33% | 100.00% | 50.00% | 1 |
| Camera ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Competition ⚠️ | 50.00% | 16.67% | 25.00% | 6 |
| Display ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Gaming ⚠️ | 0.00% | 0.00% | 0.00% | 2 |
| Hardware ⚠️ | 0.00% | 0.00% | 0.00% | 3 |
| Other | 40.00% | 71.43% | 51.28% | 14 |
| Performance ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Price ⚠️ | 57.14% | 100.00% | 72.73% | 4 |
| Purchase Intent ⚠️ | 0.00% | 0.00% | 0.00% | 3 |
| Software ⚠️ | 0.00% | 0.00% | 0.00% | 1 |
| Spam ⚠️ | 50.00% | 28.57% | 36.36% | 7 |
| Support ⚠️ | 0.00% | 0.00% | 0.00% | 1 |

## Confusion Matrix

| GT \ Pred | Battery | Camera | Competition | Display | Gaming | Hardware | Other | Performance | Price | Purchase Intent | Software | Spam | Support |
|---|------|---|---|---|---|---|---|---|---|---|---|---|---|
| **Battery** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Camera** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Competition** | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Display** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Gaming** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Hardware** | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Other** | 0 | 0 | 1 | 0 | 0 | 1 | 10 | 0 | 1 | 1 | 0 | 0 | 0 |
| **Performance** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Price** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| **Purchase Intent** | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Software** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Spam** | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 2 | 0 |
| **Support** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

## Notes

- ⚠️ marks classes with ≤10 test samples — metrics are unreliable.
- Text notes whether class weights were used or oversampling was applied.
- Model: `distilbert-base-uncased` → 13-class classification head.
- Training: 15 epochs, batch_size=16, lr=3e-5, warmup=10%, weight_decay=0.01.