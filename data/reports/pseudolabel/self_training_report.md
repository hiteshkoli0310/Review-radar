# Self-Training Evaluation Report (0 rounds)

Seed: 216 | Pseudo-labels added: 0 | Total train: 216

| Metric | Oversampled (46.67%) | Self-Train | Delta |
|---|---|---|---|
| Accuracy | 46.67% | 33.33% | -13.34% |
| Macro Precision | 25.12% | 18.50% | -6.62% |
| Macro Recall | 28.02% | 28.21% | +0.19% |
| Macro F1 | 21.74% | 19.14% | -2.60% |

## Per-Class Recall

| Aspect | Baseline Recall | Self-Train Recall | Delta | Support |
|---|---|---|---|---|
| Battery | 100.0% | 100.0% | +0.0% | 1 |
| Camera | 0.0% | 0.0% | +0.0% | 1 |
| Competition | 16.7% | 0.0% | -16.7% | 6 |
| Display | 0.0% | 0.0% | +0.0% | 1 |
| Gaming | 0.0% | 50.0% | +50.0% | 2 |
| Hardware | 33.3% | 0.0% | -33.3% | 3 |
| Other | 85.7% | 28.6% | -57.1% | 14 |
| Performance | 0.0% | 0.0% | +0.0% | 1 |
| Price | 100.0% | 50.0% | -50.0% | 4 |
| Purchase Intent | 0.0% | 66.7% | +66.7% | 3 |
| Software | 0.0% | 0.0% | +0.0% | 1 |
| Spam | 28.6% | 71.4% | +42.9% | 7 |
| Support | 0.0% | 0.0% | +0.0% | 1 |

## Other-Proportion

Predictions classified as **Other**: 11.1%

## Training Progression

| Round | Train Size | New Pseudo | Accuracy | Macro F1 |
|---|---|---|---|---|
