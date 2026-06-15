# Aspect Classification: Baseline vs Oversampled

| Metric | Baseline | Oversampled | Δ |
|---|---|---|---|
| Accuracy | 35.56% | 46.67% | +11.11% |
| Macro Precision | 22.40% | 25.12% | +2.72% |
| Macro Recall | 34.62% | 28.02% | -6.60% |
| Macro F1 | 24.78% | 21.74% | -3.04% |

## Per-Label Comparison

| Aspect | Baseline F1 | Oversampled F1 | Δ F1 | Baseline Recall | Oversampled Recall | Δ Recall |
|---|---|---|---|---|---|---|
| Battery | 50.00% | 40.00% | -10.00% | 100.00% | 100.00% | +0.00% |
| Camera | 0.00% | 0.00% | +0.00% | 0.00% | 0.00% | +0.00% |
| Competition | 40.00% | 25.00% | -15.00% | 33.33% | 16.67% | -16.66% |
| Display | 0.00% | 0.00% | +0.00% | 0.00% | 0.00% | +0.00% |
| Gaming | 0.00% | 0.00% | +0.00% | 0.00% | 0.00% | +0.00% |
| Hardware | 0.00% | 50.00% | +50.00% | 0.00% | 33.33% | +33.33% |
| Other | 21.05% | 58.54% | +37.49% | 14.29% | 85.71% | +71.42% |
| Performance | 0.00% | 0.00% | +0.00% | 0.00% | 0.00% | +0.00% |
| Price | 44.44% | 72.73% | +28.29% | 50.00% | 100.00% | +50.00% |
| Purchase Intent | 50.00% | 0.00% | -50.00% | 66.67% | 0.00% | -66.67% |
| Software | 50.00% | 0.00% | -50.00% | 100.00% | 0.00% | -100.00% |
| Spam | 66.67% | 36.36% | -30.31% | 85.71% | 28.57% | -57.14% |
| Support | 0.00% | 0.00% | +0.00% | 0.00% | 0.00% | +0.00% |

## Top-5 Confused Pairs (Baseline)

| True → Predicted | Count |
|---|---|
| Other → Gaming | 3 |
| Other → Spam | 3 |
| Competition → Purchase Intent | 2 |
| Hardware → Other | 2 |
| Other → Hardware | 2 |

## Top-5 Confused Pairs (Oversampled)

| True → Predicted | Count |
|---|---|
| Spam → Other | 5 |
| Competition → Other | 4 |
| Hardware → Other | 2 |
| Purchase Intent → Other | 2 |
| Camera → Battery | 1 |

## Classes with Recall < 50%

| Aspect | Baseline Recall | Oversampled Recall | Δ |
|---|---|---|---|
| Camera | 0% | 0% | +0% |
| Competition | 33% | 17% | -17% |
| Display | 0% | 0% | +0% |
| Gaming | 0% | 0% | +0% |
| Hardware | 0% | 33% | +33% |
| Performance | 0% | 0% | +0% |
| Purchase Intent | 67% | 0% | -67% |
| Software | 100% | 0% | -100% |
| Spam | 86% | 29% | -57% |
| Support | 0% | 0% | +0% |
