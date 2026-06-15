# Sentiment Evaluation Report
Comparison of sentiment approaches against **300** human-annotated comments.
**DistilBERT:** like-weighted samples, 5-fold CV estimate.

## Overall Accuracy

| Approach | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Rule-Based | 48.33% | 53.38% | 45.12% | 43.69% |
| VADER | 46.00% | 46.67% | 45.56% | 43.87% |
| RoBERTa | 63.00% | 63.34% | 61.78% | 62.12% |
| DistilBERT | 60.33% | 58.86% | 59.03% | 58.87% |
**Best approach:** RoBERTa (63.00% accuracy)
## Per-Label Breakdown
### Rule-Based

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 40.62% | 31.71% | 35.62% | 82 |
| Neutral | 46.80% | 79.17% | 58.82% | 120 |
| Negative | 72.73% | 24.49% | 36.64% | 98 |
### VADER

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 36.59% | 54.88% | 43.90% | 82 |
| Neutral | 53.44% | 58.33% | 55.78% | 120 |
| Negative | 50.00% | 23.47% | 31.94% | 98 |
### RoBERTa

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 56.25% | 54.88% | 55.56% | 82 |
| Neutral | 61.97% | 73.33% | 67.18% | 120 |
| Negative | 71.79% | 57.14% | 63.64% | 98 |
### DistilBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 45.57% | 43.90% | 44.72% | 82 |
| Neutral | 69.91% | 65.83% | 67.81% | 120 |
| Negative | 61.11% | 67.35% | 64.08% | 98 |
## Confusion Matrices
### Rule-Based

| GT \ Pred | Positive | Neutral | Negative |
|---|---|---|---|
| **Positive** | 26 | 52 | 4 |
| **Neutral** | 20 | 95 | 5 |
| **Negative** | 18 | 56 | 24 |
### VADER

| GT \ Pred | Positive | Neutral | Negative |
|---|---|---|---|
| **Positive** | 45 | 27 | 10 |
| **Neutral** | 37 | 70 | 13 |
| **Negative** | 41 | 34 | 23 |
### RoBERTa

| GT \ Pred | Positive | Neutral | Negative |
|---|---|---|---|
| **Positive** | 45 | 26 | 11 |
| **Neutral** | 21 | 88 | 11 |
| **Negative** | 14 | 28 | 56 |
### DistilBERT

| GT \ Pred | Positive | Neutral | Negative |
|---|---|---|---|
| **Positive** | 36 | 18 | 28 |
| **Neutral** | 27 | 79 | 14 |
| **Negative** | 16 | 16 | 66 |
