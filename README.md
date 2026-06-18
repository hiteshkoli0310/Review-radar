<div align="center">
  <h1>ReviewRadar</h1>
  <strong>AI-Powered Sentiment Analysis on YouTube Product Reviews</strong>
  <br>
  <a href="https://review-radar-dashboard.streamlit.app">https://review-radar-dashboard.streamlit.app</a>
</div>

<br>

<p align="center">
  <a href="https://review-radar-dashboard.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License">
  </a>
  <img src="https://img.shields.io/github/stars/hiteshkoli0310/Review-radar?style=for-the-badge&color=yellow" alt="Stars">
  <img src="https://img.shields.io/github/issues/hiteshkoli0310/Review-radar?style=for-the-badge&color=red" alt="Issues">
  <img src="https://img.shields.io/github/last-commit/hiteshkoli0310/Review-radar?style=for-the-badge&color=green" alt="Last Commit">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python">
</p>

---

## What is ReviewRadar?

ReviewRadar is an end-to-end NLP pipeline that ingests YouTube product reviews, extracts comments, and classifies them as **Positive / Neutral / Negative** using a fine-tuned DistilBERT model. Users enter any product name, and the system searches YouTube for relevant review videos, cleans the comment text, runs sentiment analysis, and displays the results through an interactive Streamlit dashboard.

Think of it as a consumer pulse-check for any product — all from publicly available YouTube review data.

---

## Key Features

- **YouTube Data Collection** — Searches for product review videos, fetches metadata (views, likes, duration), and extracts top-level comments via YouTube Data API v3
- **Text Preprocessing** — Cleaning, spam detection, short comment removal, Hinglish (Hindi-English) normalization, and language-aware translation
- **Sentiment Classification** — Fine-tuned DistilBERT 3-class classifier (Positive / Neutral / Negative) trained on 300+ human-annotated product review comments with weighted loss
- **Rule-Based & VADER Baselines** — Keyword scorer with negation handling and NLTK VADER for comparison against the DistilBERT model
- **Competitor Detection** — Keyword-based extraction of competing product mentions from comments
- **Interactive Dashboard** — Streamlit app with Product Overview, Model Transparency (confusion matrix, precision/recall/F1), Dataset Statistics, and Run Pipeline tabs

---

## How It Works

```
Product Name → YouTube Search → Comment Extraction → Preprocessing
→ DistilBERT Sentiment → Competitor Detection → Insight Report → Dashboard
```

The entire pipeline runs from a single product name input. Results are cached and viewable immediately.

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/DistilBERT-121481?style=flat-square" alt="DistilBERT">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NLTK-154f3c?style=flat-square" alt="NLTK">
  <img src="https://img.shields.io/badge/YouTube-API-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="YouTube API">
</p>

---

## Getting Started

```bash
git clone https://github.com/hiteshkoli0310/Review-radar.git
cd Review-radar
python -m venv .venv && .venv\Scripts\Activate  # Windows
pip install -e .
```

Add your YouTube Data API v3 key to `.env`:

```
YOUTUBE_API_KEY=your_key_here
```

Run the dashboard:

```bash
python run_dashboard.py
```

---

## Live Demo

[**https://review-radar-dashboard.streamlit.app**](https://review-radar-dashboard.streamlit.app)

Explore pre-computed insights or run the pipeline on any product directly from the app.

---

## Future Work

- **Aspect Classification** — Adding a DistilBERT model to classify comments by product dimension (Battery, Camera, Performance, Display, Price, etc.)
- **Topic Modeling** — Integrating BERTopic for unsupervised theme discovery
- **Multi-language Expansion** — Broader language support beyond Hinglish
- **Temporal Analysis** — Tracking sentiment trends over time as new reviews are published

---

## License

MIT License — see [LICENSE](LICENSE).

---

<p align="center">
  <em>Turning YouTube reviews into actionable product intelligence</em>
</p>
