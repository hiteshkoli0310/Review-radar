# ReviewRadar

ReviewRadar is an AI-powered product sentiment analysis platform built around YouTube video discovery, comment ingestion, NLP preprocessing, sentiment analysis, topic modeling, and Streamlit analytics.

## Architecture Overview

The project follows a `src/`-based modular layout:

- `src/reviewradar/data_collection` for YouTube search and comment extraction
- `src/reviewradar/preprocessing` for cleaning, normalization, and feature preparation
- `src/reviewradar/eda` for reusable exploratory analysis helpers
- `src/reviewradar/models` for sentiment, aspect sentiment, topic modeling, and summarization
- `src/reviewradar/visualization` for reusable Plotly chart builders
- `src/reviewradar/app` for Streamlit pages and app orchestration
- `src/reviewradar/pipelines` for end-to-end workflow coordination
- `src/reviewradar/config` for config, paths, and environment loading
- `src/reviewradar/utils` for logging, I/O, constants, and shared helpers

## Data Layout

- `data/raw/` for immutable API pulls
- `data/interim/` for transitional cleaned data
- `data/processed/` for analysis-ready parquet datasets
- `data/models/` for serialized models, embeddings, and topic artifacts

Parquet is the default storage format for tabular datasets because it is compact, fast, and analytics-friendly.

## Development Workflow

1. Copy `.env.example` to `.env` and add your API credentials.
2. Install dependencies with `pip install -e .` for development or `pip install -r requirements.txt` for a lightweight environment.
3. Use notebooks only for EDA and experimentation.
4. Keep production logic inside `src/reviewradar/`.
5. Write tests under `tests/` for data transforms and pipeline behavior.

## Notes

- Keep secrets out of version control.
- Prefer config-driven paths and parameters.
- Use scripts or pipeline entry points for repeatable runs.
