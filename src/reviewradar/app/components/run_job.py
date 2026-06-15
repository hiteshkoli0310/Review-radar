"""Streamlit job runner — wraps pipeline orchestration with progress tracking."""

from __future__ import annotations

from typing import Any

import streamlit as st

from reviewradar.app.pipeline_orchestrator import run_pipeline_for_product
from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer as SentimentScorer


def run_pipeline_with_progress(
    product_name: str,
    sentiment_scorer: SentimentScorer,
) -> dict[str, Any] | None:
    """Run the full pipeline inside a ``st.status()`` context with live progress.

    Returns the product insight dict, or ``None`` on failure / empty results.
    """
    status = st.status(f"Running pipeline for **{product_name}**...", expanded=True)
    status_output = status  # used as a container for .write calls

    progress_data: dict[str, Any] = {
        "steps": [],
        "current_step": "",
        "step_index": 0,
        "total_steps": 1,
    }

    def _progress(msg: str, step: int, total: int) -> None:
        progress_data["current_step"] = msg
        progress_data["step_index"] = step
        progress_data["total_steps"] = total
        status_output.write(f"  {msg}")
        status.update(label=f"**{progress_data['current_step']}**")

    try:
        insights = run_pipeline_for_product(
            product_query=product_name,
            sentiment_scorer=sentiment_scorer,
            progress_callback=_progress,
        )

        if insights is None:
            status.update(
                label=f"No data collected for **{product_name}**",
                state="error",
            )
            st.warning(f"No YouTube data found for '{product_name}'. Try a different product name.")
            return None

        status.update(
            label=f"Pipeline complete for **{product_name}** — {insights.get('total_comments', 0)} comments analyzed",
            state="complete",
        )
        return insights

    except Exception as exc:
        status.update(
            label=f"Pipeline failed for **{product_name}**",
            state="error",
        )
        st.exception(exc)
        return None
