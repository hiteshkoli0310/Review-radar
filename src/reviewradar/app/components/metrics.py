"""Reusable metric / KPI display helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st


def metric_card(label: str, value: Any, delta: str | None = None) -> None:
    st.metric(label=label, value=str(value), delta=delta)


def metric_row(cols: list[dict[str, Any]]) -> None:
    """Display a row of metric cards. Each dict: {label, value, delta?}."""
    n = len(cols)
    columns = st.columns(n)
    for col, data in zip(columns, cols):
        with col:
            metric_card(data.get("label", ""), data.get("value", ""), data.get("delta"))


def styled_dataframe(df: Any, height: int = 300, hide_index: bool = True) -> None:
    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=hide_index,
    )


def section_header(title: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.subheader(f"{prefix}{title}")
    st.markdown("<hr style='margin:0 0 1rem 0'>", unsafe_allow_html=True)
