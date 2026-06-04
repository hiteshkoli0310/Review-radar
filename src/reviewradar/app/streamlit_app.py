"""Streamlit entry point for ReviewRadar."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="ReviewRadar", layout="wide")
    st.title("ReviewRadar")
    st.write("AI-powered sentiment and topic analysis for YouTube product reviews.")


if __name__ == "__main__":
    main()