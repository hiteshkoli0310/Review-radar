"""Sampling helpers for human annotation datasets."""

from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def generate_balanced_sample(
    frame: pd.DataFrame,
    sample_size: int = 300,
    random_state: int = 42,
    product_column: str = "product_query",
) -> pd.DataFrame:
    """Generate a reproducible product-balanced annotation sample."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    if frame.empty:
        return frame.copy()

    if product_column not in frame.columns:
        sample_count = min(sample_size, len(frame))
        return frame.sample(n=sample_count, random_state=random_state).reset_index(drop=True)

    target_size = min(sample_size, len(frame))
    products = sorted(frame[product_column].dropna().astype(str).unique().tolist())
    if not products:
        return frame.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    base_quota = target_size // len(products)
    remainder = target_size % len(products)
    sampled_parts: list[pd.DataFrame] = []
    sampled_indexes: set[int] = set()

    for index, product in enumerate(products):
        product_rows = frame[frame[product_column].astype(str) == product]
        quota = base_quota + (1 if index < remainder else 0)
        quota = min(quota, len(product_rows))
        if quota == 0:
            continue

        sample = product_rows.sample(n=quota, random_state=random_state + index)
        sampled_parts.append(sample)
        sampled_indexes.update(sample.index.tolist())

    sampled = pd.concat(sampled_parts) if sampled_parts else pd.DataFrame(columns=frame.columns)
    remaining_slots = target_size - len(sampled)
    if remaining_slots > 0:
        remaining_rows = frame.drop(index=list(sampled_indexes))
        if not remaining_rows.empty:
            fill_count = min(remaining_slots, len(remaining_rows))
            fill_sample = remaining_rows.sample(
                n=fill_count,
                random_state=random_state + len(products),
            )
            sampled = pd.concat([sampled, fill_sample])

    logger.info("Generated annotation sample with %s rows", len(sampled))
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
