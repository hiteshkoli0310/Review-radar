"""Build and save human annotation templates."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

ANNOTATION_METADATA_COLUMNS = [
    "product_query",
    "video_id",
    "video_title",
    "channel_name",
    "comment_id",
    "comment_text",
    "cleaned_comment_text",
    "comment_like_count",
    "detected_language",
]

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
ASPECT_LABELS = [
    "Gaming",
    "Display",
    "Battery",
    "Camera",
    "Performance",
    "Price",
    "Competition",
    "Purchase Intent",
    "Software",
    "Hardware",
    "Spam",
    "Support",
    "Other",
]
ANNOTATION_NOTE_LABELS = [
    "Ambiguous",
    "Future Demand",
    "Language Error",
    "Question",
    "Unrelated",
]

ANNOTATION_COLUMNS = [
    "sentiment_label",
    "aspect_label",
    "review_notes",
] + [f"is_{label.lower().replace(' ', '_')}" for label in ANNOTATION_NOTE_LABELS]


def build_annotation_dataset(sample: pd.DataFrame) -> pd.DataFrame:
    """Build a human annotation CSV template from a sampled master dataset."""
    dataset = sample.copy()
    for column in ANNOTATION_METADATA_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = pd.NA

    dataset = dataset[ANNOTATION_METADATA_COLUMNS]
    for column in ANNOTATION_COLUMNS:
        dataset[column] = ""
    return dataset


def save_annotation_dataset(dataset: pd.DataFrame, output_path: Path) -> Path:
    """Save an annotation dataset template as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved annotation dataset to %s", output_path)
    return output_path


def write_annotation_guidelines(output_path: Path) -> Path:
    """Write annotation guidelines for sentiment and aspect labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_guidelines_text(), encoding="utf-8")
    logger.info("Saved annotation guidelines to %s", output_path)
    return output_path


def _note_labels_docs() -> str:
    docs = ""
    for label in ANNOTATION_NOTE_LABELS:
        if label == "Ambiguous":
            docs += "### Ambiguous\n\nUse when the comment's sentiment or aspect is unclear or could reasonably be interpreted multiple ways.\n\nExample: \"It's okay I guess\" -> Ambiguous\n\n"
        elif label == "Future Demand":
            docs += "### Future Demand\n\nUse when the comment expresses intent to buy, anticipation for a future product, demand for a feature, or desire for a product release.\n\nExample: \"Can't wait for this to launch\" -> Future Demand\n\n"
        elif label == "Language Error":
            docs += "### Language Error\n\nUse when the comment has broken language (translation artifacts, garbled text, mixed languages) that makes sentiment or aspect classification unreliable.\n\nExample: \"dhe ipo poy eduthond van athil irunnn kanunnn\" -> Language Error\n\n"
        elif label == "Question":
            docs += "### Question\n\nUse when the comment is structurally a question — typically maps to Neutral sentiment unless the question implies strong positive or negative framing.\n\nExample: \"Does this support 4K?\" -> Question\n\n"
        elif label == "Unrelated":
            docs += "### Unrelated\n\nUse when the comment content is off-topic, does not discuss the product, or is unrelated to the video's subject matter.\n\nExample: \"Nice video\" -> Unrelated\n\n"
    return docs


def _guidelines_text() -> str:
    sentiment_labels = "\n".join(f"- {label}" for label in SENTIMENT_LABELS)
    aspect_labels = "\n".join(f"- {label}" for label in ASPECT_LABELS)
    note_labels = "\n".join(f"- {label}" for label in ANNOTATION_NOTE_LABELS)
    note_docs = _note_labels_docs()
    return f"""# ReviewRadar Annotation Guidelines

Use these guidelines to label each comment manually. Label the comment as written,
using the original comment text and metadata for context. Do not infer facts that are
not present in the comment.

## Allowed Sentiment Labels

{sentiment_labels}

### Positive

Use `Positive` when the comment contains praise, recommendation, satisfaction,
approval, excitement, or a clearly favorable opinion.

Examples:
- "OLED screen is amazing" -> Positive
- "I love this console" -> Positive
- "Worth buying at this price" -> Positive

### Neutral

Use `Neutral` for questions, factual statements, unclear opinions, mixed comments
without a clear dominant sentiment, or comments that do not evaluate the product.

Examples:
- "Does it support 4K?" -> Neutral
- "It was released last year" -> Neutral
- "I have this model" -> Neutral

### Negative

Use `Negative` when the comment contains complaints, criticism, dissatisfaction,
warnings, disappointment, or a clearly unfavorable opinion.

Examples:
- "Too expensive" -> Negative
- "Battery life is terrible" -> Negative
- "Do not buy this" -> Negative

## Allowed Aspect Labels

{aspect_labels}

Choose the main aspect being discussed. If multiple aspects are present, select the
most important or most sentiment-bearing aspect in the comment.

### Gaming

Use for games, gameplay, game library, exclusive titles, multiplayer, fun factor, or
gaming experience.

Example: "Mario Kart is fun" -> Gaming

### Display

Use for screen quality, OLED/LCD, brightness, refresh rate, size, resolution, or
visual appearance.

Example: "OLED screen is amazing" -> Display

### Battery

Use for battery life, charging, power drain, charger, or portability affected by
battery.

Example: "Battery drains too fast" -> Battery

### Camera

Use for camera quality, photos, video recording, stabilization, selfies, or lenses.

Example: "Camera quality is excellent" -> Camera

### Performance

Use for speed, lag, frame rate, processor, thermals, loading time, or responsiveness.

Example: "It lags after the update" -> Performance

### Price

Use for price, value for money, discounts, expensive/cheap, affordability, or deals.

Example: "Too expensive for what it offers" -> Price

### Competition

Use for comparisons with competing products or brands.

Example: "Steam Deck is better than Switch" -> Competition

### Purchase Intent

Use when the comment expresses buying plans, recommendations to buy/not buy, ownership
intent, or purchase decisions.

Example: "Should I buy this now?" -> Purchase Intent

### Software

Use for operating system, updates, UI, apps, firmware, bugs, or software features.

Example: "The new update fixed the menu lag" -> Software

### Hardware

Use for build quality, buttons, controllers, ports, storage, speakers, body, or other
physical components.

Example: "The joystick feels cheap" -> Hardware

### Spam

Use for promotional content, solicitation, gibberish, irrelevant self-promotion, or
comments that do not contribute meaningful discussion about the product.

Examples:
- "Bhai please give steam deck please dedo" -> Spam
- "Switch 2" -> Spam (if standalone, no evaluation)

### Support

Use for customer service experiences, warranty claims, return/replacement requests,
technical support interactions, or service quality.

Example: "Anna phone kodi nanu Swiggy" -> Support

### Other

Use when no listed aspect fits or the comment is too vague to assign a meaningful
aspect.

Example: "Nice" -> Other

## Allowed Note Labels

{note_labels}

Choose zero or one note label to add additional context. Leave blank if the comment
does not fit any note category.

{note_docs}
"""

