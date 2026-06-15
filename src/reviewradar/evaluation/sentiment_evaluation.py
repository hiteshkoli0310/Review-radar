"""Sentiment model evaluation against human-annotated ground truth."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.annotation.annotation_dataset_builder import SENTIMENT_LABELS


logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "better", "brilliant", "excellent",
    "fantastic", "favorite", "favourite", "good", "great", "happy", "impressive",
    "incredible", "love", "loved", "lovely", "nice", "perfect", "perfection",
    "recommend", "solid", "superb", "terrific", "wonderful", "worth", "worthy",
    "fast", "smooth", "sharp", "stunning", "gorgeous", "super", "amazed",
    "beast", "killer", "legend", "best", "top", "king", "goat", "must buy",
    "must have", "bang for buck", "value", "underrated", "outstanding",
    "impressed", "pleased", "satisfied", "reliable", "durable", "convenient",
    "intuitive", "easy", "seamless", "flawless", "comfortable", "lightweight",
    "powerful", "efficient", "quality", "premium", "innovative", "game changer",
    "gamechanger", "breakthrough", "upgrade", "improvement", "improved",
}

NEGATIVE_WORDS = {
    "bad", "worse", "worst", "terrible", "horrible", "awful", "poor", "garbage",
    "trash", "hate", "hated", "ugly", "junk", "broken", "useless", "pathetic",
    "disappointed", "disappointing", "disaster", "sucks", "suck", "crap", "shit",
    "overpriced", "expensive", "overrated", "underwhelming", "mediocre", "average",
    "slow", "lag", "laggy", "bug", "bugs", "buggy", "crash", "crashes", "freeze",
    "freezes", "glitch", "glitchy", "terrible", "dreadful", "atrocious", "abysmal",
    "regret", "regretful", "mistake", "avoid", "stay away", "not worth", "ripoff",
    "rip off", "scam", "fraud", "fault", "defective", "fail", "failed", "failure",
    "problem", "issue", "struggle", "unstable", "inconsistent", "unreliable",
    "fragile", "cheap", "flimsy", "noisy", "loud", "bulky", "heavy", "outdated",
    "obsolete", "boring", "dull", "annoying", "frustrating", "infuriating",
    "uncomfortable", "difficult", "confusing", "waste", "waste of money",
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "unbelievably", "absolutely",
    "completely", "totally", "utterly", "highly", "so", "too", "super",
}

NEGATORS = {"not", "no", "never", "neither", "nor", "cannot", "can't", "don't",
             "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "isn't",
             "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"}


class RuleBasedScorer:
    """Simple keyword-counting sentiment scorer using curated lexicons."""

    def __init__(self) -> None:
        self.pos_words = POSITIVE_WORDS
        self.neg_words = NEGATIVE_WORDS
        self.intensifiers = INTENSIFIERS
        self.negators = NEGATORS

    def predict(self, text: str) -> str:
        score = self._score(str(text).lower())
        if score > 0:
            return "Positive"
        if score < 0:
            return "Negative"
        return "Neutral"

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def _score(self, text: str) -> int:
        words = re.findall(r"\b[a-z]+\b", text)
        score = 0
        i = 0
        while i < len(words):
            word = words[i]
            negated = False
            if i > 0 and words[i - 1] in self.negators:
                negated = True

            multiplier = 1
            if i > 0 and words[i - 1] in self.intensifiers:
                multiplier = 2

            if word in self.pos_words:
                score += multiplier * (-1 if negated else 1)
            elif word in self.neg_words:
                score -= multiplier * (-1 if negated else 1)
            i += 1
        return score


class VaderScorer:
    """VADER sentiment scorer via NLTK."""

    def __init__(self) -> None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self._sia = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> str:
        scores = self._sia.polarity_scores(str(text))
        compound = scores["compound"]
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def score(self, text: str) -> dict[str, float]:
        return self._sia.polarity_scores(str(text))


class RobertaScorer:
    """HuggingFace RoBERTa sentiment pipeline with lazy loading and GPU."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
        self._model_name = model_name
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> str:
        self._load()
        result = self._pipeline(str(text))[0]
        return result["label"].title()

    def predict_batch(self, texts: list[str]) -> list[str]:
        self._load()
        results = self._pipeline(texts, truncation=True, max_length=512)
        return [r["label"].title() for r in results]


LABEL_MAP = {"Positive": 0, "Neutral": 1, "Negative": 2}
ID2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}


class DistilBertScorer:
    """Fine-tuned DistilBERT for sentiment classification."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = None

    def train(
        self,
        texts: list[str],
        labels: list[str],
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
        output_dir: str = "models/distilbert_sentiment",
        num_epochs: int = 10,
        batch_size: int = 16,
        lr: float = 3e-5,
        freeze_encoder: bool = False,
        sample_weights: list[float] | None = None,
    ) -> None:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            get_linear_schedule_with_warmup,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3, id2label=ID2LABEL, label2id=LABEL_MAP
        ).to(self._device)

        if freeze_encoder:
            for name, param in model.distilbert.named_parameters():
                param.requires_grad = False

        label_ids = [LABEL_MAP[l] for l in labels]
        class_counts = torch.bincount(torch.tensor(label_ids))
        class_weights = (1.0 / class_counts.float())
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(self._device)

        use_sample_weights = sample_weights is not None and any(w != 1.0 for w in sample_weights)
        if use_sample_weights:
            loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

        if use_sample_weights:
            sw = torch.tensor(sample_weights, dtype=torch.float32)
            dataset = TensorDataset(
                encodings["input_ids"], encodings["attention_mask"],
                torch.tensor(label_ids), sw,
            )
        else:
            dataset = TensorDataset(
                encodings["input_ids"], encodings["attention_mask"], torch.tensor(label_ids),
            )

        val_loader = None
        if val_texts and val_labels:
            val_ids = [LABEL_MAP[l] for l in val_labels]
            val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_ids))
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        total_steps = len(loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                if use_sample_weights:
                    input_ids, attention_mask, labels_batch, sw_batch = [b.to(self._device) for b in batch]
                else:
                    input_ids, attention_mask, labels_batch = [b.to(self._device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if use_sample_weights:
                    per_sample_loss = loss_fn(logits, labels_batch)
                    weighted_loss = (per_sample_loss * sw_batch).mean()
                else:
                    weighted_loss = loss_fn(logits, labels_batch)
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += weighted_loss.item()

            val_acc = 0.0
            if val_loader:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels_batch = [b.to(self._device) for b in batch]
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=-1)
                        correct += (preds == labels_batch).sum().item()
                        total += labels_batch.size(0)
                val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

        if not val_loader:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        self._model_path = output_dir
        self._model = model
        self._tokenizer = tokenizer

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self._model_path)
        self._model = DistilBertForSequenceClassification.from_pretrained(self._model_path).to(self._device)
        self._model.eval()

    def predict(self, text: str) -> str:
        self._load()
        import torch
        inputs = self._tokenizer(str(text), truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return ID2LABEL[pred]

    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[str]:
        self._load()
        import torch
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = self._tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            encodings = {k: v.to(self._device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self._model(**encodings)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()
            all_preds.extend(preds)
        return [ID2LABEL[p] for p in all_preds]


def _parse_ground_truth(label: Any) -> str:
    if pd.isna(label) or str(label).strip() == "":
        return "Neutral"
    return str(label).strip().title()


def evaluate_sentiment(
    annotated: pd.DataFrame,
    scorers: dict[str, Any],
    text_column: str = "cleaned_comment_text",
    label_column: str = "sentiment_label",
) -> dict[str, Any]:
    """Evaluate all scorers against ground truth sentiment labels."""
    ground_truth = annotated[label_column].apply(_parse_ground_truth)
    texts = annotated[text_column].fillna("").tolist()

    results: dict[str, Any] = {}
    for name, scorer in scorers.items():
        predictions = scorer.predict_batch(texts)
        metrics = _compute_metrics(ground_truth.tolist(), predictions)
        metrics["predictions"] = predictions
        results[name] = metrics
    results["ground_truth"] = ground_truth.tolist()
    return results


def _compute_metrics(
    ground_truth: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    labels = ["Positive", "Neutral", "Negative"]

    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0
    )
    cm = confusion_matrix(ground_truth, predictions, labels=labels)

    per_label = {}
    for i, label in enumerate(labels):
        per_label[label.lower()] = {
            "precision": round(float(precision[i]), 4),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_label": per_label,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
    }


def build_evaluation_report(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write evaluation results as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sentiment_evaluation.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    md = _format_report_markdown(results)
    md_path = output_dir / "sentiment_evaluation_report.md"
    md_path.write_text(md, encoding="utf-8")

    logger.info("Saved sentiment evaluation to %s", md_path)


def _format_report_markdown(results: dict[str, Any]) -> str:
    lines = ["# Sentiment Evaluation Report\n"]
    lines.append(f"Comparison of sentiment approaches against **{len(results.get('ground_truth', []))}** human-annotated comments.\n")

    lines.append("## Overall Accuracy\n\n")
    lines.append("| Approach | Accuracy | Macro Precision | Macro Recall | Macro F1 |")
    lines.append("|---|---|---|---|---|")
    scores = []
    for name in results:
        if name == "ground_truth":
            continue
        m = results[name]
        lines.append(
            f"| {name} | {m['accuracy']:.2%} | {m['macro_precision']:.2%} "
            f"| {m['macro_recall']:.2%} | {m['macro_f1']:.2%} |"
        )
        scores.append((name, m["accuracy"]))
    lines.append("")

    best = max(scores, key=lambda x: x[1]) if scores else ("", 0)
    lines.append(f"**Best approach:** {best[0]} ({best[1]:.2%} accuracy)\n")

    lines.append("## Per-Label Breakdown\n")
    for name in results:
        if name == "ground_truth":
            continue
        m = results[name]
        lines.append(f"### {name}\n")
        lines.append("| Label | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for label in ("positive", "neutral", "negative"):
            p = m["per_label"].get(label, {})
            lines.append(
                f"| {label.title()} | {p.get('precision', 0):.2%} "
                f"| {p.get('recall', 0):.2%} | {p.get('f1', 0):.2%} "
                f"| {p.get('support', 0)} |"
            )
        lines.append("")

    lines.append("## Confusion Matrices\n")
    for name in results:
        if name == "ground_truth":
            continue
        m = results[name]
        labels = m["confusion_labels"]
        cm = m["confusion_matrix"]
        lines.append(f"### {name}\n")
        header = "| GT \\ Pred | " + " | ".join(labels) + " |"
        sep = "|---|---" + "---|" * len(labels)
        lines.append(header)
        lines.append(sep)
        for i, row_label in enumerate(labels):
            row = f"| **{row_label}** | " + " | ".join(str(cm[i][j]) for j in range(len(labels))) + " |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)
