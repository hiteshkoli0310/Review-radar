"""Aspect classification model — fine-tuned DistilBERT for aspect prediction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from reviewradar.annotation.annotation_dataset_builder import ASPECT_LABELS as ALL_ASPECT_LABELS


logger = logging.getLogger(__name__)

ASPECT_LABELS = sorted(ALL_ASPECT_LABELS)
LABEL_MAP = {label: i for i, label in enumerate(ASPECT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(ASPECT_LABELS)}


class DistilBertAspectScorer:
    """Fine-tuned DistilBERT for aspect classification."""

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
        output_dir: str = "models/distilbert_aspect",
        num_epochs: int = 15,
        batch_size: int = 16,
        lr: float = 3e-5,
        use_class_weights: bool = True,
        sample_weights: list[float] | None = None,
        from_checkpoint: str | None = None,
    ) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if from_checkpoint:
            tokenizer = DistilBertTokenizerFast.from_pretrained(from_checkpoint)
            model = DistilBertForSequenceClassification.from_pretrained(
                from_checkpoint, num_labels=len(ASPECT_LABELS),
            ).to(self._device)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=len(ASPECT_LABELS),
                id2label=ID2LABEL, label2id=LABEL_MAP,
            ).to(self._device)

        label_ids = [LABEL_MAP[l] for l in labels]
        if use_class_weights:
            class_counts = torch.bincount(torch.tensor(label_ids), minlength=len(ASPECT_LABELS))
            class_weights = (1.0 / class_counts.float().clamp(min=1))
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = class_weights.to(self._device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            loss_fn = nn.CrossEntropyLoss(reduction="none")

        if sample_weights is not None:
            sw_tensor = torch.tensor(sample_weights, dtype=torch.float)

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

        val_loader = None
        if val_texts and val_labels:
            val_ids = [LABEL_MAP[l] for l in val_labels]
            val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_ids))
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        if sample_weights is not None:
            sw_tensor_for_dataset = sw_tensor
            dataset = TensorDataset(
                encodings["input_ids"], encodings["attention_mask"],
                torch.tensor(label_ids), sw_tensor_for_dataset,
            )
        else:
            dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(label_ids))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )

        best_val_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                if sample_weights is not None:
                    input_ids, attention_mask, labels_batch, sw_batch = [b.to(self._device) for b in batch]
                else:
                    input_ids, attention_mask, labels_batch = [b.to(self._device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_batch)
                if sample_weights is not None:
                    loss = loss * sw_batch
                loss = loss.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self._model_path)
        self._model = DistilBertForSequenceClassification.from_pretrained(self._model_path).to(self._device)
        self._model.eval()

    def predict(self, text: str) -> str:
        self._load()
        inputs = self._tokenizer(str(text), truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return ID2LABEL[pred]

    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[str]:
        self._load()
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

    def predict_with_scores(self, texts: list[str], batch_size: int = 64) -> list[dict[str, Any]]:
        self._load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = self._tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            encodings = {k: v.to(self._device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self._model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            for prob in probs:
                top_idx = torch.argmax(prob).item()
                results.append({
                "aspect": ID2LABEL[top_idx],
                "confidence": float(prob[top_idx]),
                "scores": {ID2LABEL[i]: float(prob[i]) for i in range(len(ASPECT_LABELS))},
            })
        return results


def evaluate_aspect(
    test_df: pd.DataFrame,
    scorer: DistilBertAspectScorer,
    text_column: str = "cleaned_comment_text",
    label_column: str = "aspect_label",
) -> dict[str, Any]:
    """Evaluate aspect classifier against ground truth."""
    ground_truth = test_df[label_column].fillna("Other").str.strip().tolist()
    texts = test_df[text_column].fillna("").tolist()

    predictions = scorer.predict_batch(texts)

    labels = ASPECT_LABELS
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0
    )
    cm = confusion_matrix(ground_truth, predictions, labels=labels)

    per_label = {}
    for i, label in enumerate(labels):
        per_label[label.lower().replace(" ", "_")] = {
            "label": label,
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
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


def build_aspect_evaluation_report(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write aspect evaluation results as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "aspect_classification.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    md = _format_report_markdown(results)
    md_path = output_dir / "aspect_classification_report.md"
    md_path.write_text(md, encoding="utf-8")

    logger.info("Saved aspect classification report to %s", md_path)


def _format_report_markdown(results: dict[str, Any]) -> str:
    n_test = len(results.get("ground_truth", []))
    lines = ["# Aspect Classification Report\n"]
    lines.append(f"DistilBERT fine-tuned on **{n_test}** held-out test samples.\n")

    lines.append("## Overall Accuracy\n\n")
    lines.append(f"| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {results['accuracy']:.2%} |")
    lines.append(f"| Macro Precision | {results['macro_precision']:.2%} |")
    lines.append(f"| Macro Recall | {results['macro_recall']:.2%} |")
    lines.append(f"| Macro F1 | {results['macro_f1']:.2%} |")
    lines.append("")

    lines.append("## Per-Label Breakdown\n")
    lines.append("| Aspect | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for label_info in results["per_label"].values():
        flag = " ⚠️" if label_info["support"] <= 10 else ""
        lines.append(
            f"| {label_info['label']}{flag} | {label_info['precision']:.2%} "
            f"| {label_info['recall']:.2%} | {label_info['f1']:.2%} "
            f"| {label_info['support']} |"
        )
    lines.append("")

    lines.append("## Confusion Matrix\n")
    labels = results["confusion_labels"]
    cm = results["confusion_matrix"]
    header = "| GT \\ Pred | " + " | ".join(labels) + " |"
    sep = "|---|---" + "---|" * len(labels)
    lines.append(header)
    lines.append(sep)
    for i, row_label in enumerate(labels):
        row = f"| **{row_label}** | " + " | ".join(str(cm[i][j]) for j in range(len(labels))) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- ⚠️ marks classes with ≤10 test samples — metrics are unreliable.")
    lines.append("- Text notes whether class weights were used or oversampling was applied.")
    lines.append("- Model: `distilbert-base-uncased` → 13-class classification head.")
    lines.append("- Training: 15 epochs, batch_size=16, lr=3e-5, warmup=10%, weight_decay=0.01.")

    return "\n".join(lines)
