"""Plotly chart builders for dashboard visualizations."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go


def sentiment_bar_chart(sentiment_dist: dict[str, dict[str, Any]]) -> go.Figure:
    labels = ["Positive", "Neutral", "Negative"]
    values = [sentiment_dist.get(l.lower(), {}).get("count", 0) for l in labels]
    pcts = [sentiment_dist.get(l.lower(), {}).get("pct", 0) for l in labels]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v} ({p:.1f}%)" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="%{x}: %{y} comments (%{customdata:.1f}%)<extra></extra>",
        customdata=pcts,
    ))
    fig.update_layout(
        title="Sentiment Distribution",
        yaxis_title="Comments",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def aspect_bar_chart(aspects: list[dict[str, Any]]) -> go.Figure:
    if not aspects:
        return go.Figure()
    aspects = aspects[:8]
    labels = [a["aspect"] for a in aspects]
    values = [a["count"] for a in aspects]
    pcts = [a["pct"] for a in aspects]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color="#3498db",
        text=[f"{v} ({p:.1f}%)" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="%{y}: %{x} comments (%{customdata:.1f}%)<extra></extra>",
        customdata=pcts,
    ))
    fig.update_layout(
        title="Most Discussed Aspects",
        xaxis_title="Comments",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def competitor_bar_chart(competitors: list[dict[str, Any]]) -> go.Figure:
    if not competitors:
        return go.Figure()
    competitors = competitors[:8]
    labels = [c["competitor"] for c in competitors]
    values = [c["mentions"] for c in competitors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color="#e67e22",
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Competitor Mentions",
        xaxis_title="Mentions",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def aspect_sentiment_heatmap(matrix: dict[str, dict[str, int]]) -> go.Figure:
    if not matrix:
        return go.Figure()
    aspects = sorted(matrix.keys())
    sentiments = ["positive", "neutral", "negative"]
    z = [[matrix[a][s] for s in sentiments] for a in aspects]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Positive", "Neutral", "Negative"],
        y=aspects,
        colorscale="Blues",
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="%{y}: %{x} = %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Aspect × Sentiment Matrix",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def confusion_matrix_heatmap(
    cm: list[list[int]], labels: list[str], title: str = "Confusion Matrix"
) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Reds",
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=max(400, 25 * len(labels)),
        margin=dict(l=80, r=20, t=40, b=80),
    )
    return fig


def sentiment_comparison_chart(eval_data: dict[str, Any]) -> go.Figure:
    approaches = [k for k in eval_data if k != "ground_truth"]
    metrics_display = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()
    colors = {"Rule-Based": "#95a5a6", "VADER": "#f39c12",
              "RoBERTa": "#9b59b6", "DistilBERT": "#2ecc71"}
    for app in approaches:
        data = eval_data.get(app, {})
        if isinstance(data, dict):
            vals = [data.get(m, 0) * 100 for m in metrics_display]
            fig.add_trace(go.Bar(
                name=app.replace("Scorer", ""),
                x=metric_labels,
                y=vals,
                marker_color=colors.get(app, "#3498db"),
            ))

    fig.update_layout(
        title="Sentiment Model Comparison",
        barmode="group",
        yaxis_title="%",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def per_class_f1_chart(eval_data: dict[str, Any]) -> go.Figure:
    approaches = [k for k in eval_data if k != "ground_truth"]
    labels = ["Positive", "Neutral", "Negative"]
    colors = {"Rule-Based": "#95a5a6", "VADER": "#f39c12",
              "RoBERTa": "#9b59b6", "DistilBERT": "#2ecc71"}

    fig = go.Figure()
    for app in approaches:
        pl = eval_data.get(app, {}).get("per_label", {})
        vals = [pl.get(l.lower(), {}).get("f1", 0) * 100 for l in labels]
        fig.add_trace(go.Bar(
            name=app,
            x=labels,
            y=vals,
            marker_color=colors.get(app, "#3498db"),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        title="Per-Class F1 Score by Model",
        barmode="group",
        yaxis_title="F1 %",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def language_pie_chart(master: "pd.DataFrame") -> go.Figure:
    lang_counts = master["detected_language"].value_counts().reset_index()
    lang_counts.columns = ["language", "count"]
    fig = px.pie(
        lang_counts.head(8), names="language", values="count",
        title="Language Distribution",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig



