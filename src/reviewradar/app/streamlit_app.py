"""Single-page Streamlit dashboard for ReviewRadar."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure src is on sys.path
_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from reviewradar.app.config import PAGE_TITLE, DEFAULT_PRODUCT, SENTIMENT_MODEL
import pandas as pd

from reviewradar.app.components.data_loader import (
    load_master,
    get_product_names,
    get_product_df,
    get_total_comments,
    get_annotated_count,
    parse_insight_report,
    load_sentiment_eval,
    load_aspect_eval,
    get_chart_path,
    invalidate_master_cache,
)
from reviewradar.app.components.charts import (
    sentiment_bar_chart,
    competitor_bar_chart,
    confusion_matrix_heatmap,
    sentiment_comparison_chart,
    per_class_f1_chart,
    language_pie_chart,
)
from reviewradar.app.components.metrics import metric_row, styled_dataframe, section_header
from reviewradar.app.components.run_job import run_pipeline_with_progress
from reviewradar.app.pipeline_orchestrator import check_product_cached
from reviewradar.pipelines.run_pipeline import build_product_slug


# ── Cached model resources (loaded once per session) ────────────────────────


@st.cache_resource
def get_sentiment_scorer():
    from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer
    scorer = DistilBertScorer(model_path=str(SENTIMENT_MODEL))
    scorer._load()
    return scorer


# ── Product Overview tab ────────────────────────────────────────────────────


def _normalize_sentiment_keys(insight: dict[str, Any]) -> dict[str, Any]:
    """Ensure sentiment_distribution keys are lowercase (dashboard expects lowercase)."""
    sd = insight.get("sentiment_distribution", {})
    if sd and any(k[0].isupper() for k in sd):
        insight["sentiment_distribution"] = {k.lower(): v for k, v in sd.items()}
    return insight


def _resolve_insight(product: str) -> tuple[dict[str, Any] | None, str]:
    """Return (insight_dict, source_label) for the given product.

    Checks session state first (for freshly-pipelined products),
    then oversampled disk reports, then baseline disk reports.
    """
    pr = st.session_state.get("pipeline_results", {})
    if product in pr:
        insight = pr[product]
        return _normalize_sentiment_keys(insight), "pipeline"

    insight = parse_insight_report(product, use_oversampled=True)
    if insight:
        return insight, "oversampled"

    insight = parse_insight_report(product)
    if insight:
        return insight, "baseline"

    return None, "none"


def _product_overview_tab() -> None:
    st.header("Product Overview")

    # Check if a new product was just processed — auto-select it
    pipeline_result = st.session_state.get("pipeline_results", {})
    default_product = DEFAULT_PRODUCT
    if pipeline_result:
        pipeline_products = list(pipeline_result.keys())
        if pipeline_products:
            default_product = pipeline_products[0]

    try:
        master = load_master()
        products = get_product_names()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        master = pd.DataFrame()
        products = []

    default_idx = products.index(default_product) if default_product in products else 0
    selected = st.selectbox("Select a product", products, index=default_idx)

    product_df = get_product_df(selected) if not master.empty else pd.DataFrame()
    insight, source = _resolve_insight(selected)

    # Show a one-time badge if this product was just pipelined
    if st.session_state.pop("show_pipeline_badge", False):
        st.success("This product was just analyzed by the pipeline.")

    # ── KPI row ────────────────────────────────────────────────────────
    if insight:
        sd = insight.get("sentiment_distribution", {})
        total = insight.get("total_comments", len(product_df) if not product_df.empty else 0)
        pos = sd.get("positive", {}).get("pct", 0)
        neg = sd.get("negative", {}).get("pct", 0)
        metric_row([
            {"label": "Comments Analyzed", "value": f"{total:,}"},
            {"label": "Positive", "value": f"{pos}%", "delta": f"{pos - neg:+.1f}% vs Negative"},
            {"label": "Neutral", "value": f"{sd.get('neutral', {}).get('pct', 0)}%"},
            {"label": "Negative", "value": f"{neg}%"},
        ])
    elif not product_df.empty:
        metric_row([
            {"label": "Comments", "value": f"{len(product_df):,}"},
            {"label": "Translated", "value": f"{product_df['is_translated'].sum():,}"},
        ])
    else:
        st.info("Select a product to view insights.")

    col1, col2 = st.columns(2)

    # ── Sentiment bar ──────────────────────────────────────────────────
    with col1:
        if insight:
            sd = insight.get("sentiment_distribution", {})
            st.plotly_chart(sentiment_bar_chart(sd), use_container_width=True)
        else:
            st.info("Sentiment data not available for this product.")

    # ── Competitor mentions ────────────────────────────────────────────
    with col2:
        if insight and insight.get("competitor_mentions"):
            st.plotly_chart(competitor_bar_chart(insight["competitor_mentions"]), use_container_width=True)
        else:
            st.info("No competitor mentions found.")

    # ── Sample comments ────────────────────────────────────────────────
    with st.expander("Sample Comments", expanded=False):
        if insight and not product_df.empty:
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_a:
                sent_filter = st.selectbox(
                    "Filter by sentiment",
                    ["All", "Positive", "Neutral", "Negative"],
                    key="sample_sentiment",
                )
            with col_b:
                max_likes = int(product_df["comment_like_count"].max())
                for _key in ("min_likes_slider", "min_likes_input", "min_likes"):
                    if _key not in st.session_state:
                        st.session_state[_key] = 0

                def _sync_from_slider():
                    v = st.session_state["min_likes_slider"]
                    st.session_state["min_likes"] = v
                    st.session_state["min_likes_input"] = v

                def _sync_from_input():
                    v = st.session_state["min_likes_input"]
                    st.session_state["min_likes"] = v
                    st.session_state["min_likes_slider"] = v

                sub_a, sub_b = st.columns([3, 1])
                with sub_a:
                    st.slider(
                        "Minimum likes", 0, max_likes,
                        key="min_likes_slider", on_change=_sync_from_slider,
                    )
                with sub_b:
                    st.number_input(
                        "", 0, max_likes,
                        key="min_likes_input", on_change=_sync_from_input,
                    )
                min_likes = st.session_state["min_likes"]
            with col_c:
                st.markdown("##### &nbsp;")
                if st.button("🔄 Shuffle", use_container_width=True):
                    st.session_state["shuffle_seed"] = st.session_state.get("shuffle_seed", 0) + 1

            # Apply filters
            filtered = product_df.copy()
            if sent_filter != "All":
                if "predicted_sentiment" not in filtered.columns:
                    scorer = get_sentiment_scorer()
                    filtered["predicted_sentiment"] = scorer.predict_batch(
                        filtered["cleaned_comment_text"].fillna("").tolist()
                    )
                filtered = filtered[filtered["predicted_sentiment"] == sent_filter]
            filtered = filtered[filtered["comment_like_count"] >= min_likes]
            filtered = filtered.dropna(subset=["cleaned_comment_text"])

            # Stable sampling (only Shuffle changes the seed)
            seed = st.session_state.get("shuffle_seed", 0)
            n = min(15, len(filtered))
            sampled = filtered.sample(n, random_state=seed) if n > 0 else filtered

            for _, row in sampled.head(10).iterrows():
                likes = int(row["comment_like_count"])
                text = str(row["cleaned_comment_text"])[:200]
                st.markdown(f"- 👍 {likes}  *{text}*")
        elif not product_df.empty:
            st.info("Insight data not available for this product.")
        else:
            st.info("Select a product to browse comments.")


# ── Model Transparency tab ──────────────────────────────────────────────────


def _model_transparency_tab() -> None:
    st.header("Model Transparency")

    # ── Sentiment panel ────────────────────────────────────────────────
    section_header("Sentiment Classification", icon="📊")

    sent_eval = load_sentiment_eval()

    # Comparison chart
    st.plotly_chart(sentiment_comparison_chart(sent_eval), use_container_width=True)

    # Per-class F1 comparison
    st.plotly_chart(per_class_f1_chart(sent_eval), use_container_width=True)

    # DistilBERT details
    db = sent_eval.get("DistilBERT", sent_eval.get("DistilBertScorer", {}))
    if isinstance(db, dict):
        acc = db.get("accuracy", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DistilBERT Accuracy", f"{acc:.2%}")
        col2.metric("Macro Precision", f"{db.get('macro_precision', 0):.2%}")
        col3.metric("Macro Recall", f"{db.get('macro_recall', 0):.2%}")
        col4.metric("Macro F1", f"{db.get('macro_f1', 0):.2%}")
        st.caption("Full fine-tune DistilBERT on 300 annotated samples, like-weighted (5-fold CV ~60%)")

        per_label = db.get("per_label", {})
        cm = db.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        cm_labels = db.get("confusion_labels", ["Positive", "Neutral", "Negative"])
        if per_label:
            # Compute per-class accuracy from confusion matrix
            total = sum(sum(row) for row in cm)
            per_class_acc = {}
            for i, label in enumerate(cm_labels):
                tp = cm[i][i]
                tn = total - sum(cm[i]) - sum(cm[j][i] for j in range(len(cm))) + tp
                fp = sum(cm[j][i] for j in range(len(cm))) - tp
                fn = sum(cm[i]) - tp
                per_class_acc[label.lower()] = (tp + tn) / total if total > 0 else 0

            rows = []
            for k, v in per_label.items():
                rows.append({
                    "Label": v.get("label", k),
                    "Accuracy": f"{per_class_acc.get(k, 0):.2%}",
                    "Precision": f"{v.get('precision', 0):.2%}",
                    "Recall": f"{v.get('recall', 0):.2%}",
                    "F1": f"{v.get('f1', 0):.2%}",
                    "Support": v.get("support", 0),
                })
            st.subheader("Per-Class Performance (DistilBERT)")
            styled_dataframe(rows, height=200)

        cm_labels = db.get("confusion_labels", ["Positive", "Neutral", "Negative"])
        cm = db.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]])
        st.plotly_chart(confusion_matrix_heatmap(cm, cm_labels, "DistilBERT Confusion Matrix"), use_container_width=True)

    # ── Aspect panel ───────────────────────────────────────────────────
    section_header("Aspect Classification", icon="🏷️")

    asp_eval = load_aspect_eval()
    acc_asp = asp_eval.get("accuracy", 0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DistilBERT Accuracy", f"{acc_asp:.2%}")
    col2.metric("Macro Precision", f"{asp_eval.get('macro_precision', 0):.2%}")
    col3.metric("Macro Recall", f"{asp_eval.get('macro_recall', 0):.2%}")
    col4.metric("Macro F1", f"{asp_eval.get('macro_f1', 0):.2%}")
    st.caption("Oversampled model — 13 classes, trained on 897 balanced samples (69/class)")

    per_label_asp = asp_eval.get("per_label", {})
    if per_label_asp:
        rows = []
        for k, v in per_label_asp.items():
            rows.append({
                "Aspect": v.get("label", k),
                "Precision": f"{v.get('precision', 0):.2%}",
                "Recall": f"{v.get('recall', 0):.2%}",
                "F1": f"{v.get('f1', 0):.2%}",
                "Support": v.get("support", 0),
            })
        st.subheader("Per-Class Performance")
        styled_dataframe(rows, height=350)

    cm_asp = asp_eval.get("confusion_matrix", [])
    cm_labels_asp = asp_eval.get("confusion_labels", [])
    if cm_asp and cm_labels_asp:
        st.plotly_chart(confusion_matrix_heatmap(cm_asp, cm_labels_asp, "Aspect Confusion Matrix"), use_container_width=True)


# ── Dataset Stats tab ───────────────────────────────────────────────────────


def _dataset_stats_tab() -> None:
    st.header("Dataset Statistics")

    try:
        master = load_master()
        total = get_total_comments(master)
        annotated = get_annotated_count()
        products = get_product_names()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning("Master dataset not found. Run the pipeline first!")
        return
    products_str = ", ".join(products) if products else "None"

    metric_row([
        {"label": "Total Comments", "value": f"{total:,}"},
        {"label": "Annotated", "value": f"{annotated:,}"},
        {"label": "Products", "value": len(products)},
        {"label": "Spam Flagged", "value": f"{master['is_spam'].sum():,}"},
        {"label": "Translated", "value": f"{master['is_translated'].sum():,}"},
    ])

    section_header("Per-Product Breakdown")
    prod_stats = master.groupby("product_query").agg(
        Comments=("comment_id", "count"),
        Spam=("is_spam", "sum"),
        Translated=("is_translated", "sum"),
        Short=("is_short_comment", "sum"),
    ).reset_index()
    styled_dataframe(prod_stats, height=200)

    col1, col2 = st.columns(2)
    with col1:
        section_header("Language Distribution")
        st.plotly_chart(language_pie_chart(master), use_container_width=True)

    with col2:
        section_header("Top Languages")
        lang_counts = master["detected_language"].value_counts().head(8).reset_index()
        lang_counts.columns = ["Language", "Count"]
        styled_dataframe(lang_counts, height=250)

    section_header("Annotation Dataset")
    col5, col6, col7 = st.columns(3)
    ann_df = pd.read_csv("data/annotation/manual_review_sample.csv") if annotated else None
    if ann_df is not None:
        sent_dist = ann_df["sentiment_label"].value_counts()
        asp_dist = ann_df["aspect_label"].value_counts()
        col5.metric("Sentiment Labels", f"{len(sent_dist)} (Pos/Neu/Neg)")
        col6.metric("Aspect Labels", f"{len(asp_dist)} classes")
        col7.metric("Per Product", "100 each")
    else:
        col5.metric("Annotated Samples", f"{annotated:,}")

    section_header("Precomputed Charts")
    chart_names = [
        "sentiment_per_product.png",
        "aspect_distribution.png",
        "aspect_per_product.png",
        "sentiment_aspect_heatmap.png",
    ]
    chart_cols = st.columns(2)
    for i, name in enumerate(chart_names):
        path = get_chart_path(name)
        if path:
            with chart_cols[i % 2]:
                st.image(str(path), use_container_width=True)


# ── Run Pipeline tab ────────────────────────────────────────────────────────


def _run_pipeline_tab() -> None:
    st.header("Run Pipeline")
    st.info("Enter a product name to search YouTube, extract comments, and generate insights.")

    product_name = st.text_input("Product name", placeholder="e.g. iPhone 17, Steam Deck")
    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button(
            "Run Full Pipeline",
            type="primary",
            disabled=not product_name,
        )

    if run_clicked and product_name:
        slug = build_product_slug(product_name)
        cached = check_product_cached(slug)

        if cached:
            st.info(f"**{product_name}** was already processed — regenerating insights...")

        sentiment_scorer = get_sentiment_scorer()

        insights = run_pipeline_with_progress(product_name, sentiment_scorer)

        if insights:
            st.session_state["pipeline_results"] = {product_name: insights}
            st.session_state["show_pipeline_badge"] = True
            invalidate_master_cache()
            st.rerun()
        else:
            st.error(f"No data could be collected for '{product_name}'.")

    st.divider()
    st.markdown("**Quick links** to existing reports:")
    st.markdown("- [Annotation Guidelines](data/annotation/annotation_guidelines.md)")
    st.markdown("- [Sentiment Evaluation Report](data/reports/sentiment_evaluation_report.md)")
    st.markdown("- [Aspect Evaluation Report](data/reports/aspect_classification_oversampled_report.md)")
    st.markdown("- [Insights Summary](data/reports/insights/product_insights_summary.md)")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="📡")

    st.title("📡 ReviewRadar")
    st.caption("AI-powered consumer intelligence from YouTube product reviews")

    tab_overview, tab_transparency, tab_stats, tab_pipeline = st.tabs([
        "🔍 Product Overview", "📊 Model Transparency", "📈 Dataset Stats", "⚙️ Run Pipeline",
    ])

    with tab_overview:
        _product_overview_tab()

    with tab_transparency:
        _model_transparency_tab()

    with tab_stats:
        _dataset_stats_tab()

    with tab_pipeline:
        _run_pipeline_tab()


if __name__ == "__main__":
    main()
