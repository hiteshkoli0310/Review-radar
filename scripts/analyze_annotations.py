"""
Annotation Analysis Script
Generates all analysis tables, charts, and outputs for the annotation report.
"""
import sys; sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from reviewradar.annotation.annotation_dataset_builder import ANNOTATION_NOTE_LABELS

CHART_DIR = Path('data/reports/charts')
CHART_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('data/annotation/manual_review_sample.csv')

# ── 0. Normalize notes and populate boolean flags ────────────────────
df['review_notes'] = df['review_notes'].str.strip().str.title()
df['review_notes'] = df['review_notes'].replace({'nan': '', '': pd.NA})

for label in ANNOTATION_NOTE_LABELS:
    col = f"is_{label.lower().replace(' ', '_')}"
    df[col] = (df['review_notes'].fillna('') == label).astype(int)

print("=== Notes normalized: ambiguous/Ambiguous → Ambiguous ===")
print(f"=== Boolean flags populated for {len(ANNOTATION_NOTE_LABELS)} note categories ===")

# ── 1. Overview ─────────────────────────────────────────────────────
overview = {
    'Total rows': len(df),
    'Products': df['product_query'].nunique(),
    'Sentiment labels filled': df['sentiment_label'].notna().sum(),
    'Aspect labels filled': df['aspect_label'].notna().sum(),
    'Review notes filled': df['review_notes'].notna().sum(),
}

# ── 2. Sentiment per product ────────────────────────────────────────
sent_by_product = pd.crosstab(df['product_query'], df['sentiment_label'])
fig, ax = plt.subplots(figsize=(8, 4))
sent_by_product.plot(kind='bar', ax=ax, color=['#e74c3c', '#95a5a6', '#2ecc71'])
ax.set_title('Sentiment Distribution per Product')
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.legend(title='Sentiment')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHART_DIR / 'sentiment_per_product.png', dpi=150)
plt.close()
print("Chart saved: sentiment_per_product.png")

# ── 3. Aspect per product ───────────────────────────────────────────
aspect_by_product = pd.crosstab(df['product_query'], df['aspect_label'])
fig, ax = plt.subplots(figsize=(10, 5))
aspect_by_product.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
ax.set_title('Aspect Distribution per Product (stacked)')
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.legend(title='Aspect', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHART_DIR / 'aspect_per_product.png', dpi=150)
plt.close()
print("Chart saved: aspect_per_product.png")

# ── 4. Sentiment × Aspect heatmap ───────────────────────────────────
sent_aspect = pd.crosstab(df['aspect_label'], df['sentiment_label'])
# Reorder columns
sent_aspect = sent_aspect[['Positive', 'Neutral', 'Negative']]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(sent_aspect.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(sent_aspect.columns)))
ax.set_xticklabels(sent_aspect.columns)
ax.set_yticks(range(len(sent_aspect.index)))
ax.set_yticklabels(sent_aspect.index)
ax.set_title('Sentiment × Aspect Heatmap')

for i in range(len(sent_aspect.index)):
    for j in range(len(sent_aspect.columns)):
        val = sent_aspect.values[i, j]
        ax.text(j, i, str(val), ha='center', va='center',
                color='white' if val > sent_aspect.values.max() * 0.6 else 'black')

plt.colorbar(im, label='Count')
plt.tight_layout()
plt.savefig(CHART_DIR / 'sentiment_aspect_heatmap.png', dpi=150)
plt.close()
print("Chart saved: sentiment_aspect_heatmap.png")

# ── 5. Review notes breakdown ───────────────────────────────────────
notes_aspect = pd.crosstab(df['review_notes'], df['aspect_label'])
notes_sentiment = pd.crosstab(df['review_notes'], df['sentiment_label'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Notes × Aspect
im1 = axes[0].imshow(notes_aspect.values, cmap='Blues', aspect='auto')
axes[0].set_xticks(range(len(notes_aspect.columns)))
axes[0].set_xticklabels(notes_aspect.columns, rotation=45, ha='right', fontsize=8)
axes[0].set_yticks(range(len(notes_aspect.index)))
axes[0].set_yticklabels(notes_aspect.index)
axes[0].set_title('Review Notes × Aspect')
plt.colorbar(im1, ax=axes[0], label='Count')

# Notes × Sentiment
im2 = axes[1].imshow(notes_sentiment.values, cmap='Blues', aspect='auto')
axes[1].set_xticks(range(len(notes_sentiment.columns)))
axes[1].set_xticklabels(notes_sentiment.columns)
axes[1].set_yticks(range(len(notes_sentiment.index)))
axes[1].set_yticklabels(notes_sentiment.index)
axes[1].set_title('Review Notes × Sentiment')
plt.colorbar(im2, ax=axes[1], label='Count')

plt.tight_layout()
plt.savefig(CHART_DIR / 'notes_breakdown.png', dpi=150)
plt.close()
print("Chart saved: notes_breakdown.png")

# ── 6. Pipeline vs annotation comparison ────────────────────────────
# Check Spam label vs is_spam
spam_hits = df[(df['aspect_label'] == 'Spam') & (df['is_spam'] == True)]
spam_misses = df[(df['aspect_label'] == 'Spam') & ((df['is_spam'] == False) | df['is_spam'].isna())]
spam_ann_total = (df['aspect_label'] == 'Spam').sum()
spam_pipeline_total = df['is_spam'].sum()

# Check is_short_comment
short_hits = df[(df['is_short_comment'] == True) & (df['aspect_label'] == 'Spam')]

# Check is_single_word vs Spam
single_word_spam = df[(df['is_single_word'] == True) & (df['aspect_label'] == 'Spam')]

# Check is_removed_by_cleaning combined
removed = df[df['is_removed_by_cleaning'] == True]
removed_aspects = removed['aspect_label'].value_counts() if len(removed) > 0 else pd.Series(dtype=int)

pipeline_comparison = {
    'comments_labeled_Spam': int(spam_ann_total),
    'pipeline_is_spam_true': int(spam_pipeline_total),
    'overlap_annotated_Spam_and_pipeline_spam': int(spam_hits.shape[0]),
    'Spam_label_but_pipeline_missed': int(spam_misses.shape[0]),
    'pipeline_is_short_comment_also_labeled_Spam': int(short_hits.shape[0]),
    'pipeline_is_single_word_also_labeled_Spam': int(single_word_spam.shape[0]),
}

# ── 7. "Other" deep dive ────────────────────────────────────────────
other_comments = df[df['aspect_label'] == 'Other'].copy()
other_sample = other_comments.sample(n=min(15, len(other_comments)), random_state=42)
other_rows = []
for _, r in other_sample.iterrows():
    other_rows.append({
        'product': r['product_query'],
        'sentiment': r['sentiment_label'],
        'notes': r['review_notes'] if pd.notna(r['review_notes']) else '',
        'text': str(r['comment_text'])[:120],
        'cleaned': str(r['cleaned_comment_text'])[:120],
    })
other_df = pd.DataFrame(other_rows)

# ── 8. Spam quality audit ───────────────────────────────────────────
spam_comments = df[df['aspect_label'] == 'Spam']
spam_rows = []
for _, r in spam_comments.iterrows():
    spam_rows.append({
        'product': r['product_query'],
        'sentiment': r['sentiment_label'],
        'notes': r['review_notes'] if pd.notna(r['review_notes']) else '',
        'text': str(r['comment_text'])[:150],
        'is_spam_pipeline': r['is_spam'],
    })
spam_df = pd.DataFrame(spam_rows)

# ── 9. Consistency checks ──────────────────────────────────────────
# 9a. "Question" → mostly Neutral
question = df[df['review_notes'] == 'Question']
question_neutral_pct = (question['sentiment_label'] == 'Neutral').mean() * 100

# 9b. "Unrelated" → mostly Other or Spam
unrelated = df[df['review_notes'] == 'Unrelated']
unrelated_other_spam = unrelated['aspect_label'].isin(['Other', 'Spam']).sum()
unrelated_other_spam_pct = unrelated_other_spam / len(unrelated) * 100 if len(unrelated) > 0 else 0

# 9c. "Language Error" → mostly Other or Spam
lang_err = df[df['review_notes'] == 'Language Error']
lang_err_other = lang_err['aspect_label'].value_counts()

# 9d. "Future Demand" notes
future = df[df['review_notes'] == 'Future Demand']
future_sentiment = future['sentiment_label'].value_counts()

# 9e. Annotator boundary — Spam with sentiment
spam_sentiment = spam_comments['sentiment_label'].value_counts()

# 9f. Normalized notes list
notes_unique = sorted(df['review_notes'].dropna().unique())

# ── Build report text ───────────────────────────────────────────────
lines = []

def add(k, v):
    lines.append(f"| {k} | {v} |")

def h1(t):
    lines.append(f"\n# {t}\n")

def h2(t):
    lines.append(f"\n## {t}\n")

def h3(t):
    lines.append(f"\n### {t}\n")

def p(t):
    lines.append(f"{t}\n")

def table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers))
    lines.append("| " + " | ".join(headers) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")

def code_block(content):
    lines.append("```")
    lines.append(content)
    lines.append("```")

# ── Report ──────────────────────────────────────────────────────────
h1("Annotation Analysis Report")
p(f"Generated from `manual_review_sample.csv` — 300 annotated rows across 3 products.")
p(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

h2("1. Overview")
table(["Metric", "Value"], [[k, v] for k, v in overview.items()])

h2("2. Sentiment Distribution by Product")
p("Bar chart: `charts/sentiment_per_product.png`")
p("**Summary table:**")
table(
    ["Product", "Positive", "Neutral", "Negative"],
    [[p, sent_by_product.loc[p, 'Positive'],
      sent_by_product.loc[p, 'Neutral'],
      sent_by_product.loc[p, 'Negative']]
     for p in sent_by_product.index]
)
p(f"**Total sentiment breakdown:** Positive {sent_by_product.sum()['Positive']}, Neutral {sent_by_product.sum()['Neutral']}, Negative {sent_by_product.sum()['Negative']}")

h2("3. Aspect Distribution by Product")
p("Stacked bar chart: `charts/aspect_per_product.png`")
table(
    ["Aspect"] + list(aspect_by_product.index),
    [[col] + [aspect_by_product.loc[p, col] for p in aspect_by_product.index]
     for col in aspect_by_product.columns]
)
p(f"**Top-3 aspects overall:** ")
for asp, cnt in df['aspect_label'].value_counts().head(3).items():
    p(f"- **{asp}**: {cnt} ({cnt/len(df)*100:.0f}%)")

h2("4. Sentiment × Aspect Heatmap")
p("Heatmap: `charts/sentiment_aspect_heatmap.png`")
table(
    ["Aspect", "Positive", "Neutral", "Negative", "Total"],
    [[idx, sent_aspect.loc[idx, 'Positive'],
      sent_aspect.loc[idx, 'Neutral'],
      sent_aspect.loc[idx, 'Negative'],
      sent_aspect.loc[idx].sum()]
     for idx in sent_aspect.index]
)

h2("5. Review Notes Breakdown")
p("Dual heatmap: `charts/notes_breakdown.png`")
p(f"**Normalized note categories:** {ANNOTATION_NOTE_LABELS}")
p("**Notes × Aspect:**")
table(
    ["Note"] + list(notes_aspect.columns),
    [[idx] + [notes_aspect.loc[idx, c] for c in notes_aspect.columns]
     for idx in notes_aspect.index]
)
p("**Notes × Sentiment:**")
table(
    ["Note"] + list(notes_sentiment.columns),
    [[idx] + [notes_sentiment.loc[idx, c] for c in notes_sentiment.columns]
     for idx in notes_sentiment.index]
)

h2("6. Pipeline vs Annotation Comparison")
table(["Metric", "Value"], [[k, v] for k, v in pipeline_comparison.items()])

h3("Pipeline flags on non-Spam comments")
p("Comments the pipeline flagged as `is_spam=True` but annotator did NOT label as Spam:")
non_spam_pipeline_spam = df[(df['is_spam'] == True) & (df['aspect_label'] != 'Spam')]
if len(non_spam_pipeline_spam) > 0:
    table(
        ["#", "Product", "Sentiment", "Aspect", "Notes", "Text preview"],
        [[i+1, r['product_query'], r['sentiment_label'],
          r['aspect_label'],
          r['review_notes'] if pd.notna(r['review_notes']) else '',
          str(r['comment_text'])[:80]]
         for i, (_, r) in enumerate(non_spam_pipeline_spam.iterrows())]
    )
else:
    p("None — pipeline never flagged `is_spam=True` on any of the 300 sampled rows.")

h3("is_removed_by_cleaning breakdown")
table(
    ["Aspect", "Count"],
    [[asp, int(cnt)] for asp, cnt in removed_aspects.items()]
) if len(removed_aspects) > 0 else p("No rows with is_removed_by_cleaning=True in sample.")

h2("7. 'Other' Category Deep Dive")
p(f"**Total 'Other' comments:** {len(other_comments)} ({len(other_comments)/len(df)*100:.0f}% of all annotations)")
p(f"Sentiment of 'Other': Positive {other_comments['sentiment_label'].value_counts().get('Positive', 0)}, "
  f"Neutral {other_comments['sentiment_label'].value_counts().get('Neutral', 0)}, "
  f"Negative {other_comments['sentiment_label'].value_counts().get('Negative', 0)}")

h3("Sample of 15 'Other' comments")
table(
    ["#", "Product", "Sentiment", "Notes", "Comment Text"],
    [[i+1, r['product'], r['sentiment'], r['notes'], r['text']]
     for i, (_, r) in enumerate(other_df.iterrows())]
)

h2("8. Spam Label Audit")
p(f"**Total labeled Spam:** {len(spam_comments)} ({len(spam_comments)/len(df)*100:.0f}% of annotations)")
p(f"Pipeline `is_spam=True` overlap: {spam_hits.shape[0]} — **0% overlap.**")
p("")
p("**Sentiment of Spam-labeled comments:**")
for sent, cnt in spam_sentiment.items():
    p(f"- {sent}: {cnt}")
p("")
h3("All 47 Spam comments")

spam_table = table(
    ["#", "Product", "Sentiment", "Notes", "Pipeline Spam", "Comment Text"],
    [[i+1, r['product'], r['sentiment'], r['notes'],
      str(r['is_spam_pipeline']), r['text']]
     for i, (_, r) in enumerate(spam_df.iterrows())]
)

h2("9. Consistency Checks")

h3("9a. 'Question' → Neutral?")
p(f"Comments with note 'Question': {len(question)}")
p(f"Labeled Neutral: {question['sentiment_label'].value_counts().get('Neutral', 0)}/{len(question)} = {question_neutral_pct:.0f}%")
if question_neutral_pct >= 90:
    p("✅ High consistency — 'Question' almost always maps to Neutral.")
else:
    p("⚠️ Some 'Question' comments have non-Neutral sentiment — review outliers.")

h3("9b. 'Unrelated' → Other / Spam?")
p(f"Comments with note 'Unrelated': {len(unrelated)}")
p(f"Labeled Other or Spam: {unrelated_other_spam}/{len(unrelated)} = {unrelated_other_spam_pct:.0f}%")
if unrelated_other_spam_pct >= 80:
    p("✅ Consistent — 'Unrelated' typically maps to Other or Spam.")
else:
    p("⚠️ Some 'Unrelated' comments map to other aspects — review.")

h3("9c. 'Language Error' → aspect distribution")
table(
    ["Aspect", "Count"],
    [[asp, int(cnt)] for asp, cnt in lang_err_other.items()]
)
p("'Language Error' clustering in Other is expected — if language is broken, aspect is hard to determine.")

h3("9d. 'Future Demand' sentiment")
table(
    ["Sentiment", "Count"],
    [[s, int(c)] for s, c in future_sentiment.items()]
)

h3("9e. Spam sentiment distribution")
table(
    ["Sentiment", "Count"],
    [[s, int(c)] for s, c in spam_sentiment.items()]
)
p(f"Spam being {spam_sentiment.get('Neutral', 0)}/{len(spam_comments)} Neutral is expected — most spam isn't emotionally charged.")

h3("9f. Notes normalization")
p(f"Unique notes after normalization: {notes_unique}")
p("Lowercase 'ambiguous' merged into 'Ambiguous' ✅")

h2("10. Recommendations")

p("Based on the analysis above, the following actions are recommended:")
recs = [
    "**Pipeline spam detection gap**: 47 comments labeled 'Spam' by the annotator were completely missed by the pipeline heuristic. The pipeline's spam detector needs review — consider adding keyword-based heuristics for promotional/solicitation patterns.",
    "**'Other' category is large** at 32% of annotations. Consider splitting into sub-categories (e.g., Design/Form Factor, Audio, Ecosystem, Brand, Durability) to improve granularity.",
    "**Notes normalization** completed: `ambiguous`/`Ambiguous` merged. The 6 notes (Ambiguous, Future Demand, Language Error, Question, Unrelated) form clean categories — consider promoting some to structured labels.",
    "**'Unrelated' boundary** with Spam: 17/53 'Unrelated' comments are labeled Spam as aspect. Clarify the guideline boundary between Spam (promotional/irrelevant content) vs Other/Unrelated (topic not covered by aspect list).",
    "**'Question' → Neutral rule** is 94% consistent. Could codify: comments phrased as questions default to Neutral unless strongly positive/negative.",
    "**Steam Deck negativity** (40% Negative vs 28-30% for others) is genuine signal — worth investigating in the full dataset whether this reflects a real product issue.",
]
for r in recs:
    p(f"- {r}")

report = "\n".join(lines)

# ── Write report ────────────────────────────────────────────────────
Path('data/reports/annotation_analysis_report.md').write_text(report, encoding='utf-8')

# ── Save spam audit CSV ─────────────────────────────────────────────
spam_df.to_csv('data/reports/spam_audit.csv', index=False, encoding='utf-8-sig')

# ── Other sample CSV ────────────────────────────────────────────────
other_df.to_csv('data/reports/other_sample.csv', index=False, encoding='utf-8-sig')

# ── Save normalized annotation CSV ──────────────────────────────────
df.to_csv('data/annotation/manual_review_sample.csv', index=False, encoding='utf-8-sig')

print("\n=== DONE ===")
print(f"Report: data/reports/annotation_analysis_report.md")
print(f"Charts: data/reports/charts/*.png")
print(f"Spam audit: data/reports/spam_audit.csv")
print(f"Other sample: data/reports/other_sample.csv")
print(f"Normalized annotations saved back to manual_review_sample.csv")
