"""Iterative self-training: pseudo-label with per-class thresholds, retrain, repeat."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from reviewradar.evaluation.aspect_evaluation import DistilBertAspectScorer, ASPECT_LABELS, evaluate_aspect

_BASELINE: dict = {
    "accuracy": 0.4667, "macro_f1": 0.2174,
    "per_label": {
        "battery":{"label":"Battery","recall":1.0,"f1":0.4,"support":1},
        "camera":{"label":"Camera","recall":0.0,"f1":0.0,"support":1},
        "competition":{"label":"Competition","recall":0.1667,"f1":0.25,"support":6},
        "display":{"label":"Display","recall":0.0,"f1":0.0,"support":1},
        "gaming":{"label":"Gaming","recall":0.0,"f1":0.0,"support":2},
        "hardware":{"label":"Hardware","recall":0.3333,"f1":0.5,"support":3},
        "other":{"label":"Other","recall":0.8571,"f1":0.5854,"support":14},
        "performance":{"label":"Performance","recall":0.0,"f1":0.0,"support":1},
        "price":{"label":"Price","recall":1.0,"f1":0.7273,"support":4},
        "purchase_intent":{"label":"Purchase Intent","recall":0.0,"f1":0.0,"support":3},
        "software":{"label":"Software","recall":0.0,"f1":0.0,"support":1},
        "spam":{"label":"Spam","recall":0.2857,"f1":0.3636,"support":7},
        "support":{"label":"Support","recall":0.0,"f1":0.0,"support":1},
    },
}

def build_eval_report(results, baseline, n_pseudo, n_rounds):
    lines = [f"# Self-Training Evaluation Report ({n_rounds} rounds)\n"]
    lines.append(f"Seed: 216 | Pseudo-labels added: {n_pseudo} | Total train: {216 + n_pseudo}\n")
    lines.append("| Metric | Oversampled (46.67%) | Self-Train | Delta |")
    lines.append("|---|---|---|---|")
    for m in ["accuracy","macro_precision","macro_recall","macro_f1"]:
        b_map = {"accuracy":0.4667,"macro_precision":0.2512,"macro_recall":0.2802,"macro_f1":0.2174}
        b = b_map[m]
        o = results.get(m,0)
        lines.append(f"| {m.replace('_',' ').title()} | {b:.2%} | {o:.2%} | {o-b:+.2%} |")
    lines.append("\n## Per-Class Recall\n")
    lines.append("| Aspect | Baseline Recall | Self-Train Recall | Delta | Support |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(ASPECT_LABELS):
        k = label.lower().replace(" ","_")
        b_r = baseline["per_label"][k]["recall"]
        o_r = results["per_label"].get(k,{}).get("recall",0)
        s = baseline["per_label"][k]["support"]
        lines.append(f"| {label} | {b_r:.1%} | {o_r:.1%} | {o_r-b_r:+.1%} | {s} |")
    preds = results.get("predictions",[])
    other_pct = sum(1 for p in preds if p=="Other")/max(len(preds),1)*100
    lines.append(f"\n## Other-Proportion\n")
    lines.append(f"Predictions classified as **Other**: {other_pct:.1f}%\n")
    return "\n".join(lines)

def main():
    out_dir = Path("data/reports/pseudolabel")
    out_dir.mkdir(parents=True,exist_ok=True)

    ann = pd.read_csv("data/annotation/manual_review_sample.csv")
    ann = ann[ann["aspect_label"].notna()&(ann["aspect_label"].str.strip()!="")].copy()
    gt = ann["aspect_label"].str.strip().tolist()
    texts = ann["cleaned_comment_text"].fillna("").tolist()
    ann_ids = set(ann["comment_id"])

    train_t,test_t,train_l,test_l = train_test_split(texts,gt,test_size=0.15,random_state=42,stratify=gt)
    train_t_seed,val_t,train_l_seed,val_l = train_test_split(train_t,train_l,test_size=0.15,random_state=42,stratify=train_l)
    test_df = pd.DataFrame({"cleaned_comment_text":test_t,"aspect_label":test_l})

    master = pd.read_csv("data/exports/reviewradar_master_raw.csv")
    unlabeled = master[~master["comment_id"].isin(ann_ids)].copy()

    print(f"Seed: {len(train_t_seed)} train, {len(val_t)} val, {len(test_t)} test")
    print(f"Unlabeled: {len(unlabeled)}")

    cur_t = list(train_t_seed)
    cur_l = list(train_l_seed)
    rem = unlabeled.copy()
    model_path = "models/distilbert_aspect_self_train"
    rounds_log = []

    for rnd in range(1,4):
        print(f"\n=== ROUND {rnd}: {len(cur_t)} samples ===")

        scorer = DistilBertAspectScorer()
        scorer.train(texts=cur_t,labels=cur_l,val_texts=val_t,val_labels=val_l,
                     output_dir=model_path,num_epochs=15,batch_size=16,lr=3e-5,use_class_weights=True)
        results = evaluate_aspect(test_df,scorer)
        print(f"  Test: acc={results['accuracy']:.2%}, mF1={results['macro_f1']:.2%}")

        if len(rem)==0:
            break

        text_unl = rem["cleaned_comment_text"].fillna("").tolist()
        scored = scorer.predict_with_scores(text_unl)

        max_per_class = 150
        cc = {}
        acc_t, acc_l, acc_idx = [], [], []
        for i,s in enumerate(scored):
            label, conf = s["aspect"], s["confidence"]
            thresh = 0.95 if label=="Other" else 0.85
            if conf < thresh:
                continue
            if cc.get(label,0) >= max_per_class:
                continue
            cc[label] = cc.get(label,0)+1
            acc_t.append(text_unl[i]); acc_l.append(label); acc_idx.append(i)

        print(f"  Accepted: {len(acc_t)}")
        for lab,cnt in sorted(cc.items()):
            print(f"    {lab}: {cnt}")

        if not acc_t:
            print("  No new labels. Stopping.")
            break

        cur_t.extend(acc_t); cur_l.extend(acc_l)
        accepted_ids = set(rem.iloc[acc_idx]["comment_id"])
        rem = rem[~rem["comment_id"].isin(accepted_ids)].copy()
        rounds_log.append({"round":rnd,"train_size":len(cur_t),"new":len(acc_t),
                           "acc":results["accuracy"],"mf1":results["macro_f1"]})

    # Final evaluation
    final = DistilBertAspectScorer(model_path=model_path)
    final_results = evaluate_aspect(test_df,final)

    n_pseudo = len(cur_t)-216
    print(f"\n=== FINAL ===")
    print(f"Accuracy: {final_results['accuracy']:.2%} (baseline: 46.67%)")
    print(f"Macro F1: {final_results['macro_f1']:.2%} (baseline: 21.74%)")
    print(f"Train: 216 seed + {n_pseudo} pseudo = {len(cur_t)}")

    report = build_eval_report(final_results,_BASELINE,n_pseudo,len(rounds_log))
    # Add progression
    report += "\n## Training Progression\n\n| Round | Train Size | New Pseudo | Accuracy | Macro F1 |\n|---|---|---|---|---|\n"
    for r in rounds_log:
        report += f"| {r['round']} | {r['train_size']} | {r['new']} | {r['acc']:.2%} | {r['mf1']:.2%} |\n"

    (out_dir/"self_training_report.md").write_text(report,encoding="utf-8")
    (out_dir/"self_training_results.json").write_text(json.dumps(final_results,indent=2,default=str),encoding="utf-8")
    print(f"Report: {out_dir/'self_training_report.md'}")
    print("Done.")

if __name__=="__main__":
    main()
