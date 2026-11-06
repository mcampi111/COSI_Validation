#!/usr/bin/env python3
"""
COSI classifier checks for Perrine's two comments on Paper 1.

  1. TV question: which of the COSI categories the classifier ever predicts,
     and where answers mentioning TV / radio end up.
  2. 95% CI for the "Other" (Autres) precision, recall and F1.

Prints and saves AGGREGATED results only (counts, rates, CIs).
No answer text and no row-level data are printed or written.

Usage (on the Mac, in the folder with the files):
  source ~/venvs/general/bin/activate
  python cosi_checks.py --gold COSI_LABELS_3000.xlsx --pred result_Q1.xlsx

If a column cannot be guessed, the script stops and lists the column names;
rerun with --text-col / --gold-col / --pred-col.
"""
import argparse
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

TV_REGEX = r"\btv\b|t[ée]l[ée]|radio"


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def load(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def pick(df, name, candidates, what):
    if name:
        if name not in df.columns:
            sys.exit(f"Column {name!r} not found in {what}. Columns: {list(df.columns)}")
        return name
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"Could not guess the {what} column. Columns: {list(df.columns)}\n"
             f"Rerun with the matching --*-col option.")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def prf(y_true, y_pred):
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p + r else np.nan
    return tp, fp, fn, p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="hand-labelled file (e.g. COSI_LABELS_3000.xlsx)")
    ap.add_argument("--pred", required=True, help="classifier output file (e.g. result_Q1.xlsx)")
    ap.add_argument("--text-col")
    ap.add_argument("--gold-col")
    ap.add_argument("--pred-col")
    ap.add_argument("--pred-text-col", help="text column in the prediction file, if different")
    ap.add_argument("--other-label", default="Autres")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    text_cands = ["Text", "text", "TEXT", "Réponse", "Reponse", "REPONSE", "reponse",
                  "Verbatim", "verbatim", "Answer", "answer", "QUESTION_COSI", "COSI"]
    gold = load(a.gold)
    pred = load(a.pred)
    tg = pick(gold, a.text_col, text_cands, "gold text")
    tp_ = pick(pred, a.pred_text_col or a.text_col, text_cands, "prediction text")
    gc = pick(gold, a.gold_col, ["Label", "label", "LABEL", "gold", "Gold", "category", "Category"], "gold label")
    pc = pick(pred, a.pred_col, ["prediction", "Prediction", "pred", "predicted", "Predicted",
                                 "label_pred", "Label_pred", "Label", "label", "category"], "predicted label")
    print(f"Columns used: gold text={tg!r}, gold label={gc!r}, pred text={tp_!r}, pred label={pc!r}\n")

    # ---------- 0. Numeric prediction codes: recover code -> category ----------
    if pd.api.types.is_numeric_dtype(pred[pc]):
        print("=== Predictions are numeric codes: recovering the code -> category mapping ===")
        if "Labels" in pred.columns and "Labels" != pc:
            ct = pd.crosstab(pred[pc], pred["Labels"].astype(str))
            print("Cross-check with the 'Labels' column of the prediction file "
                  "(most frequent value per code, share):")
            for code, row in ct.iterrows():
                print(f"  {code:>3} -> {row.idxmax()!s:<55} {row.max() / row.sum():.2f}")
            print()
        gk = gold[[tg, gc]].assign(key=gold[tg].map(norm))
        pk = pred[[tp_, pc]].assign(key=pred[tp_].map(norm)).drop_duplicates("key")
        mm = gk.merge(pk[["key", pc]], on="key", how="inner")
        ct = pd.crosstab(mm[pc], mm[gc])
        mapping = {}
        print(f"From the matched set (n = {len(mm)}): majority gold category per code")
        for code, row in ct.iterrows():
            mapping[code] = row.idxmax()
            print(f"  {code:>3} -> {row.idxmax():<55} purity {row.max() / row.sum():.2f}  (n = {int(row.sum())})")
        dup = pd.Series(mapping).value_counts()
        dup = dup[dup > 1]
        if len(dup):
            print("WARNING: several codes map to the same category:", dict(dup))
        unseen = sorted(set(pred[pc].dropna().unique()) - set(mapping))
        if unseen:
            print("WARNING: codes never seen in the matched set (left as numbers):", unseen)
        print("CHECK this mapping before trusting anything below.\n")
        pred[pc] = pred[pc].map(lambda v: mapping.get(v, str(v)))

    # ---------- 1a. Which categories exist and which are ever predicted ----------
    gold_counts = gold[gc].value_counts()
    pred_counts = pred[pc].value_counts()
    print("=== Categories in the hand-labelled set (n) ===")
    print(gold_counts.to_string(), "\n")
    print(f"=== Categories predicted on the full file (n = {len(pred)}) ===")
    print(pred_counts.to_string(), "\n")
    never = sorted(set(gold_counts.index) - set(pred_counts.index), key=str)
    print(f"Categories present in the gold set but NEVER predicted ({len(never)}):")
    for c in never:
        print(f"  {c}  (gold n = {gold_counts[c]})")
    print()
    pd.concat([gold_counts.rename("gold_n").rename(index=str),
               pred_counts.rename("pred_n_fullfile").rename(index=str)], axis=1) \
        .fillna(0).astype(int).to_csv("cosi_check_category_counts.csv")

    # ---------- 1b. Answers mentioning TV / radio on the full file ----------
    tv_mask = pred[tp_].astype(str).str.lower().str.contains(TV_REGEX, regex=True, na=False)
    print(f"=== Full file: answers mentioning TV/radio = {int(tv_mask.sum())} "
          f"({100 * tv_mask.mean():.1f}%) ===")
    tv_dist = pred.loc[tv_mask, pc].value_counts()
    print("Where the classifier puts them:")
    print((tv_dist.to_frame("n").assign(pct=lambda d: (100 * d.n / d.n.sum()).round(1))).to_string(), "\n")
    tv_dist.to_csv("cosi_check_tv_mentions_predicted.csv")

    # ---------- Match gold and predictions by normalised text ----------
    g = gold[[tg, gc]].copy()
    g["key"] = g[tg].map(norm)
    p = pred[[tp_, pc]].copy()
    p["key"] = p[tp_].map(norm)
    # Keep every hand-labelled row (repeated short answers are separate responses);
    # the classifier is deterministic, so identical texts share one prediction.
    n_dup_g = g.duplicated("key").sum()
    p = p.drop_duplicates("key")
    m = g.merge(p[["key", pc]], on="key", how="inner", suffixes=("_gold", "_pred"))
    gcol = gc + "_gold" if gc == pc else gc
    pcol = pc + "_pred" if gc == pc else pc
    print(f"=== Matched validation set: {len(m)} answers "
          f"(gold rows with a repeated text, all kept: {n_dup_g}) ===\n")

    conf = pd.crosstab(m[gcol], m[pcol], rownames=["gold"], colnames=["pred"])
    conf.to_csv("cosi_check_confusion_matrix.csv")

    tv_labels = [c for c in conf.index.union(conf.columns)
                 if re.search(TV_REGEX, str(c).lower())]
    if tv_labels:
        for c in tv_labels:
            if c in conf.index:
                print(f"Gold = {c!r} (n = {int(conf.loc[c].sum())}), predicted as:")
                row = conf.loc[c]
                print(row[row > 0].sort_values(ascending=False).to_string(), "\n")
            else:
                print(f"{c!r} is predicted but absent from the matched gold set.\n")
    else:
        print("No category label matches TV/radio in the matched set.\n")

    # ---------- 2. "Other" precision / recall / F1 with 95% CI ----------
    yt = (m[gcol] == a.other_label).to_numpy()
    yp = (m[pcol] == a.other_label).to_numpy()
    tp, fp, fn, P, R, F = prf(yt, yp)
    pl, ph = wilson(tp, tp + fp)
    rl, rh = wilson(tp, tp + fn)

    rng = np.random.default_rng(1)
    n = len(m)
    boots = []
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, n)
        boots.append(prf(yt[idx], yp[idx])[5])
    fl, fh = np.nanpercentile(boots, [2.5, 97.5])

    print(f"=== '{a.other_label}' detection on the matched set ===")
    print(f"gold {a.other_label} = {int(yt.sum())}, predicted {a.other_label} = {int(yp.sum())}")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}")
    print(f"Precision = {P:.3f}  95% CI [{pl:.3f}; {ph:.3f}]  (Wilson)")
    print(f"Recall    = {R:.3f}  95% CI [{rl:.3f}; {rh:.3f}]  (Wilson)")
    print(f"F1        = {F:.3f}  95% CI [{fl:.3f}; {fh:.3f}]  (bootstrap, {a.n_boot} resamples)")
    print(f"Overall accuracy (all categories) = {(m[gcol] == m[pcol]).mean():.3f}")

    print("\nSaved (aggregated only): cosi_check_category_counts.csv, "
          "cosi_check_tv_mentions_predicted.csv, cosi_check_confusion_matrix.csv")


if __name__ == "__main__":
    main()
