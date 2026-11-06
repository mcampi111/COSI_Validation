#!/usr/bin/env python3
"""
COSI Paper 1: need-category analyses, rebuilt from the Methods of the manuscript.

  1. Rebuild the final sample (one questionnaire per patient):
       age outside 18-99 -> questionnaire excluded
       patients with a screening need -> all their questionnaires excluded
       earliest eligible questionnaire per patient (ties: most speech tests)
     and check it against the flow chart (24,570 patients, 6,011 "Other").
  2. Table S6  distribution of first-ranked needs (+ "Other" corrected)
  3. Table S7  age, DI, FA by predefined category: ANOVA, eta2, Kruskal-Wallis, Tukey
  4. Table S8  sex x category: chi2, Cramer's V, Haberman residuals, BH (with/without Other)
  5. Figure 3A bar chart
  6. Local export of the first-ranked "Other" answers of the final sample
     (for the thematic tagging; this file stays on the Mac, never upload it)

Prints and saves AGGREGATED results only, except the local export in step 6.

Usage (Mac):
  source ~/venvs/general/bin/activate
  cd ~/Desktop/Postdoc_Pasteur/Perrine/COSI2/data
  python cosi_categories.py
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRECISION, RECALL = 0.830, 0.941          # "Other" detection, validation 5 Nov 2026

CODE_MAP = {0: "Other", 1: "Screening", 3: "Conversation with group in quiet",
            4: "Conversation with 1 or 2 in quiet", 6: "Television/radio at normal volume",
            7: "Conversation with group in noise", 8: "Increased social contact",
            9: "Church or meeting", 10: "Feel embarrassed or stupid",
            11: "Unfamiliar speaker on phone"}

FR_MAP = [  # (normalised French pattern, English label); order matters
    (r"^autre", "Other"),
    (r"depistage", "Screening"),
    (r"1 ou 2.*bruit", "Conversation with 1 or 2 in noise"),
    (r"1 ou 2.*silence", "Conversation with 1 or 2 in quiet"),
    (r"groupe.*bruit", "Conversation with group in noise"),
    (r"groupe.*silence", "Conversation with group in quiet"),
    (r"television|radio", "Television/radio at normal volume"),
    (r"non familier", "Unfamiliar speaker on phone"),
    (r"familier", "Familiar speaker on phone"),
    (r"sonner", "Hear phone ring from another room"),
    (r"sonnette|frapper", "Hear front door bell or knock"),
    (r"trafic", "Hear traffic"),
    (r"contact social", "Increased social contact"),
    (r"embarrass|stupide", "Feel embarrassed or stupid"),
    (r"exclu|mis a l.ecart", "Feeling left out"),
    (r"contrarie|colere|fache", "Feeling upset or angry"),
    (r"eglise|reunion", "Church or meeting"),
]

FIX = {  # file Label -> true classifier category (verified by text match with result_Q1, 6 Nov 2026)
    "Augmenter le contact social": "Conversation en groupe dans le bruit",
    "Conversation en groupe dans le bruit": "Television/radio au volume normal",
    "Eglise ou reunion": "Augmenter le contact social",
    "Interlocuteur familier au telephone": "Interlocuteur non familier au telephone",
    "Interlocuteur non familier au telephone": "Se sentir embarrasse ou stupide",
    "Se sentir embarrasse ou stupide": "Eglise ou reunion",
}

DI_MAP = [(r"beaucoup mieux|much better", 3), (r"un peu mieux|leger|slightly", 1),
          (r"pire|moins bien|worse", -1), (r"pas de diff|aucune|no diff|identique|meme", 0),
          (r"mieux|better", 2)]
FA_MAP = [(r"toujours|almost always", 3), (r"souvent|plupart|most of", 2),
          (r"parfois|moitie|half", 1), (r"occasion", 0),
          (r"rarement|jamais|hardly", -1)]

SPEECH = {"SIQ_pre": "SRT_SPIQ_B_NA_1", "SIQ_post": "SRT_SPIQ_B_A_2",
          "SIN_pre": "SNR_SPIN_B_NA_1", "SIN_post": "SNR_SPIN_B_A_2"}
RANGE = {"SIQ": (0, 120), "SIN": (-10, 20)}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def map_by_patterns(series, patterns, what):
    out = {}
    for v in series.dropna().unique():
        if isinstance(v, (int, float, np.integer, np.floating)) and what in ("DI", "FA"):
            out[v] = float(v)
            continue
        n = norm(v)
        hit = next((lab for pat, lab in patterns if re.search(pat, n)), None)
        out[v] = hit
    unknown = [k for k, v in out.items() if v is None]
    if unknown:
        sys.exit(f"STOP: unmapped {what} values {unknown}.")
    if what in ("DI", "FA") and len(set(out.values())) < len(out):
        sys.exit(f"STOP: two {what} labels share a score: {out}.")
    return series.map(out)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def check(label, got, expected):
    flag = "OK" if got == expected else "DIFFERENT"
    print(f"  {label:<45} {got:>8,}   manuscript {expected:>8,}   {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="COSI_EVOLUTION_250703.xlsx")
    ap.add_argument("--labeled", default="COSI_EVOLUTION_by_need_labeled_trained.xlsx")
    ap.add_argument("--out", default="cosi_categories_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ load
    xl = pd.ExcelFile(a.labeled)
    print("Sheets in labelled file:", xl.sheet_names)
    need_sheets = [s for s in xl.sheet_names if re.search(r"need\s*\d", s, re.I)]
    q1 = pd.read_excel(xl, sheet_name="need1")
    lab_col = next((c for c in q1.columns if c.lower() in ("label", "labels", "flat_pred", "prediction")), None)
    if lab_col is None:
        sys.exit(f"STOP: no label column in need1. Columns: {list(q1.columns)}")
    main_df = pd.read_excel(a.main)
    demo = main_df[["CLE", "GENRE", "AGE", "PRIMO_RENEW"]].drop_duplicates("CLE")
    df = q1.drop(columns=[c for c in ("GENRE", "AGE", "PRIMO_RENEW") if c in q1.columns]) \
           .merge(demo, on="CLE", how="left")
    print(f"need1 rows: {len(df):,}; label column: {lab_col!r}; "
          f"demographics matched: {df['GENRE'].notna().sum():,}\n")

    # ------------------------------------------------------------- recoding
    def to_cat(s):
        if pd.api.types.is_numeric_dtype(s):
            bad = set(s.dropna().unique()) - set(CODE_MAP)
            if bad:
                sys.exit(f"STOP: unknown numeric codes {sorted(bad)}")
            return s.map(CODE_MAP)
        return map_by_patterns(s.replace(FIX), FR_MAP, "category")

    df["cat"] = to_cat(df[lab_col])
    df["DI"] = map_by_patterns(df["Degre_amelioration"], DI_MAP, "DI")
    df["FA"] = map_by_patterns(df["Aptitudes_Finales_"], FA_MAP, "FA")
    for col, sc in (("Degre_amelioration", "DI"), ("Aptitudes_Finales_", "FA")):
        print(f"{sc} recoding:")
        print(df.groupby(col, dropna=True)[sc].first().to_string(), "\n")

    # Mapping sanity check: share of answers mentioning TV / telephone / group, per category
    txt = df["OPEN_ANSWER"].astype(str).map(norm)
    chk = pd.DataFrame({"cat": df["cat"],
                        "tv": txt.str.contains(r"\btv\b|tele(?!phon)|televis|radio"),
                        "phone": txt.str.contains(r"telephon|portable"),
                        "group": txt.str.contains(r"groupe|plusieurs|repas|famille")})
    print("Mapping check (share of answers mentioning...):")
    print((chk.groupby("cat")[["tv", "phone", "group"]].mean() * 100).round(1)
          .assign(n=chk["cat"].value_counts()).to_string(), "\n")

    for k, c in SPEECH.items():
        lo, hi = RANGE[k[:3]]
        v = pd.to_numeric(df[c], errors="coerce")
        df[k] = v.where(v.between(lo, hi))
    df["n_speech"] = df[list(SPEECH)].notna().sum(axis=1)
    df["women"] = np.where(df["GENRE"].isna(), np.nan,
                           df["GENRE"].astype(str).str.upper().str.startswith("F").astype(float))

    # ------------------------------------------------------------- selection
    print("=== Selection (check against the flow chart) ===")
    check("questionnaires extracted", len(df), 39642)
    check("patients extracted", df["CUSTOMER_CODE"].nunique(), 25098)
    age_bad = df["AGE"].notna() & ~df["AGE"].between(18, 99)
    pts_before = df["CUSTOMER_CODE"].nunique()
    d1 = df[~age_bad]
    check("excluded for age: questionnaires", int(age_bad.sum()), 391)
    check("excluded for age: patients", pts_before - d1["CUSTOMER_CODE"].nunique(), 220)

    # screening: try "need1 only" and "any need" and keep the one matching 480 / 308
    scr_need1 = set(d1.loc[d1["cat"] == "Screening", "CUSTOMER_CODE"])
    scr_any = set(scr_need1)
    for s in need_sheets:
        if s == "need1":
            continue
        other = pd.read_excel(xl, sheet_name=s)
        lc = next((c for c in other.columns if c.lower() in ("label", "labels", "flat_pred", "prediction")), None)
        if lc is None:
            continue
        oc = to_cat(other[lc])
        keep = other["CLE"].isin(d1["CLE"]) if "CLE" in other else other["CUSTOMER_CODE"].isin(d1["CUSTOMER_CODE"])
        scr_any |= set(other.loc[keep & (oc == "Screening"), "CUSTOMER_CODE"])
    options = {}
    for name, pts in (("need1 only", scr_need1), ("any need", scr_any)):
        nq = int(d1["CUSTOMER_CODE"].isin(pts).sum())
        options[name] = (nq, len(pts))
        print(f"  screening rule '{name}': {nq:,} questionnaires, {len(pts):,} patients "
              f"(manuscript 480 / 308)")
    rule = next((k for k, v in options.items() if v == (480, 308)), "need1 only")
    scr = scr_need1 if rule == "need1 only" else scr_any
    print(f"  -> using '{rule}'")
    d2 = d1[~d1["CUSTOMER_CODE"].isin(scr)]
    check("eligible questionnaires", len(d2), 38771)
    check("eligible patients", d2["CUSTOMER_CODE"].nunique(), 24570)

    final = (d2.sort_values(["CUSTOMER_CODE", "QUESTIONNAIRE_DATE", "n_speech", "CLE"],
                            ascending=[True, True, False, True])
               .drop_duplicates("CUSTOMER_CODE"))
    N = len(final)
    check("final sample, patients", N, 24570)
    check("final sample, 'Other'", int((final["cat"] == "Other").sum()), 6011)
    print(f"  {'mean DI (n)':<45} {final['DI'].mean():.2f} ({final['DI'].notna().sum():,})   manuscript 2.25 (24,343)")
    print(f"  {'mean FA (n)':<45} {final['FA'].mean():.2f} ({final['FA'].notna().sum():,})   manuscript 2.19 (21,938)")
    print(f"  {'SIQ post n':<45} {final['SIQ_post'].notna().sum():,}   manuscript 8,563")
    print(f"  {'women (%)':<45} {int(final['women'].sum()):,} ({100*final['women'].mean():.1f})   manuscript 12,975 (52.8)\n")

    # ------------------------------------------------------------- Table S6
    vc = final["cat"].value_counts()
    s6 = pd.DataFrame({"n": vc, "pct": (100 * vc / N).round(1)})
    pre = [c for c in s6.index if c != "Other"]
    s6 = pd.concat([s6.loc[pre], s6.loc[["Other"]]])
    s6.to_csv(out / "TableS6_categories.csv")
    o = vc.get("Other", 0) / N
    print("=== Table S6: first-ranked needs (n = %s patients) ===" % f"{N:,}")
    print(s6.to_string())
    print(f"Other corrected = {100*o:.1f}% x {PRECISION}/{RECALL} = {100*o*PRECISION/RECALL:.1f}%\n")

    # ------------------------------------------------------------- Table S7
    P = final[final["cat"].isin(pre)]
    cats = list(pre)
    print(f"=== Table S7: predefined categories only (n = {len(P):,}) ===")
    rows, tukey_rows = [], []
    for var in ("AGE", "DI", "FA"):
        groups = [P.loc[P["cat"] == c, var].dropna().to_numpy() for c in cats]
        F, pF = stats.f_oneway(*groups)
        allv = np.concatenate(groups)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
        eta2 = ss_b / ((allv - allv.mean()) ** 2).sum()
        H, pH = stats.kruskal(*groups)
        dfb, dfw = len(groups) - 1, len(allv) - len(groups)
        for c, g in zip(cats, groups):
            rows.append({"var": var, "cat": c, "mean": round(g.mean(), 2),
                         "sd": round(g.std(ddof=1), 2), "n": len(g)})
        print(f"{var}: F({dfb}, {dfw:,}) = {F:.2f}, p {fmt_p(pF)}, eta2 = {eta2:.3f}; "
              f"Kruskal-Wallis H = {H:.1f}, p {fmt_p(pH)}")
        tk = stats.tukey_hsd(*groups)
        ci = tk.confidence_interval(0.95)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                d = groups[i].mean() - groups[j].mean()
                hi_i, lo_i = (i, j) if d > 0 else (j, i)
                lo, up = ci.low[hi_i, lo_i], ci.high[hi_i, lo_i]
                tukey_rows.append({"var": var, "higher": cats[hi_i], "lower": cats[lo_i],
                                   "diff": round(abs(d), 2), "ci_low": round(lo, 2),
                                   "ci_high": round(up, 2), "p_adj": tk.pvalue[i, j]})
    s7 = pd.DataFrame(rows).pivot(index="cat", columns="var")
    s7 = s7.reindex(cats)
    print(s7.to_string(), "\n")
    s7.to_csv(out / "TableS7A_by_category.csv")
    tk = pd.DataFrame(tukey_rows)
    tk_sig = tk[tk["p_adj"] < 0.05].sort_values(["var", "diff"], ascending=[True, False])
    for var in ("AGE", "DI", "FA"):
        n_sig = (tk_sig["var"] == var).sum()
        print(f"Tukey {var}: {n_sig} of {len(cats)*(len(cats)-1)//2} pairs significant")
        print(tk_sig[tk_sig["var"] == var].assign(p_adj=lambda d: d.p_adj.map(fmt_p))
              .drop(columns="var").to_string(index=False), "\n")
    tk.to_csv(out / "TableS7B_tukey_all_pairs.csv", index=False)

    # ------------------------------------------------------------- Table S8
    def chi_block(D, label):
        D = D[D["women"].notna()]
        tab = pd.crosstab(D["cat"], D["women"].astype(bool))
        tab = tab.reindex([c for c in s6.index if c in tab.index])
        chi2, p, dof, E = stats.chi2_contingency(tab.values, correction=False)
        n = tab.values.sum()
        V = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        r = tab.sum(1).values[:, None]
        c = tab.sum(0).values[None, :]
        adj = (tab.values - E) / np.sqrt(E * (1 - r / n) * (1 - c / n))
        res_w = adj[:, list(tab.columns).index(True)]
        p_res = 2 * stats.norm.sf(np.abs(res_w))
        q = bh(p_res)
        print(f"--- {label}: chi2({dof}) = {chi2:.1f}, p {fmt_p(p)}, Cramer's V = {V:.3f}, n = {n:,}")
        blk = pd.DataFrame({"women_n": tab[True], "women_pct": (100 * tab[True] / tab.sum(1)).round(1),
                            "men_n": tab[False], "residual_women": res_w.round(1),
                            "p_FDR": [fmt_p(x) for x in q]}, index=tab.index)
        print(blk.to_string(), "\n")
        return blk
    print("=== Table S8: sex x need category ===")
    chi_block(P, "Predefined categories").to_csv(out / "TableS8_predefined.csv")
    chi_block(final, "Including Other").to_csv(out / "TableS8_with_other.csv")

    # ------------------------------------------------------------- Figure 3A
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(s6) + 0.8))
        y = np.arange(len(s6))[::-1]
        colors = ["black" if c == "Other" else "#8c8c8c" for c in s6.index]
        ax.barh(y, s6["n"], color=colors, height=0.7)
        for yi, (n_, p_) in zip(y, zip(s6["n"], s6["pct"])):
            ax.text(n_ + s6["n"].max() * 0.01, yi, f"{n_:,} ({p_}%)", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(s6.index)
        ax.set_xlabel("Patients (n)")
        ax.set_xlim(0, s6["n"].max() * 1.22)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(direction="out")
        fig.tight_layout()
        fig.savefig(out / "Figure3A_categories_1patient.pdf")
        fig.savefig(out / "Figure3A_categories_1patient.png", dpi=200)
        print(f"Figure 3A saved in {out}/")
    except Exception as e:
        print("Figure 3A not drawn:", e)

    # ------------------------------------------------------------- export for tagging
    exp = final.loc[final["cat"] == "Other", ["OPEN_ANSWER"]].rename(columns={"OPEN_ANSWER": "OPEN_ANSWER"})
    exp["Label"] = "Autres"
    exp.to_csv(out / "LOCAL_ONLY_other_q1_final_sample.csv", index=False)
    print(f"Local export for thematic tagging: {out}/LOCAL_ONLY_other_q1_final_sample.csv "
          f"({len(exp):,} answers). Do NOT upload this file.")


if __name__ == "__main__":
    main()
