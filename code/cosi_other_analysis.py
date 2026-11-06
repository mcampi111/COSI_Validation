#!/usr/bin/env python3
"""
COSI 'Other' analysis  ->  word frequencies + bigrams + thematic histogram.
Run on the full dataset (any size, e.g. 300k rows):

    pip install pandas openpyxl matplotlib --break-system-packages
    python3 cosi_other_analysis.py your_file.xlsx

Outputs (next to the script):
    cosi_other_needs.png / .svg   thematic histogram (slide-ready)
    cosi_other_unigrams.csv       top words with counts and %
    cosi_other_bigrams.csv        top word pairs
    cosi_other_themes.csv         per-theme counts, % , and split by SEX / AGE band

Adjust COLUMN NAMES + OTHER_LABELS below if your file differs.
"""
import sys, re, unicodedata, csv
from collections import Counter

# ---------- CONFIG ----------
LABEL_COL = "Label"          # column holding the COSI category
TEXT_COL  = "OPEN_ANSWER"    # column holding the free text
SEX_COL   = "SEXE"           # optional; set to None if absent
AGE_COL   = "AGE"            # optional; set to None if absent
OTHER_LABELS = {"autres","autre","other"}
CHILD_AGE_MAX = 17           # rows with AGE <= this count as children
# ----------------------------

def load(path):
    if path.lower().endswith((".xlsx",".xls")):
        import pandas as pd
        df = pd.read_excel(path, dtype=str)
        return df.fillna("")
    import pandas as pd
    return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")

def sa(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

STOP=set("au aux avec ce ces dans de des du elle en et il je la le les leur lui ma mais me mes moi mon ne pas pour qu que qui sa se ses son sur ta te toi ton tu un une vos votre vous est sont avoir etre fait plus tres tout tous toute si car donc aussi sans mme mr jour quotidien chose peu assez bien mieux trop the and i to of".split())
INVAR={'fois','mois','temps','voix','moins','sons','gens','bruits','autres'}
def tok(t):
    t=sa(t.lower()); t=re.sub(r"[''`]"," ",t); out=[]
    for w in re.findall(r"[a-z]+",t):
        if w in STOP or len(w)<=2: continue
        if w not in INVAR and len(w)>3 and w.endswith('s') and not w.endswith('ss'): w=w[:-1]
        out.append(w)
    return out

# Thematic dictionaries (rule-based, descriptive). Edit freely.
THEMES = {
 "Tinnitus management":                        [r"acouph",r"acoufe",r"accouph",r"bourdonnement",r"sifflement",r"tinnitus"],
 "Device: discretion / aesthetics / format":   [r"discre",r"invisible",r"\bintra",r"esthet",r"\bpetit",r"rechargeab",r"\bpile",r"qui (ne )?se voi",r"contour"],
 "Weak / distant / masked voices":             [r"voix faible",r"voix basse",r"chuchot",r"parle.* (bas|doucement|faible)",r"de loin|eloign|a distance|lointain",r"masque",r"lecture labiale",r"sans voir",r"n.articul"],
 "Price / administrative":                     [r"100% sante|rac 0|reste a charge",r"\bmdph\b|\bmda\b|\bcmu\b|\bcss\b",r"devis",r"renouvel|remplac",r"mutuelle|prix|cher|abordable"],
 "Laterality / stereo / localization":         [r"stereo",r"binaural",r"reequilibr",r"localis",r"cone d.ombre",r"\bcros\b",r"baha",r"hemichamp",r"cote (droit|gauche)",r"deux oreilles",r"d.ou vient"],
 "Pediatric / school":                         [r"\benfant|\bfils|\bfille|\bclasse|ecole|college|lycee|amphi|\bcours|maitresse|professeur|orthophon|implant cochleaire|apprentissage|langage|diction|oralite|micro hf|roger"],
 "Environmental safety (cars, alarms)":        [r"securite|danger|alerte|alarme|detecteur|sonnette|sonnerie",r"voiture|train|metro|velo|conduit|permis|poids lourd|gps",r"signal"],
 "Theatre / cinema / lectures":                [r"theatre|cinema|spectacle|conference|amphi"],
 "Hyperacusis / listening fatigue / cognition":[r"hyperacous",r"bruits? fort|sons? fort|intoleran",r"fatigu",r"concentr|memoire|cognit|attention",r"reposer le cerveau|agress"],
 "Music / fine perception / nature":           [r"musiqu|piano|saxophone|cornemuse|cor de chasse|flute|chorale|chant|instrument|\bnote|violon|contrebasse|chef de choeur",r"oiseau|cigale|nature"],
 "Vertigo / balance / pathology":              [r"vertige|menier|equilibre postural|cholesteat|otospongiose|neurinome|surdite brusque|surdite brutale|congenital|cophos"],
}

def main(path):
    df = load(path)
    for c in (LABEL_COL,TEXT_COL):
        if c not in df.columns: sys.exit(f"Column '{c}' not found. Columns: {list(df.columns)}")
    mask = df[LABEL_COL].astype(str).str.strip().str.lower().isin(OTHER_LABELS)
    sub = df[mask]
    N=len(sub)
    print(f"'Other' rows: {N} / {len(df)} total")
    texts=[sa(str(t).lower()) for t in sub[TEXT_COL]]

    # unigrams + bigrams
    uni=Counter(); bi=Counter()
    for raw in sub[TEXT_COL]:
        toks=tok(str(raw)); uni.update(toks)
        for i in range(len(toks)-1): bi[(toks[i],toks[i+1])]+=1
    with open("cosi_other_unigrams.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["word","count","pct_of_rows"])
        for word,c in uni.most_common(40): w.writerow([word,c,round(100*c/N,2)])
    with open("cosi_other_bigrams.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["bigram","count"])
        for (a,b),c in bi.most_common(25): w.writerow([f"{a} {b}",c])

    # themes (multi-label) + sex/age split
    def numage(x):
        try: return float(str(x).replace(",","."))
        except: return None
    rows=[]
    for k,pats in THEMES.items():
        idx=[i for i,t in enumerate(texts) if any(re.search(p,t) for p in pats)]
        c=len(idx); row=[k,c,round(100*c/N,2)]
        if SEX_COL and SEX_COL in df.columns:
            sx=sub[SEX_COL].astype(str).str.strip().tolist()
            row += [sum(1 for i in idx if sx[i].upper().startswith("F")),
                    sum(1 for i in idx if sx[i].upper().startswith(("H","M")))]
        if AGE_COL and AGE_COL in df.columns:
            ag=[numage(a) for a in sub[AGE_COL].tolist()]
            row += [sum(1 for i in idx if ag[i] is not None and ag[i]<=CHILD_AGE_MAX),
                    sum(1 for i in idx if ag[i] is not None and ag[i]>CHILD_AGE_MAX)]
        rows.append(row)
    header=["theme","count","pct"]+(["n_F","n_H"] if SEX_COL in df.columns else [])+(["n_children","n_adults"] if AGE_COL in df.columns else [])
    rows_sorted=sorted(rows,key=lambda r:-r[1])
    with open("cosi_other_themes.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows_sorted)

    # chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    items=sorted(((r[0],r[1]) for r in rows), key=lambda x:x[1])
    labels=[k for k,_ in items]; vals=[v for _,v in items]; pct=[100*v/N for v in vals]
    INK="#1A2640"; BLUE="#2B6CB0"
    plt.rcParams.update({'font.size':11.5,'axes.edgecolor':INK,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    fig,ax=plt.subplots(figsize=(11.5,7))
    ax.barh(range(len(items)),pct,color=BLUE,edgecolor="white",height=0.72)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels)
    ax.set_xlabel("% of 'Other' free-text responses  (multi-label)")
    ax.set_title("What the COSI 'Other' category actually contains\n"
                 f"Recurring listening needs not modeled by the 16 standard categories  (n = {N})",
                 loc='left',fontweight='bold',color=INK,pad=14)
    for i,(v,c) in enumerate(zip(vals,pct)): ax.text(c+max(pct)*0.01,i,f"{c:.1f}%  (n={v})",va='center',fontsize=9,color=INK)
    ax.set_xlim(0,max(pct)*1.20)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='x',alpha=0.25)
    fig.text(0.01,0.01,"Rule-based thematic tagging (keyword/regex), descriptive. Categories are non-exclusive, so percentages do not sum to 100.",fontsize=7.5,color="#5a6472")
    plt.tight_layout(rect=[0,0.03,1,1])
    plt.savefig("cosi_other_needs.png",dpi=150); plt.savefig("cosi_other_needs.svg")
    print("Wrote: cosi_other_needs.png/.svg, cosi_other_unigrams.csv, cosi_other_bigrams.csv, cosi_other_themes.csv")

if __name__=="__main__":
    if len(sys.argv)<2: sys.exit("usage: python3 cosi_other_analysis.py <file.xlsx|csv>")
    main(sys.argv[1])
