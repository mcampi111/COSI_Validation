# The French COSI in routine hearing-aid practice

Analysis code for the paper:

> Morvan P., Campi M., Peters G. W., Thai-Van H. *Patient-reported benefit, speech perception and coverage of the COSI taxonomy in a nationwide French cohort of hearing-aid users* (working title). Submitted to the *International Journal of Audiology*.

The paper and its Supplementary Information will be added to the `paper` folder after publication.

## Overview

The Client Oriented Scale of Improvement (COSI) asks patients to name, in their own words, the listening needs in which they most want to improve, and then to rate the benefit obtained with their hearing aids. This study used routinely collected data from 24,570 adults fitted with hearing aids in 780 centres of one national hearing-care network in France (2024), keeping one questionnaire per patient.

The study had three aims:

1. To test whether perceived benefit (COSI Degree of Improvement and Final Ability) is associated with speech perception in quiet and in noise, and whether it varies with patient characteristics.
2. To describe the measurement properties of the French COSI in routine practice: score distribution and ceiling effect, agreement between the two subscores, and internal consistency (indicative).
3. To measure how completely the 16 predefined COSI categories capture the needs that patients express, using a French language model (CamemBERT) to classify the free-text needs and a rule-based thematic analysis of the needs classified as "Other".

## Repository structure

```
code/
  classify_cosi_evolution.py   classification of the free-text needs with the trained CamemBERT model
  submit_cosi_classify.sh      SLURM script to run the classification on a computing cluster
  Validation_Cosi.py           main analyses: perceived benefit, speech perception, measurement properties
  cosi_categories.py           final sample and need-category analyses (Figure 3A, Tables S6 to S8)
  cosi_checks.py               validation of the classifier for the "Other" category and check of the category mapping
  cosi_other_analysis.py       rule-based thematic tagging of the "Other" needs; the lexicon of each theme is defined here
  debug_cosi.py                helper to inspect the structure of the input files
paper/                         paper and Supplementary Information (after publication)
```

## Pipeline

1. **Classification of the needs.** `classify_cosi_evolution.py` applies the trained CamemBERT model to each free-text need and writes the predicted category. The code to category mapping is fixed in the script and was verified against the model output and a hand-labelled validation set; the script stops if the model returns a code outside this mapping.
2. **Validation of the classifier.** `cosi_checks.py` compares the predictions with an independent hand-labelled set and reports precision, recall and F1 for the "Other" category with 95% confidence intervals (Wilson intervals for precision and recall, bootstrap for F1).
3. **Final sample and category analyses.** `cosi_categories.py` applies the selection criteria (age 18 to 99 years, exclusion of patients with a screening need, earliest eligible questionnaire per patient), checks each step against the flow chart of the paper, and computes the distribution of first-ranked needs, the comparisons of age and perceived benefit across categories (ANOVA, Kruskal-Wallis, Tukey) and the association between sex and need category (chi-square, Cramer's V, adjusted residuals with Benjamini-Hochberg correction).
4. **Main analyses.** `Validation_Cosi.py` computes the correlations between perceived benefit, age and speech perception, the sensitivity analyses, and the measurement properties (ceiling effect, paired comparison, ICC(2,1), Bland-Altman, Cronbach's alpha).
5. **Themes within "Other".** `cosi_other_analysis.py` tags the first-ranked "Other" needs of the final sample with case- and accent-insensitive regular expressions. A need can receive several themes.

All scripts print and save aggregated results only.

## Requirements

Python 3.9 or later with `pandas`, `numpy`, `scipy`, `matplotlib` and `openpyxl`. The classification step also requires `torch` and `transformers`.

```bash
pip install pandas numpy scipy matplotlib openpyxl torch transformers
```

## Data availability

The data are pseudonymised clinical records collected in routine care by a private hearing-care network. Their processing was approved by the French data protection authority (CNIL, approval number HGT/MLR/AR2410416). They cannot be shared, and no data file is included in this repository. The scripts expect the input files to be available locally; file names and paths are set at the top of each script.

## Funding

This research was funded by a grant from the Fondation Pour l'Audition (FPA) to Hung Thai-Van and Paul Avan (FPA IDA09 and FPA IDA10).

## Citation

If you use this code, please cite the paper (reference to be completed after publication):

```
@article{morvan_cosi,
  title   = {...},
  author  = {Morvan, Perrine and Campi, Marta and Peters, Gareth W. and Thai-Van, Hung},
  journal = {International Journal of Audiology},
  year    = {...}
}
```

## License

The code is released under the MIT License (see `LICENSE`).
