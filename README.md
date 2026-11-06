# Validation of the COSI in France
This repository is linked to the methodology developed in the paper with title **"TODO"**.  

The pdf for the paper is available at this url .... and provided in the folder "paper", where it is also possibile to find the Supplementary Materials.

## **Abstract**

**Background**: TODO

**Methods**: TODO

**Findings**:  TODO

**Interpretation**: TODO

**Funding**: This research was funded by a grant from Fondation Pour l’Audition (FPA) to Hung Thai-Van and Paul Avan, number FPA IDA09 and FPA IDA10.


## Contributions of the paper
The paper has the following contributions:
1. TODO
2. TODO


## Motivations For this Study

TODO

## Organization of the Repository
The repository is organized in the following folders:

```diff
+ 1) code
```
The code folder contains all the code used for the implementation of the methods and the results section, inlcuding tables and figures. We used Python and the details are given below.

1.  **classify_cosi_evolution.py** (with **submit_cosi_classify.sh**). Applies the trained CamemBERT model to the free-text COSI needs and writes the predicted category of each need.
2.  **Validation_Cosi.py**. Main analyses of the paper: perceived benefit and speech perception, measurement properties.
3.  **cosi_categories.py**. Rebuilds the final sample (one questionnaire per patient) and computes the need-category analyses: distribution of first-ranked needs (Figure 3A, Table S6), age and perceived benefit by category (Table S7), sex by category (Table S8).
4.  **cosi_checks.py**. Validation of the classifier for the "Other" category (precision, recall and F1 with 95% confidence intervals) and check of the category mapping.
5.  **cosi_other_analysis.py**. Rule-based thematic tagging of the "Other" answers; the lexicon of each theme is defined in this file.
6.  **debug_cosi.py**. Helper for inspecting the input files.

The clinical data are pseudonymised patient records and cannot be shared; no data file is included in this repository.

```diff
+ 3) paper 
```

The paper folder contains the following pdf:

1.  **Main Paper**. This pdf represents the main body of the paper.
2.  **Supplementary Information**. This pdf provides all the Supplementary Information.



## Cite

If you use this code in your project, please cite:

@article{..,
  title={..},
  author={..},
  journal={..},
  year={..}
}
