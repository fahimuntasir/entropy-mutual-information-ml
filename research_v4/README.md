# Research V4 — Robustness and statistical analysis

**Independent exploratory research; not a published paper.** This directory adds analyses to Research V3. Existing V1, V2, V3 and `main` files are unchanged.

## Protocol

- Datasets: Breast Cancer Wisconsin Diagnostic (569×30; scikit-learn/UCI) and Ionosphere (351×34; V3 attributed UCI copy).
- Selectors: all features, absolute Pearson correlation, marginal plug-in MI, and mRMR-MID.
- Feature budgets: **3, 5, 10, 15**. MI/mRMR quantile bins: **3, 5, 10**.
- Models: standardized logistic regression and random forest (120 trees), same settings as V3.
- Validation: shared **5-fold stratified cross-validation, repeated 3 times** (seed 20261116); training-fold-only feature selection and quantile thresholds. Metrics: accuracy, macro-F1, log loss in **bits**, Brier score; feature-selection Jaccard overlap is descriptive.
- Baseline verification: compare 5-feature, 5-bin conditions and controls with **all 240 original V3 fold outcomes**, absolute error ≤1e-12.
- Primary statistical contrast at k=5, 5 bins: paired mRMR-vs-MI differences, approximate **Nadeau–Bengio corrected repeated-CV t** (correction factor `1/15 + mean(test_count/train_count)`) and **Holm adjustment for 8 tests** (two datasets × two models × two metrics).
- Other feature-count and bin-count comparisons are **descriptive sensitivity analyses**, not separately significance-tested.

## Results from verified local execution

**1,740 held-out evaluations** across 29 selector configurations, two datasets, two models and 15 repeated-CV folds. Six V4 unit tests passed; all 240 matching V3 rows reproduced to floating-point precision.

At 5 MI bins, logistic regression accuracy for 3/5/10/15 selected features:

| Dataset | MI (%) | mRMR (%) |
|---|---|---|
| Breast Cancer | 92.62 / 94.67 / 95.02 / 94.97 | 95.43 / 96.25 / 97.60 / 97.66 |
| Ionosphere | 80.05 / 83.28 / 85.56 / 87.84 | 87.94 / 86.99 / 87.37 / 86.79 |

For Ionosphere with 5 selected features and logistic regression, using 3/5/10 bins: MI accuracy **86.98 / 83.28 / 82.33%** and mRMR **87.65 / 86.99 / 88.03%**. MI bin settings materially affect results.

At the preselected k=5, 5-bin comparison **no one of eight corrected tests passed Holm-adjusted p<0.05**. All corrected 95% intervals include zero. Apparent advantages are descriptive only.

For k=5 and 5 bins, selected-feature Jaccard overlap: Breast Cancer MI **1.000** vs mRMR **0.684**, Ionosphere MI **0.829** vs mRMR **0.580**. Stability does not necessarily correlate with predictive performance.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s research_v3/tests -v
python -m unittest discover -s research_v4/tests -v
python research_v3/real_world_validation.py
python research_v4/robustness_analysis.py
```

The V3 command regenerates the baseline `research_v3/outputs/fold_results.csv` for the automatic V4 cross-check. V4 writes only under `research_v4/outputs`: 1,740 per-fold scores, summaries, statistical comparisons, stability, metadata and nine PNG figures. The GitHub Actions workflow runs these checks and provides a downloadable complete output artifact.

Dataset sources and attribution: [UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [UCI Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere), and `research_v3/README.md`.

## Limitations

Two datasets do not establish broad generalizability. Repeated CV folds overlap and the corrected t test remains approximate. Comparing many feature budgets or bins post hoc can overfit the evaluation, so any selected settings need nested CV or independent external validation. The program does not implement a new algorithm and is not yet a journal submission.
