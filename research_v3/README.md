# Research V3 — real-world feature selection validation

**Status:** exploratory independent research, not a submitted or published paper. All V3 files are on the isolated branch `research-v3-real-world-validation`. The original `main` and Research V2 remain unchanged.

## Dataset attribution

- **Breast Cancer Wisconsin (Diagnostic)** — 569 rows, 30 numerical features. Loaded through `sklearn.datasets.load_breast_cancer()`. Original source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Ionosphere** — 351 rows, 34 numerical features. UCI source: https://archive.ics.uci.edu/dataset/52/ionosphere . Cite Sigillito, V., Wing, S., Hutton, L., and Baker, K. (1989), *Ionosphere* [Dataset], https://doi.org/10.24432/C5W01B . Licensed CC BY 4.0 (attribution required). The included CSV is from a public mirror at https://github.com/mikeizbicki/datasets/blob/master/csv/uci/ionosphere.csv . Its row/column counts and class balance match UCI; direct byte-for-byte comparison with the UCI original was not available here.

**Do not use these benchmarks as actual medical diagnosis or operational radar decision systems.**

## Study design

- **All features:** no feature selection.
- **Correlation:** choose 5 by absolute Pearson correlation with the target.
- **MI:** choose 5 by discrete plug-in mutual information after quantile-binning training data into up to five bins.
- **mRMR MID:** choose 5 using MI relevance minus average pairwise MI redundancy, with exactly the same binning and estimator.

Classifiers: standardized logistic regression (C=1) and random forest (120 trees, minimum leaf size 2). Evaluation: five-fold stratified CV repeated three times, shared folds for all methods, seed 20261116. Selection and discretization use training folds **only**; results are measured on held-out folds.

We report accuracy, macro-F1, log loss in **bits**, Brier score, and selection stability. The fold averages and standard deviations are descriptive. Repeated CV folds overlap, so **they are not 15 independent samples** and naïve confidence intervals would be invalid.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python research_v3/real_world_validation.py
```

Quick single-dataset test:

```bash
python research_v3/real_world_validation.py --dataset breast-cancer --repeats 1
```

Outputs in `research_v3/outputs`: CSV raw folds, method summary, paired fold differences, stability, dataset profiles; PNG charts; metadata JSON. The included GitHub Actions workflow runs the exact same two-dataset benchmark and uploads its result files.

This experiment is a controlled **comparison**, not a novel algorithm. Additional datasets, feature budgets, estimation sensitivity analyses and careful literature comparison are required before a journal submission.
