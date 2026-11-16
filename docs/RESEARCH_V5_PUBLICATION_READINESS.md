# Research V5 — Publication Readiness Assessment

**Date:** 16 November 2026  
**Status:** Independent exploratory, unpublished research. **Not ready to submit to a journal yet.**

## Reproduced experimental design

Five binary classification datasets: Breast Cancer Wisconsin (569 rows, 30 features), Ionosphere (351, 34), Sonar (208, 60), Spambase (4,601, 57), and Digits 3-vs-8 (357, 64). Sonar and Spambase are attributed public GitHub mirrors of the UCI datasets, with exact input snapshots and SHA-256 checksums in the V5 artifact. Spambase's additional rgroup column was explicitly excluded.

Nested, leakage-safe evaluation: **three outer stratified folds, repeated twice**; inner stratified two-fold selection of feature counts 5/10/15 and MI/mRMR bin counts 3/5 using training-only validation log loss. Compared all features, Pearson correlation, marginal MI and mRMR-MID. Classifiers were standardized logistic regression and 60-tree random forest.

**Verified:** 8/8 unit tests pass; 240 outer-fold evaluations; 960 inner-tuning records; local results agree exactly with the independent [successful GitHub Actions run](https://github.com/fahimuntasir/entropy-mutual-information-ml/actions/runs/43835239945) for prediction metrics and selected features.

## Nested CV accuracy (six outer evaluations, percentages)

| Dataset | LR all | LR MI | LR mRMR | RF all | RF MI | RF mRMR |
|---|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 97.98 | 94.81 | 97.72 | 95.87 | 94.90 | 96.31 |
| Ionosphere | 88.46 | 87.32 | 86.33 | 93.45 | 90.88 | 91.88 |
| Sonar | 74.28 | 74.78 | 73.08 | 79.82 | 76.21 | 76.93 |
| Spambase | 92.45 | 87.69 | 87.98 | 94.71 | 92.58 | 92.86 |
| Digits 3 vs 8 | 99.30 | 97.48 | 98.18 | 98.46 | 97.34 | 97.48 |

The all-features baseline deliberately uses more inputs than the tuned selectors. These are **descriptive repeated-CV means**, not independent-trial confidence intervals. Complete CSV output also includes correlation, macro-F1, log loss (bits), Brier score, tuning choices, selected indices and stability.

## Prespecified dataset-level inference

For logistic regression only, two paired sign-flip comparisons over five dataset-level mean differences (not over overlapping folds), with Holm multiplicity correction:

| mRMR advantage over MI | Mean difference | Exact two-sided p | Holm p |
|---|---:|---:|---:|
| Accuracy, mRMR - MI | +0.00239 | 0.8125 | 1.0000 |
| Log-loss bits, MI - mRMR | +0.01094 | 0.5000 | 1.0000 |

The smallest attainable exact two-sided p with five datasets is 0.0625, so **no significance at p<0.05 could be established by this design**. This is not evidence of equivalence or general superiority.

## Prior art / novelty assessment

- Barbieri et al. (2024), feature-selection accuracy, redundancy and stability benchmarking in *Expert Systems with Applications*: https://doi.org/10.1016/j.eswa.2024.123667
- Papaioannou et al. (2025), MI estimator choice and mRMR feature selection including real-world datasets: https://doi.org/10.3390/info16090724
- Schäfer et al. (October 2026), *SelectArena*, preprint comparing 15 feature selectors on 71 real-world datasets: https://arxiv.org/abs/2610.13644
- Naylor et al. (2026), sparse mRMR with statistical error control: https://doi.org/10.1111/sjos.70074

**Decision: NOT READY for SCIE Q3/Q4 submission on current novelty evidence.** The nested protocol is methodologically improved, but the study does not introduce an algorithm, theorem, or clearly new, generalizable empirical mechanism. Five historical benchmarks are too narrow to claim a broad benchmark contribution.

**Next:** Stop adding generic MI/mRMR comparisons. Return to the V2 result that an independent information source can improve predicted probabilities without changing hard accuracy. Formulate a pre-specified finite-sample/calibration question after targeted literature checking and test its limits with an independent validation setting. Only then reconsider a journal manuscript. Verify journal quartile, scope, actual zero-fee traditional submission option and any other charges at submission time.

This report and experiment are not peer reviewed or published. The original report and Research V2/V3/V4/main remain unchanged. Reproduce with: python research_v5/nested_validation.py on the isolated V5 branch.
