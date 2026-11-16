# Research V6 — Literature, Novelty, and Publication-Readiness Assessment

**As of:** 16 November 2026  
**Project:** Entropy and Mutual Information in Modern Machine Learning  
**Stage:** Independent exploratory work, **not** peer reviewed or submitted.

## Executive decision

**Scientific execution: VERIFIED. Journal-level originality: NOT ESTABLISHED. Recommendation: DO NOT SUBMIT YET.**

V6 developed and tested a narrow falsifiable hypothesis: a second source that improves in-distribution log loss, with almost unchanged hard accuracy, can produce *worse* log loss than an invariant one-source predictor after the second source's error rate changes. This is a valid, clear experimental finding under the stated model, but the general statistical mechanism is known and the exact threshold is model-specific. V6 is a useful building block, not a new general theory or publishable new algorithm.

## 1. Prior-art comparison

| Existing contribution | Key reference and verified scope | Overlap / limitation for V6 |
|---|---|---|
| CMI identifies conditional relevance while discouraging redundancy | Wollstadt, Schmitt & Wibral, *JMLR* 24 (2023), https://www.jmlr.org/papers/v24/21-0482.html | CMI itself and the benefits of removing duplicate source information are not new. |
| Sequential and dynamic CMI-driven feature selection | Gadgil, Covert & Lee, *ICLR* (2024), https://proceedings.iclr.cc/paper_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html | Stronger existing CMI methods; V6 does not implement a new selector. |
| mRMR depends on how MI is estimated | Papaioannou et al., *Information* 16(9), 724 (2025), https://doi.org/10.3390/info16090724 | Repeating MI/mRMR benchmarking is not an original contribution. |
| Redundancy and conditional relevance under noisy IoT multi-source conditions | Jiang, Xu & Shen, *Sensors* 26(8), 2340 (10 April 2026), https://doi.org/10.3390/s26082340 | Very close application area; V6 does not outperform or extend their robust method. |
| Robust feature selection for deployment variations | Hanada et al., *Safe Distributionally Robust Feature Selection under Covariate Shift*, arXiv (March 2026), https://arxiv.org/abs/2603.16062 | Deployment shift and sensor selection are active research topics. Their formal shift assumption is different from V6's. |
| Redundant features and distribution-shift vulnerability | Ortiz-Jiménez et al., *Redundant features can hurt robustness to distribution shift*, ICML workshop (2020), https://corsmal.eecs.qmul.ac.uk/resources/2020.07.17_CORSMAL_ICML_UDL_paper.pdf | Robustness problems caused by training-time evidence are already established. |
| Proper losses are not calibration-only metrics | scikit-learn probability calibration documentation, https://scikit-learn.org/stable/modules/calibration.html | Lower log loss / Brier does not, by itself, isolate calibration from resolution. |

**Review status:** Targeted evidence, not an exhaustive systematic literature search. No defensible claim of *first discovery* can currently be made.

## 2. Pre-specified V6 hypothesis and model

- A balanced binary target Y passes through two independent noisy binary channels.
- Reliable source X1: error **p=0.20**, invariant across train/test.
- Secondary source X2: training error **q=0.30**, test error varies from **0.30 to 0.60**.
- Additional 18 columns reproduce the V2 feature layout (duplicate, corrupted copy, 16 unrelated noise variables).
- Competing train-only selectors: top-two marginal MI, greedy conditional MI, single-source X1, known-source oracle.
- Downstream classifier: the same regularized logistic regression in each method.
- Metrics: held-out accuracy, log loss (bits) and Brier score.

**Correct terminology:** Changing P(X2|Y) changes the posterior P(Y|X1,X2), so this is *class-conditional sensor-noise and posterior shift*. It is **not** standard covariate shift (which assumes an invariant P(Y|X)). No model has access to target labels for fitting or selecting features.

The exact training-distribution Bayes posterior is held fixed under test-time shifts. With the above channel parameters, analytical enumeration of all eight joint binary states yields **q*=0.3961033434** as the point at which the stale two-source model ceases to beat the invariant one-source model on expected log loss.

This *specific numerical threshold is not universal*, and its derivation is straightforward under the explicit model; do not claim a new theorem.

## 3. Verified results

**Independent GitHub run:** https://github.com/fahimuntasir/entropy-mutual-information-ml/actions/runs/43839104786

- 8 tests passed.
- 3 training sizes × 30 replicate seeds × 6 target noise values × 4 approaches = **2,160 held-out evaluations**.
- 5,000 test records per replicate/shift.
- Greedy CMI selected both independent source groups 25/30 times at n=100, and 30/30 times at n=300 and n=1000 under training q=.30.

Primary (pre-specified): n=1000, 30 paired independent replicate seeds, CMI versus marginal MI.

| Test X2 error | Mean accuracy difference | Mean reduction in log loss (bits) | Paired 95% interval |
|---|---:|---:|---|
| q_test=.30 | 0.0000 | **+0.077140** | +0.074826 to +0.079454 |
| q_test=.50 | 0.0000 | **−0.081399** | −0.088485 to −0.074313 |

An additional exact comparison shows: at q_test=.50, X2 has **zero** information about Y beyond X1; at q_test=.60 it provides **positive** conditional information again, but with the *opposite* direction from training. A predictor adjusted to the true target channel can benefit; the stale predictor becomes worse. The observed reversal is about model reliance under changing source reliability, not a violation of information theory.

## 4. Strengths and blockers

**Strengths:** Exact mathematical reference; held-out experimental validation; measured confidence intervals over independent seeds; carefully controlled shift; reproducible code; direct connection to the original Information Theory course project.

**Blockers:** One binary synthetic channel; no real changing sensor data; no shift detection or mitigation method; no comparison to contemporary robust multi-source algorithms; no analysis of target-label-free adaptation; one downstream classifier; the basic relationship follows from known Bayes and proper-scoring theory; prior art already covers feature-selection robustness and source redundancy. Student's journal target is a reputable SCIE Q3/Q4 title with **zero publication fees**, but journal selection is premature until research novelty is established.

The intervals are replicate-based for each fixed q and do not establish uncertainty over other source distributions. The same test seeds were deliberately reused across different q values.

## 5. Narrow candidate for additional research — NOT yet a claimed contribution

**Candidate question:** *Can a simple target-label-free monitor of agreement between two noisy sensors identify when a secondary source's reliability has degraded enough that it should be downweighted or excluded, preserving predictive log loss without sacrificing valuable information when conditions are stable?*

Potential assumptions: a trustworthy reference sensor with known error rate, conditional independence of sources given the label, stable target prevalence, and unlabeled paired target-sensor measurements. These assumptions are substantial and may fail in real applications.

This is **a candidate hypothesis**, not a validated new method; agreement-based noisy-label/sensor estimation is an established field and requires another targeted literature check before any originality claim. A convincing paper would need a deployable algorithm, failure-mode analysis, existing robust-sensor-fusion benchmarks, and tests on independent shifts including violation of assumptions.

**Gate:** Before drafting the journal manuscript, complete prior-art review for unlabeled reliability estimation and shift-aware weighting, then either establish an original identifiable contribution or pivot the paper's claim. Do not endlessly expand generic datasets or claim that q*=0.396 is itself novel.

## 6. Reproducibility and preservation

Run from the repository root:

    python -m pip install -r requirements.txt
    python -m unittest discover -s research_v6/tests -v
    python research_v6/sensor_shift_experiment.py

Outputs and figures reside in research_v6/outputs and the GitHub Actions artifact. No changes were made to the original report, default main, or previous V2–V5 branch files.

**Readiness verdict: NOT READY FOR JOURNAL SUBMISSION.**
