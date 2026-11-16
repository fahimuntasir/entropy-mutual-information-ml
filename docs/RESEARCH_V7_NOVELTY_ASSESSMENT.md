# Research V7 — Prior Art, Identifiability, and Publication-Readiness Assessment

**Review date:** 16 November 2026  
**Status:** Exploratory independent Information Theory work; not submitted, accepted, or peer-reviewed.  
**Target:** Reputable SCIE Q3/Q4 journal with genuinely zero mandatory submission/publication fees. This target has **not** yet been independently verified for any current journal.

## Decision

**Experiment: verified. Practical limitation: demonstrated. Journal-level originality: NOT established. Recommendation: do not submit this work as a novel method.**

The V7 exercise identifies a simple model-based reliability estimator from unlabeled sensor agreement, implements a few transparent adaptation policies, and stress-tests them. The estimator is a direct moment inversion under a calibrated anchor and conditional independence. Existing literature already covers unlabeled accuracy estimation and observer reliability. It would be inaccurate to claim this algorithm is original.

## 1. Relevant established and recent literature

| Source | Main established contribution | Consequence for V7 |
|---|---|---|
| Dawid & Skene, 1979, *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*, JRSS C, https://doi.org/10.2307/2346806 | Infer observer error rates from imperfect observations, using a latent-label statistical model | Estimating reliabilities without ground-truth labels is a classical research idea |
| Hui & Walter, 1980, *Estimating the Error Rates of Diagnostic Tests*, Biometrics, https://pubmed.ncbi.nlm.nih.gov/7370371/ | Identify two imperfect diagnostic tests under carefully chosen multiple-population and independence assumptions | Identifiability requires assumptions; two arbitrary tests in one population are not generally identifiable |
| Platanios, Blum & Mitchell, 2014, *Estimating Accuracy from Unlabeled Data*, UAI, https://proceedings.mlr.press/r12/university14g.html | Unlabeled classifier-accuracy estimation from agreement under independence/relaxed assumptions | Central premise overlaps V7 |
| Jaffe, Nadler & Kluger, 2015, *Estimating the Accuracies of Multiple Classifiers without Labeled Data*, AISTATS, https://proceedings.mlr.press/v38/jaffe15.html | Spectral unsupervised estimator using multiple independent classifiers | Standard algorithms are more general than V7's known-anchor two-sensor special case |
| Platanios, Dubey & Mitchell, 2016, *Estimating Accuracy from Unlabeled Data: A Bayesian Approach*, ICML, https://proceedings.mlr.press/v48/platanios16.html | Bayesian estimation and aggregation of classifier accuracies from unlabeled predictions | Adaptive reliability weighting is not an unexplored principle |
| Lukats et al., 2024, survey of fully unsupervised concept-drift detectors, https://doi.org/10.1007/s41060-024-00620-y | Comprehensive label-free drift literature | Monitoring drift without labels is an established research domain |
| Sun et al., 2025, unsupervised sensor drift compensation in electronic noses, https://doi.org/10.1016/j.eswa.2024.125444 | Sensor drift mitigation via unsupervised domain adaptation | Application and adaptation are already researched |
| DecayBench, 2026, *A Reference-Free Benchmark for Trustworthy Drift Detection*, https://www.mdpi.com/2227-7390/14/17/3045 | Calibrated label-free drift detection and aggregation benchmark | Contemporary work has formal guarantees and broad evaluations beyond V7 |
| *Comparing diagnostic tests without a gold standard: How well does the Hui–Walter model really work?*, 2005, https://doi.org/10.1016/j.annepidem.2005.07.022 | Simulation reveals bias under correlated errors and covariate effects | V7's failure conditions are expected from established identifiability limitations |

**Scope caveat:** This is a targeted literature review, not an exhaustive systematic review or a proof that every imaginable extension has been published. It is enough to withhold a novelty claim for the current model.

## 2. Hypothesis and estimator

Y is binary, prior 1/2 in the baseline; X1=Y XOR E1 and X2=Y XOR E2. Assume independent symmetric errors, with known reference sensor error p=0.20. The unknown secondary sensor error q can be inferred from the unlabeled disagreement probability d:

    d = p(1-q) + (1-p)q = p + (1-2p)q
    q_hat = (d_hat - p)/(1-2p)

This is a **known elementary method-of-moments result**, conditional on the reference calibration and error independence. If p=0.5 the equation contains no information about q. If p changes, or the errors are dependent, q is not reliably identified by this inversion.

The candidate V7 method plugs q_hat into the fixed two-sensor Bayes posterior. Shrinkage and a V6-derived gating threshold are separate, pre-specified heuristic baselines. **No target test labels** enter estimation or switching; the full-information oracle is an intentionally unattainable comparison.

## 3. Exact indistinguishable-world demonstration

Two distinct plausible deployment worlds have the **same entire unlabeled joint distribution** (both have balanced labels and conditional-independence of symmetric errors):

World A: p=.20, q=.40. World B: p=.35, q=.30.

Both have disagreement probability d=.44. They therefore both have

    P(00)=P(11)=0.28
    P(01)=P(10)=0.22

but for the same conflicting evidence X1=1, X2=0, the correct posteriors are

    P(Y=1 | X1=1, X2=0, A) = 0.7272727
    P(Y=1 | X1=1, X2=0, B) = 0.4431818

In A the first sensor should be favored; in B the second sensor should be favored. **No classifier or detector observing only the unlabeled sensor pairs could distinguish these worlds universally**, because the observation law is identical. This impossibility boundary disappears if additional genuine information/assumptions are supplied (e.g., stable known reference p, third independent reference, intermittent target labels, or multiple populations meeting identifiability conditions).

The example is an **elementary re-expression of classical identifiability**, not a new general theorem. The file `research_v7/identifiability_witness.py` produces the exact joint probabilities and posteriors.

## 4. Verified experiment

- Seven scenarios: stable, moderate secondary drift, severe secondary degradation, inverted secondary sensor, conditionally dependent errors, anchor drift, and label-prior shift.
- 100 independent replicate seeds per scenario. Each replicate has separate unlabeled monitoring data and labeled evaluation data. No labels enter adaptation.
- Unlabeled monitoring sample counts M=50,200,1000; 4,000 held-out records per replicate.
- Six approaches: fixed fusion, X1 fallback, agreement plug-in, pre-specified shrinkage plug-in, agreement gate, true-distribution oracle.
- 7×100×3×6 = **12,600 method-by-test evaluations**. All predictions checked against held-out labels only **after** choosing the weights.
- Metrics: accuracy, log loss in bits, Brier score. Four pre-specified seed-paired comparisons at M=200; simultaneous interpretation uses Holm adjustment across those four tests.
- 10 preliminary unit tests passed; an additional exact identifiability witness test is included in the final version.
- Automated source and results: https://github.com/fahimuntasir/entropy-mutual-information-ml/tree/research-v7-label-free-sensor-adaptation

### Primary pre-specified paired results (M=200, 100 independent replicate seeds each)

Positive difference means agreement plug-in has **lower (better) log loss** than the named baseline.

| Scenario | Baseline | Mean improvement (bits) | 95% paired interval |
|---|---|---:|---|
| Stable | Fixed fusion | -0.008317 | [-0.010530, -0.006103] |
| Secondary degraded (q=.50) | Fixed fusion | +0.075465 | [+0.073172, +0.077758] |
| Secondary degraded (q=.50) | X1 fallback | -0.006864 | [-0.008694, -0.005034] |
| Conditionally dependent errors | X1 fallback | -0.145403 | [-0.158450, -0.132355] |

These are controlled-simulator comparisons. Confidence intervals quantify seed variability **under the chosen generator**, not generalization to unknown sensor hardware or real-world environments.

### Secondary findings

At M=200, the inverted secondary sensor (q=.65) gives plug-in log loss about 0.68656 bits, versus fixed 0.92382, X1 fallback 0.72189, oracle 0.68008. Under dependent errors, q_hat is typically near 0.32 even though the marginal secondary sensor error is 0.50. With anchor drift, q_hat approaches 0.40 although true q remains 0.30. Prior shift also harms the posterior due to incorrect assumed label prevalence.

**Interpretation:** No tested method is uniformly best. Adapting can outperform the stale model, but can be inferior to a conservative fallback and can fail severely when identification assumptions are wrong.

## 5. Journal-readiness and next decision

**No new method established.** V7's agreement estimator, shrinkage and threshold gate are conventional simple baselines; the exact counterexample is a known-style identifiability result. The novel contribution required for a reputable research journal has **not** been demonstrated. In particular, **do not** label the method as a novel 'MI algorithm', call the inverted-sensor result a new theory, or promise a Q3/Q4 acceptance.

To make a research-paper claim would require one of:

1. A provably identifiable and practically verifiable extension under *meaningfully weaker* assumptions than known methods, with targeted literature support.
2. A real-world sensor-fusion contribution and strong comparisons to current label-free adaptation techniques, including assumption violations and deployment cost.
3. A tightly scoped methods/negative-results or reproducibility paper aimed at a journal that explicitly values such contributions (scope/indexing/fees to be checked).

**Recommended decision:** Pause the sequence of generic experiments. Discuss the identifiability restriction and potential research contribution with a supervisor/domain expert, then conduct a targeted new literature review on any genuinely proposed extension before building another version. The original single-author publication ambition can remain, subject to journal and university contribution policies. There is no defensible journal-ready novelty claim in V7 as written.
