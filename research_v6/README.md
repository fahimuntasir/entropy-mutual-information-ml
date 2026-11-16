# Research V6 — Information Gain vs Predictive Regret under Sensor Reliability Shift

**Status:** Exploratory, independent research; reproducible experiment. **Not yet publishable as a journal paper.** V6 lives only on the isolated branch `research-v6-sensor-reliability-shift`. Original course report and V2–V5 remain untouched.

## Research question and falsifiable hypothesis

Suppose a classifier trains with two noisy, conditionally independent information sources X1 and X2. X1 has 20% error, X2 has 30% error. X2 provides incremental information, so conditional-MI-based feature selection should improve in-distribution predicted probabilities without necessarily improving hard accuracy. But if X2 becomes less reliable at deployment and the prediction model remains fixed, **the probability benefit should reverse**. A known invariant X1-only baseline then serves as a robustness reference.

The shift changes **P(X2|Y)** and **P(Y|X1,X2)**. It is **class-conditional source-reliability / posterior shift, not standard covariate shift**.

## Exact analytical result

For a balanced binary target and two independent binary symmetric channels, the *training-time* Bayes posterior has log odds determined by the source log-likelihood ratios. Holding the training-time posterior fixed while varying X2's test error rate produces a closed-form, exactly enumerated expected log-loss difference.

For **p=0.20, training q=0.30**, the exact crossover occurs at **test q*=0.3961033434**.

| Test error q | X1-only expected log loss, bits | Stale 2-source log loss, bits | Target-adapted 2-source log loss, bits | Stale model gain vs X1, bits |
|---|---:|---:|---:|---:|
| 0.30 | 0.721928 | 0.645177 | 0.645177 | +0.076751 |
| 0.35 | 0.721928 | 0.685109 | 0.679496 | +0.036820 |
| 0.40 | 0.721928 | 0.725040 | 0.703291 | -0.003112 |
| 0.45 | 0.721928 | 0.764972 | 0.717301 | -0.043044 |
| 0.50 | 0.721928 | 0.804903 | 0.721928 | -0.082975 |
| 0.60 | 0.721928 | 0.884766 | 0.703291 | -0.162838 |

These values are derived from **eight exact joint states**, not simulations. At q_test=0.50 X2 has zero target-domain conditional mutual information. At q_test=0.60 X2 is informative again **if its reliability change is correctly modeled**; the stale training-time model instead assigns the wrong sign to its evidence.

## Reproducible finite-sample experiment

- Data generator: original V2 twenty-feature binary model (two genuine source observations, an exact duplicate of X1, a slightly degraded copy of X1, and 16 independent noise features).
- Methods: X1-only, top-two marginal empirical MI, greedy conditional MI (two features), and known-source oracle X1+X2.
- Learner: identical regularized logistic regression, `C=10`. Feature selection and training only use training observations.
- 30 independent trials × training sizes 100/300/1000 × six X2 test-error values 0.30/0.35/0.40/0.45/0.50/0.60 × four methods = **2,160 held-out method evaluations**.
- Each trial uses 5,000 independent held-out test cases per shift condition, with the same random seed across shift conditions to reduce trend noise.
- Outcomes: accuracy, log loss **in bits**, Brier score, source recovery. Paired 95% Student-t intervals use independent replicate seeds at *each fixed shift condition*; different shift conditions reuse seeds and are not independent.

**Pre-specified main contrast** (n=1000, greedy CMI minus marginal MI, 30 independent trial pairs):

| q_test | Accuracy difference | Log-loss improvement, bits | 95% paired interval |
|---|---:|---:|---|
| 0.30 | 0.0000 | +0.077140 | [+0.074826, +0.079454] |
| 0.50 | 0.0000 | -0.081399 | [-0.088485, -0.074313] |

At n=100 and q_train=0.30, greedy CMI selected both independent source groups in 25/30 trials (83.3%). At n=300 and 1000 it did so in 30/30. Thus finite samples and selected source reliability both matter; the result should not be interpreted as universal superiority.

## Reproduce independently

From the repository root with the existing scientific Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s research_v6/tests -v
python research_v6/sensor_shift_experiment.py
```

Outputs are created only in `research_v6/outputs/`: `trial_results.csv`, `paired_comparisons.csv`, `method_summary.csv`, `exact_bayes_reference.csv`, `metadata.json`, and two figures.

The independent GitHub Actions workflow reproduces **8 verification tests**, **2,160 evaluations**, the exact analytical threshold, and the empirical sign reversal: [verified run](https://github.com/fahimuntasir/entropy-mutual-information-ml/actions/runs/43839104786).

## Originality and limitations

- The well-known identity between conditional mutual information and optimal expected log loss is **not new**.
- The exact crossover in this special two-sensor channel is a simple consequence of conditional distributions and **must not be advertised as a new general theorem**.
- Existing research covers CMI feature selection, redundant information, robust feature selection, sensor shift and model calibration; see the linked literature report `docs/RESEARCH_V6_NOVELTY_ASSESSMENT.md`.
- The experiment uses a **single controlled synthetic distribution**, assumes one invariant source and a stationary label prior, and does not detect or correct shifts at deployment.
- The logistic classifier's predicted probabilities are judged by log loss and Brier score; neither is a calibration-only measure.
- The source reliability q_test and exact crossover are known **to the evaluator**, not to a real deployed model. The oracle explicitly uses known source indices, not a generally usable selector.

**Recommendation: NOT READY for SCIE Q3/Q4 submission.** Further contribution would need to address a defensible open question—e.g., a tested source-reliability monitor or safe uncertainty-aware fallback with deployable assumptions and meaningful comparisons—and undergo another targeted prior-art check. Do not add generic benchmark datasets just to increase paper length.
