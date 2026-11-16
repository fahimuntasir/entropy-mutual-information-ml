# Research V7 — Label-free sensor agreement: adaptation and failure conditions

**Status:** Independent exploratory research. This is **not a new algorithm claim** or a journal-ready article. The original report and V2–V6 are preserved on their existing branches; only V7 files have been added here.

## Question

Does a simple estimator of secondary sensor reliability, based entirely on **unlabeled disagreement** with a calibrated reference sensor, improve probability estimates after the sensor degrades? What happens when the model assumptions are wrong?

This question follows directly from the V6 observation that an added signal may reduce log loss during training, then hurt log loss when its reliability changes. This V7 prototype is an implementation of an **established model-based idea**, not a demonstrated novel contribution.

## Identification: the important restriction

Assume two binary sensor observations of the same unknown label Y:

- X1 = Y XOR E1, with *known and stable* independent error rate p = 0.20.
- X2 = Y XOR E2, with error rate q that may vary over time.
- E1 and E2 are independent conditional on Y; their error rates are symmetric for both label classes.

The *unlabeled* disagreement rate d = P(X1 != X2) satisfies

    d = p + q - 2 p q

and therefore

    estimated_q = (empirical_d - p) / (1 - 2 p)

provided p != 0.5. The estimator uses **no target label**. The calibration of the reference p is a separate, strong supervised assumption. If both p and q are unknown or error dependence changes, this relation does **not** identify q.

## Exact indistinguishable-world counterexample

Even with infinitely many unlabeled pairs, an observer cannot distinguish these two model worlds when p is allowed to change:

| World | Reference p | Secondary q | Disagreement d | P(X1=1,X2=0) |
|---|---:|---:|---:|---:|
| A, secondary drift | 0.20 | 0.40 | 0.44 | 0.22 |
| B, anchor drift | 0.35 | 0.30 | 0.44 | 0.22 |

For balanced labels and independent symmetric errors, **the complete unlabeled joint distribution P(X1,X2) is identical in both worlds**. Yet for the same observation (X1=1,X2=0), the correct posterior is **0.72727** in A versus **0.44318** in B. Therefore *no approach observing only unlabeled (X1,X2) pairs can always identify which world or the correct target posterior without additional assumptions or information*. Run `research_v7/identifiability_witness.py` to reproduce all four joint cells.

The counterexample is an elementary identifiability consequence and **not a claimed new theorem**. It exposes the known-anchor assumption's importance.

## Experiment

Seven fixed scenarios:

1. Stable: p=.20, q=.30, independent sensors, label prior .50.
2. Moderate second-sensor shift: p=.20, q=.40, independent.
3. Secondary sensor degraded: p=.20, q=.50, independent.
4. Secondary sensor inverted: p=.20, q=.65, independent.
5. Conditionally dependent errors: p=.20, q=.50, shared-error mixture rho=.35.
6. Anchor drift: p=.35, q=.30, independent, violating the known p=.20 assumption.
7. Label-prior shift: p=.20, q=.30, prior=.75, violating the balanced-training-prior assumption.

For each scenario, **100 independent trials**, **unlabeled monitoring windows 50/200/1000**, and a separately generated **4,000 labeled test observations** (labels hidden until scoring). Each trial and scenario produces six competing predictions:

- **Fixed fusion:** the original two-source Bayes predictor with p=.20, q=.30, prior=.50.
- **X1 fallback:** known-anchor-only predictor.
- **Agreement plug-in:** replace q with its unlabeled disagreement estimate.
- **Shrinkage plug-in:** shrink estimated q toward q_train=.30, with a fixed weight M/(M+200).
- **Agreement gate:** return to X1 when estimated q exceeds V6's already-computed crossover threshold .3961033434; otherwise retain fixed fusion.
- **True-distribution oracle:** receives true p, q, label prior and error dependence; inaccessible in real deployment and therefore only an upper reference.

No target labels are used to fit or switch a model. The **oracle uses parameters supplied by the simulator**, not estimated from the unlabeled data.

Total: **7 scenarios × 100 trials × 3 windows × 6 methods = 12,600 held-out evaluations**.

Measurements are held-out accuracy, log loss **in bits**, and Brier score. Four comparison hypotheses were fixed *before* running the study at monitoring window M=200. Paired 95% intervals and Holm-adjusted two-sided p-values use the independent simulation trials as replication units **within each scenario**. Shared seeds and test cases across window sizes are not independent trials across windows.

## Key results, monitoring window M=200

| Scenario | Fixed log loss | Plug-in log loss | X1 fallback log loss | True oracle log loss |
|---|---:|---:|---:|---:|
| Stable | 0.64666 | 0.65498 | 0.72348 | 0.64666 |
| Sensor degraded q=.50 | 0.80434 | 0.72888 | 0.72201 | 0.72201 |
| Sensor inverted q=.65 | 0.92382 | 0.68656 | 0.72189 | 0.68008 |
| Dependent errors rho=.35 | 0.88457 | 0.86781 | 0.72241 | 0.69965 |
| Anchor drift | 0.90287 | 0.94932 | 1.02134 | 0.82560 |

**Interpretation:** Adaptation helps compared with stale fixed fusion when a sensor degrades; an inverted signal can be helpful if identified. But using an estimated q adds variance at baseline, is slightly worse than X1 fallback when q=.50, and fails badly with dependent errors. An anchor that itself changes makes q estimates unidentifiable. A shifted class prior creates additional probability errors not handled by changing q alone.

## How to run

From the repository root (Python 3.12 recommended; pinned top-level requirements.txt):

    python -m pip install -r requirements.txt
    python -m unittest discover -s research_v7/tests -v
    python research_v7/identifiability_witness.py
    python research_v7/adaptive_reliability.py

Outputs generated inside `research_v7/outputs/`: 12,600-row trial CSV, summaries, paired comparisons and four planned tests, identifiability witness CSV, two PNG figures and experiment metadata JSON.

## No publication novelty claim

The agreement estimator and agreement-based reliability inference have substantial prior art. The seven scenarios and exact counterexample are useful *stress-test evidence* for choosing safe assumptions, not a new algorithm or general impossibility theorem. The study remains synthetic, assumes a calibrated reference, and has no demonstration on real drifting hardware. The paper's independent SCIE Q3/Q4 **$0-fee** goal remains unchanged, but **submission is not recommended yet**. Read `docs/RESEARCH_V7_NOVELTY_ASSESSMENT.md`.

Research V7 demonstrates where a simple model can and cannot work. It does **not** justify inventing an original method name, using the word 'novel' about the formula, or claiming a calibrated reliability estimate without knowing that the assumptions hold.
