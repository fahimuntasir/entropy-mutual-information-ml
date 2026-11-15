# Research V2 — MI, conditional MI, accuracy and log loss

**Status:** exploratory independent research experiment, **not** an accepted or published research paper.

This is a separate experimental version on the branch research-v2-conditional-mi-logloss. It does **not modify** the original report, experiment.py, or any results on main.

## Research question

Does selecting features using conditional mutual information (MI) improve the predicted probabilities, as measured by held-out log loss, even when classification accuracy does not improve?

The experiment recreates the original binary synthetic-data model and its train/test seeds. There are 20 binary features: X1 and X2 are independently corrupted versions of the target; X3 is an exact copy of X1; X4 is a degraded copy; the remaining 16 features contain independent noise.

### Selectors and learner

- **MI:** two features with the highest marginal empirical MI.
- **Conditional MI:** two features greedily selected by added joint information.
- **X1 only:** one-feature comparison baseline.
- **Oracle X1+X2:** a reference using known source variables, not a real-world selection algorithm.

All selections use training labels only. Every method trains an identical regularized logistic regression model with C=10.0. Models are evaluated on independently generated test samples by:

- Accuracy: fraction of correct hard predictions (higher is better).
- Log loss in **bits**: how well the predicted probabilities match outcomes (lower is better); scikit-learn computes log loss in nats, and the script divides by ln(2).
- Brier score: mean squared probability error (lower is better).
- Both-sources rate: whether selection includes X2 and a feature from the X1 source group.

The default study runs 30 trials for each combination of p={0.05, 0.2, 0.35} and n={100, 300, 1000}, using 5,000 fresh test cases per trial. It also reports trial-paired MI-vs-CMI differences with approximate 95% Student-t intervals.

## Run it

Use the existing top-level requirements.txt (NumPy, SciPy, pandas, scikit-learn, Matplotlib). From the **repository root** run:

    python -m pip install -r requirements.txt
    python research_v2/conditional_mi_logloss.py
    python -m unittest discover -s research_v2/tests -v

Quick check:

    python research_v2/conditional_mi_logloss.py --trials 5 --test-size 1000

All files are written to research_v2/outputs (the folder is created automatically), including trial_results.csv, method_summary.csv, paired_comparisons.csv, theory_reference.csv, accuracy_comparison.png and logloss_comparison.png.

## Initial findings (30 trials)

At noise p=0.20 and 1,000 training observations, marginal MI and conditional MI had **equal test accuracy: 80.048%**. However, conditional MI had lower mean log loss (**0.64603 bits**) than marginal MI (**0.72317 bits**). Both-source recovery was 30/30 for conditional MI and 0/30 for MI.

**Caution:** Conditional MI is **not** always better. At high noise p=0.35 and n=100, conditional MI had worse accuracy and log loss because finding an additional informative feature becomes unreliable at small sample sizes. These are results for a controlled model, not proof of a general advantage. In the fixed synthetic model, an exact population Bayes calculation also predicts equal hard accuracy for the two underlying sources and improved optimal log loss from the second source; that theoretical fact is not new.

This extension is a testable research building block. Before journal submission it needs a rigorous novelty assessment, comparisons to existing research, and additional validation. Do not treat this branch as a journal submission or claim that the original course report has been upgraded into a published paper.
