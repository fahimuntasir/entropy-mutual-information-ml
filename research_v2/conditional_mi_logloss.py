"""Independent extension of the Information Theory course project.

Standalone script: does not import or alter the original experiment.py.
Reproduces its binary data model and seeds. Feature selection uses training
data only, and evaluation uses fresh held-out test observations.

Usage:
  python research_v2/conditional_mi_logloss.py
  python research_v2/conditional_mi_logloss.py --trials 5 --test-size 1000
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"
METHODS = ("MI", "Conditional MI", "X1 only", "Oracle X1+X2")

def generate_binary_data(n: int, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Match original experiment.py's 20-feature binary data model."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x1 = y ^ (rng.random(n) < p)
    x2 = y ^ (rng.random(n) < min(p + 0.1, 0.49))
    x = np.column_stack([
        x1, x2,
        x1,                              # exact duplicate
        x1 ^ (rng.random(n) < 0.05),    # degraded copy
        rng.integers(0, 2, (n, 16)),   # unrelated features
    ]).astype(int)
    return x, y

def empirical_mi_bits(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical discrete mutual information in bits."""
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    counts = np.zeros((int(xi.max()) + 1, int(yi.max()) + 1), dtype=float)
    np.add.at(counts, (xi, yi), 1)
    joint_p = counts / counts.sum()
    independent_p = joint_p.sum(axis=1)[:, None] * joint_p.sum(axis=0)[None, :]
    where = joint_p > 0
    return float(np.sum(joint_p[where] * np.log2(joint_p[where] / independent_p[where])))

def joint_binary_code(x: np.ndarray, indices: list[int]) -> np.ndarray:
    """Encode a short binary feature list as one categorical state."""
    if not indices:
        return np.zeros(len(x), dtype=np.int64)
    return np.sum(x[:, indices].astype(np.int64) * (2 ** np.arange(len(indices))), axis=1)

def select_features(x: np.ndarray, y: np.ndarray, method: str) -> list[int]:
    """Select indices on training data, breaking ties by earliest index."""
    if method == "X1 only":
        return [0]
    if method == "Oracle X1+X2":
        return [0, 1]  # simulation-only reference
    if method == "MI":
        scores = [empirical_mi_bits(x[:, j], y) for j in range(x.shape[1])]
        return np.argsort(-np.array(scores), kind="stable")[:2].astype(int).tolist()
    if method == "Conditional MI":
        chosen: list[int] = []
        for _ in range(2):
            base = empirical_mi_bits(joint_binary_code(x, chosen), y)
            increments = [
                -np.inf if j in chosen else empirical_mi_bits(joint_binary_code(x, chosen + [j]), y) - base
                for j in range(x.shape[1])
            ]
            chosen.append(int(np.argmax(increments)))
        return chosen
    raise ValueError(f"Unknown method: {method}")

def binary_entropy_bits(error_rate: float) -> float:
    if error_rate in (0.0, 1.0):
        return 0.0
    return -error_rate * np.log2(error_rate) - (1 - error_rate) * np.log2(1 - error_rate)

def theoretical_conditional_logloss_bits(p: float, q: float) -> float:
    """Exact H(Y|X1,X2) for two conditionally independent noisy copies of Y."""
    h = 0.0
    for a, b in product((0, 1), repeat=2):
        weights = []
        for y in (0, 1):
            pa = (1 - p) if a == y else p
            pb = (1 - q) if b == y else q
            weights.append(0.5 * pa * pb)
        total = sum(weights)
        for w in weights:
            if w > 0:
                h -= w * np.log2(w / total)
    return float(h)

def run_experiment(
    trials: int = 30,
    test_size: int = 5000,
    train_sizes: tuple[int, ...] = (100, 300, 1000),
    noise_levels: tuple[float, ...] = (0.05, 0.2, 0.35),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare selectors on identical train/test splits for each trial."""
    records: list[dict] = []
    for p in noise_levels:
        for n in train_sizes:
            for trial in range(trials):
                train_seed = 100000 + trial * 100 + n + int(p * 1000)
                test_seed = 900000 + trial * 100 + n + int(p * 1000)
                x_train, y_train = generate_binary_data(n, p, train_seed)
                x_test, y_test = generate_binary_data(test_size, p, test_seed)
                for method in METHODS:
                    chosen = select_features(x_train, y_train, method)
                    model = LogisticRegression(
                        C=10.0, solver="lbfgs", max_iter=1000, random_state=trial
                    )
                    model.fit(x_train[:, chosen], y_train)
                    probability = model.predict_proba(x_test[:, chosen])[:, 1]
                    predicted = model.predict(x_test[:, chosen])
                    # scikit-learn uses natural logs. Divide to report bits.
                    logloss_bits = log_loss(y_test, probability, labels=[0, 1]) / np.log(2)
                    brier = np.mean((probability - y_test) ** 2)
                    both_sources = int(1 in chosen and any(j in chosen for j in (0, 2, 3)))
                    records.append({
                        "noise_p": p, "train_n": n, "trial": trial,
                        "method": method, "accuracy": accuracy_score(y_test, predicted),
                        "logloss_bits": float(logloss_bits), "brier_score": float(brier),
                        "selected_indices": ",".join(map(str, chosen)),
                        "both_sources": both_sources,
                        "train_seed": train_seed, "test_seed": test_seed,
                    })
    runs = pd.DataFrame.from_records(records)
    summary = runs.groupby(["noise_p", "train_n", "method"], sort=True).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_logloss_bits=("logloss_bits", "mean"),
        sd_logloss_bits=("logloss_bits", "std"),
        mean_brier=("brier_score", "mean"),
        both_sources_rate=("both_sources", "mean"),
    ).reset_index()
    paired = runs.pivot(
        index=["noise_p", "train_n", "trial"],
        columns="method", values=["accuracy", "logloss_bits"]
    )
    comparison = pd.DataFrame({
        "accuracy_improvement_cmi_minus_mi":
            paired["accuracy"]["Conditional MI"] - paired["accuracy"]["MI"],
        "logloss_improvement_bits_mi_minus_cmi":
            paired["logloss_bits"]["MI"] - paired["logloss_bits"]["Conditional MI"],
    }).reset_index()
    def paired_stats(g: pd.DataFrame) -> pd.Series:
        out = {}
        for col in (
            "accuracy_improvement_cmi_minus_mi",
            "logloss_improvement_bits_mi_minus_cmi",
        ):
            values = g[col].to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            half = (
                float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
                if len(values) > 1 else np.nan
            )
            out[f"{col}_mean"] = mean
            out[f"{col}_ci95_low"] = mean - half
            out[f"{col}_ci95_high"] = mean + half
        return pd.Series(out)
    comparisons = comparison.groupby(
        ["noise_p", "train_n"], sort=True
    ).apply(paired_stats, include_groups=False).reset_index()
    theory = pd.DataFrame([{
        "noise_p": p, "second_noise_q": min(p + 0.1, 0.49),
        "bayes_accuracy_x1": 1.0 - p,
        "bayes_accuracy_x1_x2": 1.0 - p,
        "optimal_logloss_x1_bits": binary_entropy_bits(p),
        "optimal_logloss_x1_x2_bits":
            theoretical_conditional_logloss_bits(p, min(p + 0.1, 0.49)),
    } for p in noise_levels])
    return runs, summary, comparisons, theory

def save_plots(summary: pd.DataFrame, output: Path) -> None:
    """Create separate accuracy and log-loss figures, at moderate noise."""
    focus_p = 0.2 if np.isclose(summary.noise_p, 0.2).any() else float(summary.noise_p.iloc[0])
    focus = summary[np.isclose(summary.noise_p, focus_p)]
    style = {"MI": "o-", "Conditional MI": "s-", "X1 only": "^-", "Oracle X1+X2": "D--"}
    for metric, ylabel, filename in (
        ("mean_accuracy", "Held-out classification accuracy", "accuracy_comparison.png"),
        ("mean_logloss_bits", "Held-out log loss (bits; lower is better)", "logloss_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
        for method in METHODS:
            part = focus[focus.method == method].sort_values("train_n")
            ax.plot(part.train_n, part[metric], style[method], label=method, linewidth=2)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Binary source model: p={focus_p:g}, q={min(focus_p + 0.1, 0.49):g}")
        ax.set_xticks(sorted(focus.train_n.unique()))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=175)
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2 for confidence intervals")
    if args.test_size < 50:
        parser.error("--test-size must be at least 50")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, summary, comparisons, theory = run_experiment(
        trials=args.trials, test_size=args.test_size
    )
    for filename, frame in (
        ("trial_results.csv", runs),
        ("method_summary.csv", summary),
        ("paired_comparisons.csv", comparisons),
        ("theory_reference.csv", theory),
    ):
        frame.to_csv(output / filename, index=False, float_format="%.10g")
    save_plots(summary, output)
    print("\n=== Theoretical population references ===")
    print(theory.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== p=0.2, train_n=1000: held-out test ===")
    print(summary[(np.isclose(summary.noise_p, .2)) & (summary.train_n == 1000)].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== p=0.2, train_n=1000: paired CMI-vs-MI ===")
    print(comparisons[(np.isclose(comparisons.noise_p, .2)) & (comparisons.train_n == 1000)].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved CSVs and two plots in {output}")

if __name__ == "__main__":
    main()
