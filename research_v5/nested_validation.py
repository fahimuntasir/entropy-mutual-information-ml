"""Research V5: nested, leakage-safe feature-selection benchmarking.

Run from repository root:
    python research_v5/nested_validation.py

All tuning happens on inner training folds; the outer test folds are
never used to choose selected features, number of features, or MI bins.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from research_v3 import real_world_validation as v3
from research_v4 import robustness_analysis as v4

DATASETS = ("breast-cancer", "ionosphere", "sonar", "spambase", "digits-3-vs-8")
MODELS = ("Logistic regression", "Random forest")
METHODS = ("All features", "Correlation", "Mutual information", "mRMR")
K_GRID = (5, 10, 15)
BIN_GRID = (3, 5)
SEED = 20261116


def load_dataset(name):
    """Return X, y, source and evidence metadata for an immutable dataset."""
    metadata = {}
    if name in ("breast-cancer", "ionosphere"):
        x, y, names, source = v3.load_data(name)
    elif name == "sonar":
        f = ROOT/"data"/"sonar.csv"
        tab = pd.read_csv(f, header=None)
        if tab.shape != (208, 61):
            raise ValueError("Sonar mirror snapshot has unexpected shape")
        x = tab.iloc[:, :60].to_numpy(dtype=float)
        y = tab.iloc[:, 60].str.strip().map({"R": 0, "M": 1}).to_numpy()
        names = [f"freq_{i+1:02d}" for i in range(60)]
        source = "https://doi.org/10.24432/C5T01Q"
        metadata["mirror"] = "https://github.com/jbrownlee/Datasets/blob/master/sonar.csv"
        metadata["source_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    elif name == "spambase":
        f = ROOT/"data"/"spambase.tsv"
        tab = pd.read_csv(f, sep="\t")
        if tab.shape != (4601, 59):
            raise ValueError("Spambase mirror snapshot has unexpected shape")
        x = tab.iloc[:, :57].to_numpy(dtype=float)
        y = tab["spam"].astype(str).str.strip().map({"non-spam": 0, "spam": 1}).to_numpy()
        names = tab.columns[:57].tolist()
        source = "https://doi.org/10.24432/C53G6X"
        metadata["mirror"] = "https://github.com/WinVector/zmPDSwR/blob/master/Spambase/spamD.tsv"
        metadata["source_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        metadata["ignored_columns"] = ["rgroup"]
    elif name == "digits-3-vs-8":
        ds = load_digits()
        mask = np.isin(ds.target, [3, 8])
        x = ds.data[mask].astype(float)
        y = (ds.target[mask] == 8).astype(int)
        names = [f"pixel_{i:02d}" for i in range(x.shape[1])]
        source = "https://doi.org/10.24432/C50P49"
        metadata["transformation"] = "scikit-learn digits only classes 3 and 8, target=1 for digit 8"
    else:
        raise ValueError(name)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Invalid numeric data or labels for {name}")
    y = np.asarray(y, dtype=int)
    metadata.update(dataset=name, samples=len(y), features=x.shape[1],
                    positive=int(y.sum()), negative=int((1-y).sum()),
                    constant_features=int(np.sum(np.ptp(x, axis=0)==0)), source=source)
    return np.asarray(x, dtype=float), y, list(names), metadata


def model_for(model_name, seed):
    """Fixed classifier hyperparameters; scaling is fitted only on training folds."""
    if model_name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=1200, random_state=seed))
    if model_name == "Random forest":
        return RandomForestClassifier(
            n_estimators=60, min_samples_leaf=2, random_state=seed, n_jobs=1)
    raise ValueError(model_name)


def feature_plans(x, y, k_values=K_GRID, bin_values=BIN_GRID):
    """Compute selection rules on the supplied TRAINING matrix only.

    A plan key is (method, number of selected features, quantile bins).
    'All features' is a fixed-control candidate, not hyperparameter-tuned.
    """
    n_features = x.shape[1]
    k_values = sorted({min(int(k), n_features) for k in k_values if k >= 1})
    plans = {("All features", n_features, 0): tuple(range(n_features))}
    correlation_rank = np.argsort(-v3.correlation(x, y), kind="stable")
    for k in k_values:
        plans[("Correlation", k, 0)] = tuple(map(int, correlation_rank[:k]))
    for bins in bin_values:
        discretized, relevance = v4.scores_for_bins(x, y, bins)
        mi_rank = np.argsort(-relevance, kind="stable")
        mrmr_rank = v4.ranked_mrmr(discretized, relevance, max(k_values))
        for k in k_values:
            plans[("Mutual information", k, bins)] = tuple(map(int, mi_rank[:k]))
            plans[("mRMR", k, bins)] = tuple(map(int, mrmr_rank[:k]))
    return plans


def inner_tune(outer_x, outer_y, model_name, outer_id,
               inner_folds=2, k_values=K_GRID, bin_values=BIN_GRID):
    """Inner CV optimizes log loss only, using outer TRAINING data.

    Every candidate is evaluated on the same inner validation splits.
    No outer-test values/labels can enter this function.
    """
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                              random_state=SEED+1009*outer_id)
    scores = defaultdict(list)
    for inner_id, (train, validate) in enumerate(splitter.split(outer_x, outer_y)):
        x_tr, y_tr = outer_x[train], outer_y[train]
        x_val, y_val = outer_x[validate], outer_y[validate]
        plans = feature_plans(x_tr, y_tr, k_values, bin_values)
        # Reuse predictions when different selectors select the same columns.
        cache = {}
        for candidate, indices in plans.items():
            if indices not in cache:
                model = model_for(model_name, SEED+outer_id*31+inner_id)
                model.fit(x_tr[:, indices], y_tr)
                probabilities = model.predict_proba(x_val[:, indices])[:, 1]
                cache[indices] = float(log_loss(y_val, probabilities, labels=[0,1])/np.log(2))
            scores[candidate].append(cache[indices])
    means = {candidate: float(np.mean(values)) for candidate, values in scores.items()}
    chosen = {}
    for method in METHODS:
        candidates = [c for c in means if c[0] == method]
        # Deterministic tie-breaking chooses fewer features then fewer bins.
        best = min(candidates, key=lambda c: (means[c], c[1], c[2]))
        chosen[method] = (best, means[best])
    return chosen, means


def evaluate_one(name, outer_repeats=2, outer_folds=3, inner_folds=2,
                 k_values=K_GRID, bin_values=BIN_GRID):
    x, y, names, metadata = load_dataset(name)
    cv = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=SEED)
    outer_rows, tuning_rows, selections = [], [], defaultdict(list)
    for outer_id, (tr, te) in enumerate(cv.split(x,y)):
        x_tr, y_tr = x[tr], y[tr]
        x_te, y_te = x[te], y[te]
        if np.intersect1d(tr,te).size:
            raise AssertionError("Outer train and test observations overlap")
        for model_name in MODELS:
            # Tune once per model and outer fold. Test fold is inaccessible here.
            chosen, means = inner_tune(
                x_tr,y_tr,model_name,outer_id,inner_folds,k_values,bin_values)
            plans = feature_plans(x_tr,y_tr,k_values,bin_values)
            for candidate, score in means.items():
                tuning_rows.append(dict(dataset=name,outer_id=outer_id,
                    repeat=outer_id//outer_folds,model=model_name,
                    method=candidate[0],k=candidate[1],bins=candidate[2],
                    inner_mean_logloss_bits=score,
                    selected=int(candidate == chosen[candidate[0]][0])))
            for method in METHODS:
                candidate, train_val_score = chosen[method]
                indices = plans[candidate]
                start = time.perf_counter()
                estimator = model_for(model_name, SEED+outer_id*31)
                estimator.fit(x_tr[:,indices],y_tr)
                probabilities = estimator.predict_proba(x_te[:,indices])[:,1]
                predicted = estimator.predict(x_te[:,indices])
                elapsed=time.perf_counter()-start
                selections[(name,model_name,method)].append(set(indices))
                outer_rows.append(dict(dataset=name,model=model_name,method=method,
                    outer_id=outer_id,repeat=outer_id//outer_folds,
                    fold=outer_id%outer_folds,train_count=len(tr),test_count=len(te),
                    selected_k=candidate[1],selected_bins=candidate[2],
                    accuracy=float(accuracy_score(y_te,predicted)),
                    f1_macro=float(f1_score(y_te,predicted,average="macro")),
                    logloss_bits=float(log_loss(y_te,probabilities,labels=[0,1])/np.log(2)),
                    brier=float(brier_score_loss(y_te,probabilities)),
                    inner_validation_logloss_bits=train_val_score,
                    final_training_seconds=float(elapsed),
                    selected_indices=",".join(map(str,indices))))
        print(f"{name}: completed outer split {outer_id+1}/{outer_repeats*outer_folds}",flush=True)
    stability = []
    for key, sets in selections.items():
        from itertools import combinations
        overlap=[len(a&b)/len(a|b) for a,b in combinations(sets,2)]
        stability.append(dict(dataset=key[0],model=key[1],method=key[2],
            mean_jaccard=float(np.mean(overlap)),n_pairs=len(overlap)))
    return (pd.DataFrame(outer_rows),pd.DataFrame(tuning_rows),
            pd.DataFrame(stability),metadata)


def exact_signflip(differences):
    """Two-sided exact paired sign-flip test on dataset-level differences.

    Unit of analysis is DATASET, not overlapping outer CV folds.
    Five datasets yield 32 sign patterns; minimum two-sided p is 0.0625.
    """
    delta=np.asarray(differences,dtype=float)
    observed=abs(float(delta.mean()))
    null=[abs(float(np.mean(delta*np.array(signs))))
          for signs in itertools.product((-1.,1.),repeat=len(delta))]
    p=float(np.mean(np.asarray(null) >= observed - 1e-14))
    return observed,p


def summarize(outer):
    summary=outer.groupby(["dataset","model","method"],sort=True).agg(
        outer_folds=("accuracy","size"),
        mean_accuracy=("accuracy","mean"),sd_accuracy=("accuracy","std"),
        mean_f1_macro=("f1_macro","mean"),
        mean_logloss_bits=("logloss_bits","mean"),sd_logloss_bits=("logloss_bits","std"),
        mean_brier=("brier","mean"),
        mean_final_training_seconds=("final_training_seconds","mean")
    ).reset_index()
    contrasts=[]
    for (dataset,model),g in summary.groupby(["dataset","model"]):
        by=g.set_index("method")
        contrasts.append(dict(dataset=dataset,model=model,
            mrmr_minus_mi_accuracy=float(by.loc["mRMR","mean_accuracy"]-
                                         by.loc["Mutual information","mean_accuracy"]),
            mi_minus_mrmr_logloss_bits=float(by.loc["Mutual information","mean_logloss_bits"]-
                                             by.loc["mRMR","mean_logloss_bits"]),
            all_minus_best_filter_accuracy=float(by.loc["All features","mean_accuracy"]-
                max(by.loc["mRMR","mean_accuracy"],by.loc["Mutual information","mean_accuracy"]))))
    comparisons=pd.DataFrame(contrasts)
    lr=comparisons[comparisons.model=="Logistic regression"].sort_values("dataset")
    tests=[]
    for metric in ("mrmr_minus_mi_accuracy","mi_minus_mrmr_logloss_bits"):
        diff=lr[metric].to_numpy()
        stat,p=exact_signflip(diff)
        tests.append(dict(model="Logistic regression",metric=metric,
            datasets=len(diff),mean_difference=float(diff.mean()),abs_statistic=stat,
            p_two_sided_exact=p))
    test=pd.DataFrame(tests)
    # Holm correction for the two pre-specified outcomes.
    from research_v4.robustness_analysis import holm_adjust
    test["p_holm_2_tests"]=holm_adjust(test.p_two_sided_exact.to_numpy())
    return summary,comparisons,test


def figures(summary,out):
    for model in MODELS:
        d=summary[summary.model==model]
        for value,ylab,stem in (
          ("mean_accuracy","Outer held-out accuracy","accuracy"),
          ("mean_logloss_bits","Outer log loss in bits (lower is better)","logloss")):
            xlabels=list(DATASETS)
            vals=np.array([[float(d[(d.dataset==name)&(d.method==method)][value].iloc[0])
               for method in METHODS] for name in xlabels])
            pos=np.arange(len(xlabels)); width=.19
            fig,ax=plt.subplots(figsize=(10,4.5))
            for j,method in enumerate(METHODS):
                ax.bar(pos+(j-1.5)*width,vals[:,j],width,label=method)
            ax.set_xticks(pos,["Breast cancer","Ionosphere","Sonar","Spambase","Digits 3 vs 8"])
            ax.set_ylabel(ylab);ax.set_title(f"Nested CV: {model}")
            ax.legend(ncol=2,fontsize=9);ax.grid(axis="y",alpha=.2)
            fig.tight_layout()
            fig.savefig(out/f"{model.replace(' ','_')}_{stem}.png",dpi=165)
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    parser.add_argument("--outer-repeats",type=int,default=2)
    parser.add_argument("--outer-folds",type=int,default=3)
    parser.add_argument("--inner-folds",type=int,default=2)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    args=parser.parse_args()
    if min(args.outer_repeats,args.outer_folds,args.inner_folds)<1 or min(args.outer_folds,args.inner_folds)<2:
        parser.error("outer-folds/inner-folds must >=2 and repeats >=1")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_outer=[];all_tune=[];all_stability=[]; profiles=[]
    start=time.perf_counter()
    for name in args.datasets:
        outer,tuning,stability,profile=evaluate_one(name,
            args.outer_repeats,args.outer_folds,args.inner_folds)
        all_outer.append(outer);all_tune.append(tuning)
        all_stability.append(stability);profiles.append(profile)
    outer=pd.concat(all_outer,ignore_index=True)
    tuning=pd.concat(all_tune,ignore_index=True)
    stabilities=pd.concat(all_stability,ignore_index=True)
    summary,contrasts,tests=summarize(outer) if len(args.datasets)==len(DATASETS) else (None,None,None)
    outer.to_csv(args.output_dir/"outer_fold_results.csv",index=False)
    tuning.to_csv(args.output_dir/"inner_tuning_results.csv",index=False)
    stabilities.to_csv(args.output_dir/"selection_stability.csv",index=False)
    pd.DataFrame(profiles).to_csv(args.output_dir/"dataset_profiles.csv",index=False)
    if summary is not None:
        summary.to_csv(args.output_dir/"nested_summary.csv",index=False)
        contrasts.to_csv(args.output_dir/"dataset_level_contrasts.csv",index=False)
        tests.to_csv(args.output_dir/"exact_dataset_level_tests.csv",index=False)
        figures(summary,args.output_dir)
        print("\n=== PRIMARY OUTER RESULTS ===")
        print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
        print("\n=== FIVE-DATASET SIGN-FLIP TESTS ===")
        print(tests.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    metadata=dict(seed=SEED,datasets=args.datasets,outer_repeats=args.outer_repeats,
        outer_folds=args.outer_folds,inner_folds=args.inner_folds,
        feature_budget=list(K_GRID),bin_grid=list(BIN_GRID),
        selection_objective="inner validation mean log loss in bits (minimize)",
        outer_test_used_in_tuning=False,
        models=["StandardScaler+LR(C=1.0)", "RF(60 trees,min_samples_leaf=2)"],
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        elapsed_seconds=time.perf_counter()-start,
        limitations="Five heterogeneous datasets, dependent repeated outer CV, no external temporal validation; exact sign-flip inference uses only five independent dataset units.",
        profiles=profiles)
    (args.output_dir/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"\nSaved {len(outer)} outer-fold evaluations and {len(tuning)} tuning rows in {args.output_dir}; elapsed {metadata['elapsed_seconds']:.1f}s",flush=True)


if __name__=="__main__":main()
