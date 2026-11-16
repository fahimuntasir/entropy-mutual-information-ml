"""Research V4: leakage-aware MI/mRMR feature-count and bin-count sensitivity.

Run from repository root: python research_v4/robustness_analysis.py
No V1/V2/V3 file is written or changed. All output is under research_v4/outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, mutual_info_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from research_v3 import real_world_validation as v3

DATASETS = ("breast-cancer", "ionosphere")
MODEL_NAMES = v3.MODELS
FEATURE_BUDGETS = (3, 5, 10, 15)
BIN_COUNTS = (3, 5, 10)
METHODS = ("All features", "Correlation", "Mutual information", "mRMR")
SEED = v3.SEED


def scores_for_bins(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Training-only quantile discretization + plug-in MI in nats."""
    b = v3.quantile_bins(x, bins=bins)
    rel = np.array([mutual_info_score(b[:, j], y) for j in range(b.shape[1])])
    return b, rel


def ranked_mrmr(b: np.ndarray, relevance: np.ndarray, max_features: int) -> list[int]:
    """Classical mRMR MID: relevance minus average pairwise redundancy.

    Cache pair scores; never use held-out features or labels.
    """
    chosen: list[int] = []
    cache: dict[tuple[int, int], float] = {}
    d = b.shape[1]
    for _ in range(min(max_features, d)):
        candidate_scores = np.full(d, -np.inf)
        for j in range(d):
            if j in chosen:
                continue
            redundancy = []
            for previous in chosen:
                key = (min(previous, j), max(previous, j))
                if key not in cache:
                    cache[key] = mutual_info_score(b[:, j], b[:, previous])
                redundancy.append(cache[key])
            candidate_scores[j] = relevance[j] - (float(np.mean(redundancy)) if redundancy else 0.0)
        chosen.append(int(np.argmax(candidate_scores)))
    return chosen


def configurations(x_train: np.ndarray, y_train: np.ndarray,
                   budgets=FEATURE_BUDGETS, bins_choices=BIN_COUNTS):
    """All independent configurations. Baseline methods are not rerun per bin."""
    max_features = min(max(budgets), x_train.shape[1])
    yield ("All features", 0, 0, list(range(x_train.shape[1])))
    corr = np.argsort(-v3.correlation(x_train, y_train), kind="stable").astype(int).tolist()
    for k in budgets:
        yield ("Correlation", k, 0, corr[:k])
    for bins in bins_choices:
        b, relevance = scores_for_bins(x_train, y_train, bins)
        mi_order = np.argsort(-relevance, kind="stable").astype(int).tolist()
        mrmr_order = ranked_mrmr(b, relevance, max_features)
        for k in budgets:
            yield ("Mutual information", k, bins, mi_order[:k])
            yield ("mRMR", k, bins, mrmr_order[:k])


def evaluate(dataset_names=DATASETS, repeats=3, folds=5,
             budgets=FEATURE_BUDGETS, bins_choices=BIN_COUNTS):
    """Repeated stratified CV: feature selection is performed on training only."""
    records = []
    selected = defaultdict(list)
    dataset_profiles = []
    for dataset in dataset_names:
        x, y, names, source = v3.load_data(dataset)
        dataset_profiles.append(dict(dataset=dataset, samples=len(y), features=x.shape[1],
            positives=int(y.sum()), negatives=int((1-y).sum()), source=source))
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
        for split_id, (train, test) in enumerate(cv.split(x, y)):
            xt, yt = x[train], y[train]
            xv, yv = x[test], y[test]
            prediction_cache = {}
            for method, k, bins, indices in configurations(xt, yt, budgets, bins_choices):
                key = (dataset, method, int(k), int(bins))
                selected[key].append(set(indices))
                for model_name in MODEL_NAMES:
                    # Identical features in identical order yield identical predictions;
                    # reusing them avoids unnecessary model fitting.
                    cache_key = (model_name, tuple(indices))
                    if cache_key not in prediction_cache:
                        model = v3.model_for(model_name, SEED + split_id)
                        model.fit(xt[:, indices], yt)
                        proba = model.predict_proba(xv[:, indices])[:, 1]
                        predicted = model.predict(xv[:, indices])
                        prediction_cache[cache_key] = (proba, predicted)
                    proba, predicted = prediction_cache[cache_key]
                    records.append(dict(dataset=dataset, split=split_id,
                        repeat=split_id // folds, fold=split_id % folds,
                        n_train=len(train), n_test=len(test), model=model_name,
                        method=method, k=int(k), bins=int(bins), n_features=len(indices),
                        accuracy=float(accuracy_score(yv, predicted)),
                        f1_macro=float(f1_score(yv, predicted, average="macro")),
                        logloss_bits=float(log_loss(yv, proba, labels=[0,1]) / np.log(2)),
                        brier=float(brier_score_loss(yv, proba)),
                        selected_indices=",".join(map(str, indices))))
    raw = pd.DataFrame.from_records(records)
    summary = raw.groupby(["dataset", "model", "method", "k", "bins"], sort=True).agg(
        observations=("accuracy", "size"),
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_f1_macro=("f1_macro", "mean"), sd_f1_macro=("f1_macro", "std"),
        mean_logloss_bits=("logloss_bits", "mean"), sd_logloss_bits=("logloss_bits", "std"),
        mean_brier=("brier", "mean"), sd_brier=("brier", "std")).reset_index()
    stability_rows = []
    for key, sets in selected.items():
        overlaps = [len(a & b)/len(a | b) for a,b in combinations(sets, 2)]
        stability_rows.append(dict(dataset=key[0], method=key[1], k=key[2], bins=key[3],
            jaccard_mean=float(np.mean(overlaps)), pair_count=len(overlaps)))
    stability = pd.DataFrame(stability_rows)
    return raw, summary, stability, pd.DataFrame(dataset_profiles)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise error control across prespecified primary tests."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order)-rank)*p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def primary_comparisons(raw: pd.DataFrame, repeats: int, folds: int) -> pd.DataFrame:
    """Exploratory Nadeau-Bengio corrected repeated-k-fold paired t tests.

    Positive deltas mean mRMR is better than MI: accuracy difference mRMR-MI,
    log-loss difference MI-mRMR. Error estimate uses (1/(R*K) + test/train)*s^2.
    Repeated folds remain dependent; intervals/p-values are approximations.
    """
    base = raw[((raw.method.isin(["Mutual information", "mRMR"])) &
                (raw.k==5) & (raw.bins==5))]
    records=[]
    for (dataset, model), group in base.groupby(["dataset","model"]):
        ratio = float((group.n_test/group.n_train).mean())
        n_splits = repeats*folds
        assert group.shape[0]==2*n_splits, "Expected complete paired observations"
        pivot = group.pivot(index="split", columns="method", values=["accuracy","logloss_bits"])
        for metric in ("accuracy", "logloss_bits"):
            diff = (pivot[metric]["mRMR"] - pivot[metric]["Mutual information"]) if metric=="accuracy" else (
                pivot[metric]["Mutual information"] - pivot[metric]["mRMR"])
            diffs=np.asarray(diff,dtype=float)
            mu=float(diffs.mean()); sd=float(diffs.std(ddof=1))
            correction=1/n_splits+ratio
            se=float(np.sqrt(correction)*sd)
            if se==0:
                p=1.0 if mu==0 else 0.0
                ci_low=ci_high=mu
                t_value=0.0 if mu==0 else float("inf")
            else:
                t_value=mu/se
                p=float(2*student_t.sf(abs(t_value),df=n_splits-1))
                half=float(student_t.ppf(0.975,df=n_splits-1)*se)
                ci_low=mu-half;ci_high=mu+half
            records.append(dict(dataset=dataset,model=model,metric=metric,
                n_paired_folds=n_splits,test_train_ratio=ratio, correction_factor=correction,
                mrmr_improvement_mean=mu, paired_diff_sd=sd, corrected_se=se,
                ci95_low=ci_low,ci95_high=ci_high, t_statistic=t_value,
                p_raw=p))
    out=pd.DataFrame(records)
    out["p_holm_8_tests"]=holm_adjust(out.p_raw.to_numpy())
    out["significant_holm_0_05"] = out.p_holm_8_tests < .05
    return out


def verify_v3_baseline(raw: pd.DataFrame, previous_csv: Path, tolerance=1e-12):
    """Verify the V3 reference setting has identical held-out outcomes."""
    prior=pd.read_csv(previous_csv)
    current=raw[((raw.k==5) & (raw.method=="Correlation") & (raw.bins==0)) |
        ((raw.k==5) & (raw.method.isin(["Mutual information", "mRMR"])) & (raw.bins==5)) |
        (raw.method=="All features")]
    keys=["dataset","split","method","model"]
    joined=prior.merge(current,on=keys,suffixes=("_v3","_v4"),how="outer",indicator=True)
    if not (joined._merge=="both").all() or len(joined)!=len(prior):
        raise AssertionError("V3/V4 baseline rows differ")
    results={}
    for metric in ("accuracy","f1_macro","logloss_bits","brier"):
        delta=(joined[f"{metric}_v3"]-joined[f"{metric}_v4"]).abs()
        worst=float(delta.max())
        if worst>tolerance:raise AssertionError(f"V3 mismatch: {metric} {worst}")
        results[f"max_difference_{metric}"]=worst
    results.update(n_v3_rows=len(prior),n_v4_reference_rows=len(current),tolerance=tolerance)
    return results


def figures(summary: pd.DataFrame, out: Path, baseline_bin=5):
    colors={"Correlation":"#7489aa", "Mutual information":"#df8839", "mRMR":"#16897c"}
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            base=summary[(summary.dataset==dataset)&(summary.model==model)]
            for metric, y_label, stem in (("mean_accuracy","Mean held-out accuracy","accuracy"),
                    ("mean_logloss_bits","Mean held-out log loss (bits)","logloss")):
                fig,ax=plt.subplots(figsize=(7.2,4.5))
                for method in ("Correlation","Mutual information","mRMR"):
                    d=base[(base.method==method)&(base.bins==(0 if method=="Correlation" else baseline_bin))].sort_values("k")
                    ax.plot(d.k,d[metric],"o-",linewidth=2,label=method,color=colors[method])
                allrow=base[base.method=="All features"].iloc[0]
                ax.axhline(allrow[metric],ls="--",color="gray",label="All features")
                ax.set_xticks(FEATURE_BUDGETS)
                ax.set_xlabel("Number of selected features")
                ax.set_ylabel(y_label)
                ax.set_title(f"{dataset} / {model} (5-bin MI)")
                ax.grid(alpha=.2);ax.legend(fontsize=9)
                fig.tight_layout()
                fig.savefig(out/f"{dataset}_{model.replace(' ','_')}_{stem}.png",dpi=160)
                plt.close(fig)
    # Main sensitivity: log loss vs bin counts at k=5, both data sets LR only.
    fig,axs=plt.subplots(1,2,figsize=(10.4,4),sharey=False)
    for ax,dataset in zip(axs,DATASETS):
        for method in ("Mutual information","mRMR"):
            d=summary[(summary.dataset==dataset)&(summary.model=="Logistic regression") &
               (summary.method==method)&(summary.k==5)].sort_values("bins")
            ax.plot(d.bins,d.mean_logloss_bits,"o-",label=method,color=colors[method])
        ax.set_title(dataset);ax.set_xlabel("Quantile bins")
        ax.set_xticks(BIN_COUNTS);ax.set_ylabel("Mean held-out log loss (bits)")
        ax.grid(alpha=.2)
    axs[0].legend();fig.tight_layout()
    fig.savefig(out/"estimator_bin_sensitivity.png",dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    parser.add_argument("--reference-v3",type=Path,default=ROOT.parent/"research_v3"/"outputs"/"fold_results.csv")
    args=parser.parse_args()
    if args.repeats<1 or args.folds<2:parser.error("repeats>=1, folds>=2")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw,summary,stability,profiles=evaluate(repeats=args.repeats,folds=args.folds)
    stats=primary_comparisons(raw,repeats=args.repeats,folds=args.folds)
    verified=None
    if args.repeats==3 and args.folds==5 and args.reference_v3.exists():
        verified=verify_v3_baseline(raw,args.reference_v3)
    for name,data in (("fold_results",raw),("method_summary",summary),
        ("selection_stability",stability),("primary_corrected_tests",stats),
        ("dataset_profiles",profiles)):
        data.to_csv(args.output_dir/f"{name}.csv",index=False,float_format="%.12g")
    figures(summary,args.output_dir)
    ref=ROOT.parent/"research_v3"/"data"/"ionosphere.csv"
    report=dict(seed=SEED,folds=args.folds,repeats=args.repeats,
        datasets=list(DATASETS),models=list(MODEL_NAMES),budgets=list(FEATURE_BUDGETS),
        bin_counts=list(BIN_COUNTS),estimated_comparisons=len(stats),
        v3_comparison=verified,
        ionosphere_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
        python=platform.python_version(),sklearn=sklearn.__version__,scipy=scipy.__version__,
        numpy=np.__version__,pandas=pd.__version__,
        statistical_method="Nadeau-Bengio approximate corrected repeated k-fold paired t; 8-test Holm correction",
        caveat="Repeated CV is dependent and only two datasets were evaluated; inferential claims remain exploratory.")
    (args.output_dir/"experiment_metadata.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("V3 baseline cross-check:",verified)
    print("Total held-out evaluations:",len(raw))
    print("\nPRIMARY CORRECTED TESTS (mRMR improvement over MI at k=5, 5 bins)")
    print(stats[["dataset","model","metric","mrmr_improvement_mean","ci95_low","ci95_high","p_holm_8_tests"]].to_string(index=False,float_format=lambda n:f"{n:.5f}"))
    print("\nAccuracy sensitivity, logistic regression / five bins")
    s=summary[(summary.model=="Logistic regression")&(
        (summary.bins==5) | ((summary.method=="Correlation")&(summary.bins==0)))]
    print(s[["dataset","method","k","mean_accuracy","mean_logloss_bits"]].to_string(index=False,float_format=lambda n:f"{n:.4f}"))
    print("\nSaved:",args.output_dir.resolve())


if __name__=="__main__":main()
