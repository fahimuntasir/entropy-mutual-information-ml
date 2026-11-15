"""Independent Research V3: real-data validation. No changes to V1 or V2.
Run: python research_v3/real_world_validation.py
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss, mutual_info_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 20261116
METHODS = ("All features", "Correlation", "Mutual information", "mRMR")
MODELS = ("Logistic regression", "Random forest")

def load_data(name):
    if name == "breast-cancer":
        d = load_breast_cancer()
        x, y = np.asarray(d.data, float), np.asarray(d.target, int)
        features = list(map(str,d.feature_names))
        source = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
    elif name == "ionosphere":
        path = ROOT/"data"/"ionosphere.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; dataset is on this research branch")
        d = pd.read_csv(path, header=None)
        if d.shape != (351,35):
            raise ValueError(f"Ionosphere expected 351x35, got {d.shape}")
        x = d.iloc[:,:34].to_numpy(dtype=float)
        labels = d.iloc[:,34].astype(str).str.strip().str.lower()
        if set(labels.unique()) <= {"0","1"}:
            y = labels.astype(int).to_numpy()
        elif set(labels.unique()) <= {"g","b"}:
            y = labels.map({"g":1,"b":0}).to_numpy(dtype=int)
        else:
            raise ValueError("Unknown Ionosphere labels")
        features = [f"attribute_{j+1:02d}" for j in range(34)]
        source = "https://archive.ics.uci.edu/dataset/52/ionosphere"
    else: raise ValueError(name)
    if not np.isfinite(x).all() or set(np.unique(y))!={0,1}:
        raise ValueError("Invalid feature/label matrix")
    return x, y, features, source

def quantile_bins(x, bins=5):
    """Thresholds fit only on the current training subset."""
    b = np.zeros(x.shape, dtype=np.int16)
    for j in range(x.shape[1]):
        col=x[:,j]
        if np.min(col)==np.max(col): continue
        cuts=np.unique(np.quantile(col,np.arange(1,bins)/bins))
        b[:,j]=np.searchsorted(cuts,col,side="right")
    return b

def correlation(x,y):
    result=np.zeros(x.shape[1])
    yc=y-y.mean()
    yd=np.sqrt(np.dot(yc,yc))
    for j in range(x.shape[1]):
        xc=x[:,j]-x[:,j].mean()
        den=np.sqrt(np.dot(xc,xc))*yd
        if den>0: result[j]=abs(np.dot(xc,yc)/den)
    return result

def choose(x,y,method,k=5):
    k=min(k,x.shape[1])
    if method=="All features": return list(range(x.shape[1]))
    if method=="Correlation":
        return np.argsort(-correlation(x,y),kind="stable")[:k].astype(int).tolist()
    if method not in ("Mutual information","mRMR"): raise ValueError(method)
    b=quantile_bins(x)
    rel=np.array([mutual_info_score(b[:,j],y) for j in range(x.shape[1])])
    if method=="Mutual information":
        return np.argsort(-rel,kind="stable")[:k].astype(int).tolist()
    # mRMR MID: relevance minus mean pairwise redundancy.
    chosen=[]; cache={}
    for _ in range(k):
        scores=np.full(x.shape[1],-np.inf)
        for j in range(x.shape[1]):
            if j in chosen: continue
            reds=[]
            for prev in chosen:
                pair=tuple(sorted((j,prev)))
                if pair not in cache: cache[pair]=mutual_info_score(b[:,j],b[:,prev])
                reds.append(cache[pair])
            scores[j]=rel[j]-(np.mean(reds) if reds else 0)
        chosen.append(int(np.argmax(scores)))
    return chosen

def model_for(name,seed):
    if name=="Logistic regression":
        return make_pipeline(StandardScaler(),
                LogisticRegression(C=1.,solver="lbfgs",max_iter=1500,random_state=seed))
    if name=="Random forest":
        return RandomForestClassifier(n_estimators=120,min_samples_leaf=2,n_jobs=1,random_state=seed)
    raise ValueError(name)

def run(name,repeats=3,folds=5,k=5):
    x,y,features,source=load_data(name)
    cv=RepeatedStratifiedKFold(n_splits=folds,n_repeats=repeats,random_state=SEED)
    records=[]; selections=defaultdict(list)
    for split_id,(tr,te) in enumerate(cv.split(x,y)):
        a,b=x[tr],y[tr]
        u,v=x[te],y[te]
        for method in METHODS:
            chosen=choose(a,b,method,k) # strictly training-only
            selections[method].append(set(chosen))
            for modelname in MODELS:
                clf=model_for(modelname,SEED+split_id).fit(a[:,chosen],b)
                prob=clf.predict_proba(u[:,chosen])[:,1]
                pred=clf.predict(u[:,chosen])
                records.append(dict(dataset=name,split=split_id,repeat=split_id//folds,
                    fold=split_id%folds,method=method,model=modelname,
                    n_features=len(chosen),accuracy=accuracy_score(v,pred),
                    f1_macro=f1_score(v,pred,average="macro"),
                    logloss_bits=log_loss(v,prob,labels=[0,1])/np.log(2),
                    brier=brier_score_loss(v,prob),
                    selected_indices=",".join(map(str,chosen)),
                    selected_names="|".join(features[j] for j in chosen)))
    stability=[]
    for method,sets in selections.items():
        if method=="All features": continue
        overlaps=[len(s&t)/len(s|t) for i,s in enumerate(sets) for t in sets[i+1:]]
        stability.append(dict(dataset=name,method=method,
                              mean_jaccard=np.mean(overlaps),pair_count=len(overlaps)))
    profile=dict(dataset=name,samples=len(y),features=x.shape[1],
                 positives=int(y.sum()),negatives=int((1-y).sum()),
                 constant_features=int(np.sum(np.ptp(x,axis=0)==0)),source=source)
    return pd.DataFrame(records),pd.DataFrame(stability),profile

def write_figures(summary,out):
    for dataset in summary.dataset.unique():
        d=summary[(summary.dataset==dataset)&(summary.model=="Logistic regression")]
        d=d.set_index("method").reindex(METHODS)
        for metric,caption in [("accuracy","Accuracy (higher is better)"),
                               ("logloss_bits","Log loss in bits (lower is better)")]:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar(range(4),d[f"mean_{metric}"],yerr=d[f"sd_{metric}"],capsize=3)
            ax.set_xticks(range(4),METHODS,rotation=15,ha="right")
            ax.set_ylabel(caption);ax.set_title(f"{dataset}: repeated stratified cross-validation")
            ax.grid(axis="y",alpha=.2);fig.tight_layout()
            fig.savefig(out/f"{dataset}_{metric}.png",dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("both","breast-cancer","ionosphere"),default="both")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--features",type=int,default=5)
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    a=p.parse_args()
    if min(a.repeats,a.features)<1 or a.folds<2:p.error("invalid repeat/fold/feature count")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    names=("breast-cancer","ionosphere") if a.dataset=="both" else (a.dataset,)
    blocks=[]; stabs=[]; profiles=[]
    for name in names:
        b,s,d=run(name,a.repeats,a.folds,a.features)
        blocks.append(b);stabs.append(s);profiles.append(d)
    raw=pd.concat(blocks,ignore_index=True)
    stab=pd.concat(stabs,ignore_index=True)
    summary=raw.groupby(["dataset","model","method"],sort=True).agg(
        folds=("accuracy","size"),
        mean_accuracy=("accuracy","mean"),sd_accuracy=("accuracy","std"),
        mean_f1_macro=("f1_macro","mean"),sd_f1_macro=("f1_macro","std"),
        mean_logloss_bits=("logloss_bits","mean"),
        sd_logloss_bits=("logloss_bits","std"),
        mean_brier=("brier","mean"),sd_brier=("brier","std")).reset_index()
    comparisons=[]
    for (name,model),g in raw.groupby(["dataset","model"]):
        t=g.pivot(index="split",columns="method",values=["accuracy","logloss_bits"])
        da=t["accuracy"]["mRMR"]-t["accuracy"]["Mutual information"]
        dl=t["logloss_bits"]["Mutual information"]-t["logloss_bits"]["mRMR"]
        comparisons.append(dict(dataset=name,model=model,n_folds=len(da),
             mean_accuracy_mrmr_minus_mi=da.mean(),sd_accuracy_mrmr_minus_mi=da.std(),
             mean_logloss_bits_mi_minus_mrmr=dl.mean(),
             sd_logloss_bits_mi_minus_mrmr=dl.std()))
    raw.to_csv(a.output_dir/"fold_results.csv",index=False)
    summary.to_csv(a.output_dir/"method_summary.csv",index=False)
    stab.to_csv(a.output_dir/"selection_stability.csv",index=False)
    pd.DataFrame(comparisons).to_csv(a.output_dir/"paired_fold_differences.csv",index=False)
    pd.DataFrame(profiles).to_csv(a.output_dir/"dataset_profiles.csv",index=False)
    write_figures(summary,a.output_dir)
    f=ROOT/"data"/"ionosphere.csv"
    metadata=dict(seed=SEED,repeats=a.repeats,folds=a.folds,features=a.features,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        sklearn=sklearn.__version__,scipy=scipy.__version__,
        ionosphere_sha256=(hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None),
        mi_estimator="5 training-only quantile bins and plug-in discrete MI",
        mrmr="MID relevance - average pairwise redundancy",
        caution="CV folds overlap; fold standard deviations are descriptive, not independent CIs.",
        sources=profiles)
    (a.output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("=== Dataset profiles ===")
    print(pd.DataFrame(profiles).to_string(index=False))
    print("=== RESEARCH V3 RESULTS ===")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("=== Descriptive paired differences ===")
    print(pd.DataFrame(comparisons).to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("Saved outputs at",a.output_dir.resolve())

if __name__=="__main__":main()
