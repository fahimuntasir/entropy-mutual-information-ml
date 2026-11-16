"""Research V6: fixed source-reliability shift in a two-sensor binary model.

Reproduces the original V1/V2 noisy-binary feature layout. X2 has training
noise q=.30, but its test noise changes. No target labels or test data are
used to train/select features. This changes P(X2|Y) and P(Y|X1,X2), so it is
not standard covariate shift.
Run: python research_v6/sensor_shift_experiment.py
"""
from __future__ import annotations
import argparse, json, platform
from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import expit
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parent
P=.20; Q_TRAIN=.30
Q_TEST=(.30,.35,.40,.45,.50,.60)
TRAIN_SIZES=(100,300,1000)
METHODS=("X1 only","Marginal MI","Greedy CMI","Oracle X1+X2")

def binary_data(n,p,q,seed):
    """Original synthetic 20-feature model, permitting X2 reliability shift."""
    r=np.random.default_rng(seed)
    y=r.integers(0,2,n)
    x1=y^(r.random(n)<p)
    x2=y^(r.random(n)<q)
    x=np.column_stack((x1,x2,x1,x1^(r.random(n)<.05),r.integers(0,2,(n,16)))).astype(int)
    return x,y

def empirical_mi_bits(x,y):
    _,a=np.unique(x,return_inverse=True)
    _,b=np.unique(y,return_inverse=True)
    c=np.zeros((a.max()+1,b.max()+1),dtype=float)
    np.add.at(c,(a,b),1)
    p=c/c.sum()
    independent=p.sum(1)[:,None]*p.sum(0)[None,:]
    mask=p>0
    return float(np.sum(p[mask]*np.log2(p[mask]/independent[mask])))

def encode_joint(x,idx):
    if not idx: return np.zeros(x.shape[0],dtype=np.int64)
    return np.sum(x[:,idx].astype(np.int64)*(2**np.arange(len(idx))),axis=1)

def selectors(x,y):
    scores=np.array([empirical_mi_bits(x[:,j],y) for j in range(x.shape[1])])
    marginal=np.argsort(-scores,kind="stable")[:2].astype(int).tolist()
    chosen=[]
    for _ in range(2):
        base=empirical_mi_bits(encode_joint(x,chosen),y)
        increments=[-np.inf if j in chosen else empirical_mi_bits(encode_joint(x,chosen+[j]),y)-base for j in range(x.shape[1])]
        chosen.append(int(np.argmax(increments)))
    return {"X1 only":[0],"Marginal MI":marginal,"Greedy CMI":chosen,"Oracle X1+X2":[0,1]}

def bayes_probability(x1,x2,p,q_train):
    logit=np.log((1-p)/p)*(2*np.asarray(x1)-1)
    if x2 is not None:logit=logit+np.log((1-q_train)/q_train)*(2*np.asarray(x2)-1)
    return expit(logit)

def exact_expected_logloss_bits(q_test,p=P,q_train=Q_TRAIN):
    single=joint=0.
    for y,x1,x2 in product((0,1),repeat=3):
        weight=.5*((1-p) if y==x1 else p)*((1-q_test) if y==x2 else q_test)
        a=float(bayes_probability(np.array([x1]),None,p,q_train)[0])
        b=float(bayes_probability(np.array([x1]),np.array([x2]),p,q_train)[0])
        single+=weight*(-np.log2(a if y else 1-a))
        joint+=weight*(-np.log2(b if y else 1-b))
    return single,joint

def analytic_table():
    rows=[]
    for q in Q_TEST:
        single,stale=exact_expected_logloss_bits(q)
        _,target=exact_expected_logloss_bits(q,q_train=q)
        rows.append(dict(q_test=q,baseline_bits=single,stale_two_source_bits=stale,
            optimal_target_two_source_bits=target,target_conditional_mi_bits=single-target,
            improvement_bits=single-stale))
    d0=np.diff(exact_expected_logloss_bits(0))[0]
    d1=np.diff(exact_expected_logloss_bits(1))[0]
    crossing=float(-d0/(d1-d0))
    assert .35<crossing<.45
    return pd.DataFrame(rows),crossing

def trial_metrics(trials=30,train_sizes=TRAIN_SIZES,test_size=5000):
    rows=[]
    for n in train_sizes:
        for trial in range(trials):
            train_seed=100000+trial*100+n+int(P*1000)
            test_seed=900000+trial*100+n+int(P*1000)
            train_x,train_y=binary_data(n,P,Q_TRAIN,train_seed)
            chosen=selectors(train_x,train_y)
            models={}
            for method,indices in chosen.items():
                model=LogisticRegression(C=10.,solver="lbfgs",max_iter=1000,random_state=trial)
                model.fit(train_x[:,indices],train_y)
                models[method]=model
            for q in Q_TEST:
                test_x,test_y=binary_data(test_size,P,q,test_seed)
                for method in METHODS:
                    indices=chosen[method]
                    probs=models[method].predict_proba(test_x[:,indices])[:,1]
                    rows.append(dict(n=n,trial=trial,q_test=q,method=method,
                        accuracy=float(accuracy_score(test_y,(probs>=.5).astype(int))),
                        logloss_bits=float(log_loss(test_y,probs,labels=[0,1])/np.log(2)),
                        brier=float(brier_score_loss(test_y,probs)),
                        indices=",".join(map(str,indices)),
                        selected_both_sources=int(1 in indices and any(j in indices for j in (0,2,3))),
                        train_seed=train_seed,test_seed=test_seed))
        print(f"n={n}: {trials} independent runs complete",flush=True)
    return pd.DataFrame(rows)

def paired_results(runs):
    frame=runs.pivot(index=["n","trial","q_test"],columns="method",
                     values=["accuracy","logloss_bits","brier"])
    records=[]
    for n in sorted(runs.n.unique()):
        for q in sorted(runs.q_test.unique()):
            part=frame.loc[(n,slice(None),q)]
            for method,baseline in (("Greedy CMI","Marginal MI"),
                                    ("Greedy CMI","X1 only"),
                                    ("Oracle X1+X2","X1 only")):
                gains={
                    "accuracy_gain":part["accuracy"][method]-part["accuracy"][baseline],
                    "logloss_gain_bits":part["logloss_bits"][baseline]-part["logloss_bits"][method],
                    "brier_gain":part["brier"][baseline]-part["brier"][method]}
                row=dict(n=n,q_test=q,method=method,baseline=baseline,trials=len(part))
                for name,series in gains.items():
                    v=series.to_numpy(dtype=float);mean=float(v.mean())
                    half=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
                    row[name]=mean;row[name+"_ci95_low"]=mean-half;row[name+"_ci95_high"]=mean+half
                records.append(row)
    return pd.DataFrame(records)

def figures(paired,analytic,crossing,output):
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    ax.plot(analytic.q_test,analytic.improvement_bits,"o-",label="Fixed training Bayes / exact")
    for n in TRAIN_SIZES:
        d=paired[(paired.n==n)&(paired.method=="Greedy CMI")&(paired.baseline=="X1 only")].sort_values("q_test")
        ax.errorbar(d.q_test,d.logloss_gain_bits,
                    yerr=[d.logloss_gain_bits-d.logloss_gain_bits_ci95_low,
                          d.logloss_gain_bits_ci95_high-d.logloss_gain_bits],
                    fmt="o--",capsize=2,label=f"Greedy CMI, n={n}")
    ax.axhline(0,color="black",lw=.8)
    ax.axvline(crossing,color="gray",ls=":",label=f"Exact crossover {crossing:.3f}")
    ax.set(xlabel="Test-time X2 error (training X2 error=0.30)",
           ylabel="Log-loss reduction versus X1 only (bits; >0 better)",
           title="Reliability shift: training-time information turns into regret")
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/"sensor_shift_logloss_crossover.png",dpi=170);plt.close(fig)
    d=paired[(paired.n==1000)&(paired.method=="Greedy CMI")&(paired.baseline=="X1 only")].sort_values("q_test")
    fig,ax=plt.subplots(figsize=(7.1,4.2))
    ax.errorbar(d.q_test,d.accuracy_gain,
        yerr=[d.accuracy_gain-d.accuracy_gain_ci95_low,
              d.accuracy_gain_ci95_high-d.accuracy_gain],
        fmt="s-",capsize=3)
    ax.axhline(0,color="black",lw=1)
    ax.set(xlabel="Test-time X2 error",ylabel="CMI accuracy gain versus X1 only",
           title="Hard accuracy can stay constant while log loss reverses")
    ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/"sensor_shift_accuracy.png",dpi=170);plt.close(fig)

def run(trials=30,test_size=5000,output=ROOT/"outputs"):
    if trials<2:raise ValueError("At least 2 independent trials required")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    exact,crossing=analytic_table()
    raw=trial_metrics(trials=trials,test_size=test_size)
    paired=paired_results(raw)
    summary=raw.groupby(["n","q_test","method"],as_index=False).agg(
        mean_accuracy=("accuracy","mean"),sd_accuracy=("accuracy","std"),
        mean_logloss_bits=("logloss_bits","mean"),sd_logloss_bits=("logloss_bits","std"),
        mean_brier=("brier","mean"),both_sources_rate=("selected_both_sources","mean"))
    for filename,df in (("trial_results.csv",raw),("method_summary.csv",summary),
        ("paired_comparisons.csv",paired),("exact_bayes_reference.csv",exact)):
        df.to_csv(output/filename,index=False)
    figures(paired,exact,crossing,output)
    meta=dict(p=P,q_train=Q_TRAIN,q_test=list(Q_TEST),train_sizes=list(TRAIN_SIZES),
              trials=trials,test_size=test_size,analytic_crossover_q=crossing,
              primary_cases=["n=1000, q_test=.3","n=1000, q_test=.5"],
              shift_type="P(X2|Y) changes and P(Y|X1,X2) changes; NOT ordinary covariate shift",
              inference="paired t intervals across independent replicate seeds at fixed q",
              python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              sklearn=sklearn.__version__,pandas=pd.__version__)
    (output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Exact crossover:",crossing)
    print(exact.to_string(index=False))
    print("Primary paired comparisons n=1000:")
    print(paired[(paired.n==1000)&(paired.q_test.isin([.3,.5]))].to_string(index=False))
    print("Total held-out evaluations:",len(raw))
    return raw,paired,summary,exact,meta

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=30)
    parser.add_argument("--test-size",type=int,default=5000)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    args=parser.parse_args()
    run(args.trials,args.test_size,args.output_dir)
