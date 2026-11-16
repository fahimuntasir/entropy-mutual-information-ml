"""V7: unlabeled disagreement-based sensor adaptation, with assumption stress tests.

This is an *existing model-based estimator*, not a claimed new algorithm.
It assumes known anchor error p=0.20, a balanced training prior, symmetric
channel errors, and conditionally independent errors. Target labels are
NEVER used to estimate q. They are used only after prediction for evaluation.

Usage: python research_v7/adaptive_reliability.py
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parent
P_ANCHOR=0.20
Q_TRAIN=0.30
PRIOR_TRAIN=0.50
CROSSOVER=0.3961033433919364  # exact V6 threshold with p=.20 and qtrain=.30
WINDOWS=(50,200,1000)
METHODS=("Fixed fusion","X1 fallback","Agreement plug-in",
         "Shrinkage plug-in","Agreement gate","True-distribution oracle")
SCENARIOS={
    "stable":dict(p=.20,q=.30,rho=0.,prior=.50),
    "moderate_drift":dict(p=.20,q=.40,rho=0.,prior=.50),
    "sensor_degraded":dict(p=.20,q=.50,rho=0.,prior=.50),
    "sensor_inverted":dict(p=.20,q=.65,rho=0.,prior=.50),
    "dependent_errors":dict(p=.20,q=.50,rho=.35,prior=.50),
    "anchor_drift":dict(p=.35,q=.30,rho=0.,prior=.50),
    "prior_shift":dict(p=.20,q=.30,rho=0.,prior=.75),
}

def generate(n,scenario,seed):
    """Binary symmetric channels with optional shared-error dependence.

    With probability rho, E2=E1; otherwise E2 is independent with q_ind
    chosen so that marginal P(E2=1)=q. In particular, correlated errors
    violate the disagreement estimator's conditional-independence premise.
    """
    p,q,rho,prior=(scenario[k] for k in ("p","q","rho","prior"))
    if not (0<p<1 and 0<q<1 and 0<=rho<1 and 0<prior<1):
        raise ValueError("Invalid Bernoulli parameters")
    qind=(q-rho*p)/(1-rho)
    if not 0<=qind<=1:raise ValueError("Infeasible marginal q for this rho and p")
    rng=np.random.default_rng(seed)
    y=(rng.random(n)<prior).astype(np.int8)
    e1=(rng.random(n)<p).astype(np.int8)
    e2ind=(rng.random(n)<qind).astype(np.int8)
    shared=rng.random(n)<rho
    e2=np.where(shared,e1,e2ind)
    return (y^e1).astype(np.int8),(y^e2).astype(np.int8),y

def estimate_q_from_unlabeled(x1,x2,p_anchor=P_ANCHOR):
    """Moment-based identification from P(X1!=X2)=p+q-2pq.

    Requires known stable anchor error p and conditional independence.
    """
    d=float(np.mean(np.asarray(x1)!=np.asarray(x2)))
    q_raw=(d-p_anchor)/(1-2*p_anchor)
    return float(np.clip(q_raw,1e-4,1-1e-4)),d,float(q_raw)

def posterior(x1,x2,p=P_ANCHOR,q=Q_TRAIN,prior=PRIOR_TRAIN):
    """Fixed-assumption independent-channel Bayes posterior."""
    x1=np.asarray(x1,dtype=float)
    logit=np.log(prior/(1-prior))+np.log((1-p)/p)*(2*x1-1)
    if x2 is not None:
        x2=np.asarray(x2,dtype=float)
        logit=logit+np.log((1-q)/q)*(2*x2-1)
    return expit(logit)

def oracle_posterior(x1,x2,scenario):
    """True target-domain posterior including known dependence and prior.

    This is an inaccessible upper benchmark: all true target parameters,
    including dependence and prevalence, are supplied to the oracle.
    """
    p,q,rho,prior=(scenario[k] for k in ("p","q","rho","prior"))
    qind=(q-rho*p)/(1-rho)
    table=np.empty((2,2),dtype=float)
    def error_mass(e1,e2):
        pe1=p if e1 else 1-p
        pe2ind=qind if e2 else 1-qind
        return pe1*(rho*(e1==e2)+(1-rho)*pe2ind)
    for a in (0,1):
        for b in (0,1):
            w0=(1-prior)*error_mass(a,b)
            w1=prior*error_mass(1-a,1-b)
            table[a,b]=w1/(w0+w1)
    return table[np.asarray(x1,dtype=int),np.asarray(x2,dtype=int)]

def predictions(x1,x2,qhat,window,scenario):
    """Baselines and two plug-in adaptations; no target labels accessed."""
    shrinkage=window/(window+200.0)  # fixed pre-experiment; no test-label tuning
    qshrink=Q_TRAIN+shrinkage*(qhat-Q_TRAIN)
    return {
        "Fixed fusion":posterior(x1,x2),
        "X1 fallback":posterior(x1,None),
        "Agreement plug-in":posterior(x1,x2,q=qhat),
        "Shrinkage plug-in":posterior(x1,x2,q=qshrink),
        "Agreement gate":posterior(x1,None) if qhat>=CROSSOVER else posterior(x1,x2),
        "True-distribution oracle":oracle_posterior(x1,x2,scenario),
    }

def score(y,prob):
    """Held-out log loss bits, Brier score, and hard classification accuracy."""
    prob=np.clip(np.asarray(prob,dtype=float),1e-12,1-1e-12)
    y=np.asarray(y,dtype=int)
    losses=-np.where(y==1,np.log2(prob),np.log2(1-prob))
    return float(np.mean(losses)),float(np.mean((prob-y)**2)),float(np.mean((prob>=.5)==y))

def run_trials(trials=100,test_size=4000,windows=WINDOWS,scenarios=SCENARIOS):
    rows=[]
    for sid,(name,scenario) in enumerate(scenarios.items()):
        for trial in range(trials):
            seed=20261116+sid*100000+trial*101
            x1m,x2m,_hidden_labels=generate(max(windows),scenario,seed+1)
            x1t,x2t,ytest=generate(test_size,scenario,seed+2)
            oracle=oracle_posterior(x1t,x2t,scenario)
            for m in windows:
                qhat,disagree,qraw=estimate_q_from_unlabeled(x1m[:m],x2m[:m])
                preds=predictions(x1t,x2t,qhat,m,scenario)
                for method,prob in preds.items():
                    loss,brier,accuracy=score(ytest,prob)
                    rows.append(dict(scenario=name,trial=trial,window=m,method=method,
                        q_true=scenario["q"],p_true=scenario["p"],
                        rho_true=scenario["rho"],prior_true=scenario["prior"],
                        q_hat=qhat,q_hat_unclipped=qraw,monitor_disagreement=disagree,
                        logloss_bits=loss,brier=brier,accuracy=accuracy,
                        monitor_seed=seed+1,test_seed=seed+2))
        print(f"{name}: {trials} replicates completed",flush=True)
    return pd.DataFrame(rows)

def paired_comparisons(runs):
    comparisons=[]
    pivot=runs.pivot(index=["scenario","trial","window"],columns="method",
                     values=["logloss_bits","brier","accuracy"])
    pairs=[
        ("Agreement plug-in","Fixed fusion"),
        ("Agreement plug-in","X1 fallback"),
        ("Shrinkage plug-in","Fixed fusion"),
        ("Agreement gate","Fixed fusion"),
        ("Agreement gate","X1 fallback"),
        ("True-distribution oracle","Fixed fusion"),
    ]
    for (scenario,m),g in pivot.groupby(level=["scenario","window"]):
        for method,baseline in pairs:
            obj=dict(scenario=scenario,window=int(m),method=method,baseline=baseline,
                     replicates=len(g))
            for metric in ("logloss_bits","brier","accuracy"):
                a=g[metric][method].to_numpy()
                b=g[metric][baseline].to_numpy()
                delta=(a-b) if metric=="accuracy" else (b-a)
                mean=float(delta.mean())
                se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
                h=float(student_t.ppf(.975,len(delta)-1)*se)
                obj["gain_"+metric]=mean
                obj["ci95_"+metric+"_lower"]=mean-h
                obj["ci95_"+metric+"_upper"]=mean+h
            comparisons.append(obj)
    return pd.DataFrame(comparisons)

def holm(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p);out=np.empty(len(p))
    current=0.
    for rank,index in enumerate(order):
        current=max(current,(len(p)-rank)*p[index])
        out[index]=min(current,1.)
    return out

def planned_tests(runs,window=200):
    """Four planned paired tests; seed-level units independent *within scenario*.

    Positive log-loss gain means adaptive better than baseline. Correlation
    across different scenarios is not treated as independent evidence.
    """
    tasks=[
      ("stable","Agreement plug-in","Fixed fusion"),
      ("sensor_degraded","Agreement plug-in","Fixed fusion"),
      ("sensor_degraded","Agreement plug-in","X1 fallback"),
      ("dependent_errors","Agreement plug-in","X1 fallback"),
    ]
    rows=[]
    for scenario,method,baseline in tasks:
        d=runs[(runs.scenario==scenario)&(runs.window==window)]
        p=d.pivot(index="trial",columns="method",values="logloss_bits")
        gains=(p[baseline]-p[method]).to_numpy()
        mu=float(gains.mean());sd=float(gains.std(ddof=1))
        se=sd/np.sqrt(len(gains))
        t_stat=mu/se if se>0 else 0.
        pval=float(2*student_t.sf(abs(t_stat),df=len(gains)-1)) if se>0 else 1.
        h=float(student_t.ppf(.975,len(gains)-1)*se)
        rows.append(dict(scenario=scenario,method=method,baseline=baseline,
            monitor_window=window,trials=len(gains),gain_logloss_bits=mu,
            ci95_lower=mu-h,ci95_upper=mu+h,p_two_sided=pval))
    out=pd.DataFrame(rows)
    out["p_holm_4"]=holm(out.p_two_sided.to_numpy())
    return out

def visuals(summary,comparison,out):
    names=["stable","moderate_drift","sensor_degraded","sensor_inverted",
           "dependent_errors","anchor_drift","prior_shift"]
    data=summary[(summary.window==200)&summary.scenario.isin(names)]
    methods=["Fixed fusion","X1 fallback","Agreement plug-in",
             "Shrinkage plug-in","Agreement gate","True-distribution oracle"]
    values=np.array([[float(data[(data.scenario==s)&(data.method==method)].mean_logloss_bits.iloc[0])
                      for method in methods] for s in names])
    fig,ax=plt.subplots(figsize=(11.2,5.3))
    x=np.arange(len(names));width=.13
    for j,method in enumerate(methods):
        ax.bar(x+(j-2.5)*width,values[:,j],width,label=method)
    ax.set_xticks(x,[x.replace("_"," ") for x in names],rotation=20,ha="right")
    ax.set_ylabel("Held-out log loss (bits; lower is better)")
    ax.set_title("Unlabeled window M=200: adaptation can succeed or fail")
    ax.grid(axis="y",alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(out/"logloss_across_failure_modes.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    for s in ["stable","sensor_degraded","sensor_inverted","dependent_errors","anchor_drift"]:
        d=summary[(summary.method=="Agreement plug-in")&(summary.scenario==s)].sort_values("window")
        ax.plot(d.window,d.mean_qhat,"o-",label=s.replace("_"," "))
    ax.set_xscale("log");ax.set_xticks(WINDOWS,list(map(str,WINDOWS)))
    ax.axhline(.3,color="gray",ls=":",label="training q=.30")
    ax.set_ylabel("Estimated second-sensor error from unlabeled agreement")
    ax.set_xlabel("Unlabeled monitoring sample count")
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/"unlabeled_reliability_estimates.png",dpi=165);plt.close(fig)

def run(trials=100,test_size=4000,output=ROOT/"outputs"):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=run_trials(trials,test_size)
    summary=raw.groupby(["scenario","window","method"],as_index=False).agg(
       trials=("trial","size"),mean_accuracy=("accuracy","mean"),
       mean_logloss_bits=("logloss_bits","mean"),
       sd_logloss_bits=("logloss_bits","std"),
       mean_brier=("brier","mean"),mean_qhat=("q_hat","mean"),
       sd_qhat=("q_hat","std"))
    compare=paired_comparisons(raw)
    planned=planned_tests(raw)
    raw.to_csv(output/"trial_results.csv",index=False)
    summary.to_csv(output/"method_summary.csv",index=False)
    compare.to_csv(output/"paired_comparisons.csv",index=False)
    planned.to_csv(output/"planned_tests.csv",index=False)
    visuals(summary,compare,output)
    meta=dict(trials=trials,test_size=test_size,windows=list(WINDOWS),
       methods=list(METHODS),scenarios=SCENARIOS,reference_p=P_ANCHOR,
       train_q=Q_TRAIN,threshold=CROSSOVER,shrinkage_denominator=200,
       identification="D=p+(1-2*p)*q, valid only if anchor p known and errors conditionally independent",
       prior_assumption=.5,adaptation_uses_target_labels=False,
       evaluation_labels_used_only_for_scoring=True,
       four_planned_tests=[["stable","plug-in vs fixed"],
                           ["sensor_degraded","plug-in vs fixed"],
                           ["sensor_degraded","plug-in vs fallback"],
                           ["dependent_errors","plug-in vs fallback"]],
       python=platform.python_version(),numpy=np.__version__,
       scipy=scipy.__version__,pandas=pd.__version__)
    (output/"experiment_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("\n=== PRE-DECLARED TESTS ===")
    print(planned.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\n=== WINDOW M=200 SUMMARY ===")
    print(summary[summary.window==200][["scenario","method","mean_logloss_bits","mean_accuracy","mean_qhat"]].to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print(f"\nRows: raw={len(raw)}, summary={len(summary)}, paired={len(compare)}; saved {output}",flush=True)
    return raw,summary,compare,planned

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=int,default=100)
    p.add_argument("--test-size",type=int,default=4000)
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    args=p.parse_args()
    if args.trials<2 or args.test_size<100:p.error("trials>=2 and test-size>=100 required")
    run(args.trials,args.test_size,args.output_dir)
