"""Exact identifiability counterexample for label-free two-sensor monitoring.

Two distinct target worlds yield IDENTICAL unlabeled joint distributions
of (X1,X2), but opposite optimal decisions for a conflicting sensor pair.
The counterexample violates the 'known stable anchor p' assumption,
therefore is NOT a contradiction of the correctly specified estimator.

Run: python research_v7/identifiability_witness.py
"""
from __future__ import annotations
from pathlib import Path
import csv
import math

ROOT=Path(__file__).resolve().parent
WORLD_A={"label":"stable_anchor_secondary_drift","p":0.20,"q":0.40}
WORLD_B={"label":"anchor_drift_secondary_stable","p":0.35,"q":0.30}

def joint_probability(x1,x2,p,q,prior=0.5):
    total=0.
    for y in (0,1):
        a=(1-p) if x1==y else p
        b=(1-q) if x2==y else q
        total+=((prior if y==1 else 1-prior)*a*b)
    return total

def true_posterior(x1,x2,p,q,prior=0.5):
    a1=((1-p) if x1==1 else p)*((1-q) if x2==1 else q)*prior
    a0=((1-p) if x1==0 else p)*((1-q) if x2==0 else q)*(1-prior)
    return a1/(a1+a0)

def witness_rows():
    rows=[]
    for x1 in (0,1):
        for x2 in (0,1):
            pa=joint_probability(x1,x2,WORLD_A["p"],WORLD_A["q"])
            pb=joint_probability(x1,x2,WORLD_B["p"],WORLD_B["q"])
            if not math.isclose(pa,pb,abs_tol=1e-14):
                raise AssertionError("Worlds no longer observationally equivalent")
            rows.append(dict(x1=x1,x2=x2,unlabeled_probability_A=pa,
                unlabeled_probability_B=pb,
                true_posterior_A=true_posterior(x1,x2,WORLD_A["p"],WORLD_A["q"]),
                true_posterior_B=true_posterior(x1,x2,WORLD_B["p"],WORLD_B["q"])))
    conflict=next(r for r in rows if r["x1"]==1 and r["x2"]==0)
    if not (conflict["true_posterior_A"]>.5 and conflict["true_posterior_B"]<.5):
        raise AssertionError("Contradictory target posteriors were not demonstrated")
    return rows

def main():
    out=ROOT/"outputs";out.mkdir(parents=True,exist_ok=True)
    rows=witness_rows()
    with (out/"identifiability_witness.csv").open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print("Identical unlabeled P(X1,X2) in two worlds; incompatible target posteriors:")
    for r in rows:print(r)
    print("Saved exact witness in",out/"identifiability_witness.csv")

if __name__=="__main__":main()
