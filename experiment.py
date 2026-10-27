"""Reproduce all experiments and figures. Run from this directory: python experiment.py."""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import xlogy
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
(ROOT/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
def mi(x,y):
    _,a=np.unique(x,return_inverse=True); _,b=np.unique(y,return_inverse=True)
    c=np.zeros((a.max()+1,b.max()+1)); np.add.at(c,(a,b),1)
    p=c/c.sum(); q=p.sum(1)[:,None]*p.sum(0)[None,:]
    mask=p>0
    return float(np.sum(p[mask]*np.log2(p[mask]/q[mask])))
def joint(X,inds):
    if not inds:return np.zeros(len(X),dtype=int)
    return np.sum(X[:,inds].astype(int)*(2**np.arange(len(inds))),axis=1)
def hb(p):return 0. if p in (0,1) else float(-p*np.log2(p)-(1-p)*np.log2(1-p))
def data(n,p,seed):
    r=np.random.default_rng(seed); y=r.integers(0,2,n)
    a=y^(r.random(n)<p); b=y^(r.random(n)<min(p+.1,.49))
    X=np.column_stack([a,b,a,a^(r.random(n)<.05),r.integers(0,2,(n,16))]).astype(int)
    return X,y
rows=[]; selected={}
for p in [.05,.2,.35]:
 for n in [100,300,1000]:
  for seed in range(30):
   X,y=data(n,p,100000+seed*100+n+int(p*1000)); T,z=data(5000,p,900000+seed*100+n+int(p*1000))
   scores=np.array([mi(X[:,j],y) for j in range(20)])
   corr=np.nan_to_num(np.abs(np.corrcoef(X.T,y)[-1,:-1]))
   # Stable index tie-breaking is deliberate and documented.
   m=list(np.argsort(-scores,kind='stable')[:2]); c=list(np.argsort(-corr,kind='stable')[:2]); g=[]
   for _ in range(2):
    base=mi(joint(X,g),y)
    vals=[-np.inf if j in g else mi(joint(X,g+[j]),y)-base for j in range(20)]
    g.append(int(np.argmax(vals)))
   choices={'All':list(range(20)),'MI':m,'Correlation':c,'Conditional MI':g,'Oracle':[0,1]}
   for method,inds in choices.items():
    model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,random_state=seed).fit(X[:,inds],y)
    acc=accuracy_score(z,model.predict(T[:,inds])); useful=sum(i<4 for i in inds)/len(inds)
    rows.append(dict(p=p,n=n,seed=seed,method=method,accuracy=acc,selected=','.join(map(str,inds)),useful_fraction=useful,both_sources=int(1 in inds and any(j in inds for j in [0,2,3]))))
    selected.setdefault((p,n,method),[]).append(set(inds))
raw=pd.DataFrame(rows);raw.to_csv(ROOT/'classification_runs.csv',index=False)
summary=raw.groupby(['p','n','method']).agg(mean_accuracy=('accuracy','mean'),sd_accuracy=('accuracy','std'),both_sources=('both_sources','mean')).reset_index()
summary.to_csv(ROOT/'classification_summary.csv',index=False)
st=[]
for (p,n,m),sets in selected.items():
 vals=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
 st.append(dict(p=p,n=n,method=m,jaccard=np.mean(vals)))
pd.DataFrame(st).to_csv(ROOT/'stability.csv',index=False)
# Estimation calibration: independent Bernoulli observations, 200 repeats.
cal=[]
for n in [50,100,300,1000,3000]:
 for seed in range(200):
  r=np.random.default_rng(700000+n*1000+seed);y=r.integers(0,2,n);x=y^(r.random(n)<.2)
  noise=r.integers(0,2,(n,20)); estimates=[mi(noise[:,j],y) for j in range(20)]
  cal.append(dict(n=n,seed=seed,signal=mi(x,y),null=estimates[0],max_null=max(estimates)))
cal=pd.DataFrame(cal);cal.to_csv(ROOT/'estimation_runs.csv',index=False)
# Continuous nonlinear case, fresh test set, train-only feature ranking.
def nonlinear(n,seed):
 r=np.random.default_rng(seed); z=r.normal(size=n); y=((np.abs(z)>.6744897501960817)^(r.random(n)<.1)).astype(int)
 w=.3*(2*y-1)+r.normal(size=n)
 return np.column_stack([z,w,r.normal(size=(n,18))]),y
nr=[]
for seed in range(30):
 X,y=nonlinear(500,seed+400);T,z=nonlinear(5000,seed+800)
 score=mutual_info_classif(X,y,discrete_features=False,n_neighbors=3,random_state=seed)/np.log(2)
 corr=np.abs(np.corrcoef(X.T,y)[-1,:-1])
 for method,ind in [('MI',int(np.argmax(score))),('Correlation',int(np.argmax(corr)))]:
  model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,random_state=seed).fit(X[:,[ind]],y)
  nr.append(dict(seed=seed,method=method,feature=ind,accuracy=accuracy_score(z,model.predict(T[:,[ind]])),z_mi=score[0],z_corr=corr[0]))
nr=pd.DataFrame(nr);nr.to_csv(ROOT/'nonlinear_runs.csv',index=False)
# Generate figures from experiment results.
colors={'All':'#9a6c45','MI':'#176a99','Correlation':'#a13968','Conditional MI':'#15806f','Oracle':'#777777'}
fig,axs=plt.subplots(1,3,figsize=(10.5,3.3),sharey=True)
for ax,p in zip(axs,[.05,.2,.35]):
 for m in colors:
  d=summary[(summary.p==p)&(summary.method==m)]
  ax.errorbar(d.n,d.mean_accuracy,yerr=d.sd_accuracy,marker='o',label=m,color=colors[m],capsize=2)
 ax.set_title(f'Noise p = {p}');ax.set_xscale('log');ax.set_xticks([100,300,1000],['100','300','1000']);ax.set_xlabel('Training samples');ax.set_ylim(.48,1)
axs[0].set_ylabel('Test accuracy');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=5,fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'figures/accuracy.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/accuracy.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(5.2,3.3))
for col,label,color in [('signal','Signal estimate','#176a99'),('null','One irrelevant feature','#777777'),('max_null','Maximum of 20 irrelevant features','#a13968')]:
 d=cal.groupby('n')[col].agg(['mean','std']);ax.errorbar(d.index,d['mean'],yerr=d['std'],marker='o',label=label,color=color,capsize=2)
ax.axhline(1-hb(.2),color='#15806f',ls='--',label='True signal MI');ax.set_xscale('log');ax.set_xlabel('Samples');ax.set_ylabel('Mutual information (bits)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/estimation.pdf');fig.savefig(ROOT/'figures/estimation.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.2))
d=raw[(raw.p==.2)&(raw.n==1000)].groupby('method').both_sources.mean().reindex(['MI','Correlation','Conditional MI','Oracle'])
axs[0].bar(range(4),d.values,color=[colors[x] for x in d.index]);axs[0].set_xticks(range(4),['MI','Corr.','Cond. MI','Oracle']);axs[0].set_ylim(0,1.08);axs[0].set_ylabel('Both source groups selected');axs[0].set_title('Redundancy, n = 1000, p = 0.2')
d=nr.groupby('method').accuracy.agg(['mean','std']).reindex(['MI','Correlation']);axs[1].bar(range(2),d['mean'],yerr=d['std'],color=[colors[x] for x in d.index],capsize=4);axs[1].set_xticks(range(2),d.index);axs[1].set_ylim(.45,1);axs[1].set_ylabel('Test accuracy');axs[1].set_title('Continuous nonlinear task, n = 500');fig.tight_layout();fig.savefig(ROOT/'figures/limits.pdf');fig.savefig(ROOT/'figures/limits.png');plt.close(fig)
meta={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'true_signal_mi':1-hb(.2),'repeats':30,'calibration_repeats':200,'test_size':5000}
(ROOT/'environment.json').write_text(json.dumps(meta,indent=2))
print(summary[summary.p==.2].to_string(index=False));print(nr.groupby('method').agg(accuracy=('accuracy','mean'),sd=('accuracy','std'),nonlinear_selected=('feature',lambda s:float((s==0).mean()))));print(cal.groupby('n')[['signal','null','max_null']].mean())
