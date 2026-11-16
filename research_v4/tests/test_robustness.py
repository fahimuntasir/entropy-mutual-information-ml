"""Run python -m unittest discover -s research_v4/tests -v"""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research_v3 import real_world_validation as v3
from research_v4.robustness_analysis import (scores_for_bins,ranked_mrmr,configurations,
    holm_adjust,primary_comparisons,verify_v3_baseline)

class RobustnessTests(unittest.TestCase):
    def test_bins_fit_without_test_data(self):
        a=np.arange(25,dtype=float).reshape(5,5)
        y=np.array([0,1,1,0,1])
        x,rel=scores_for_bins(a,y,3)
        self.assertEqual(x.shape,(5,5))
        self.assertTrue(((x>=0)&(x<=2)).all())
        self.assertEqual(len(rel),5)
    def test_mrmr_duplicate(self):
        g=np.random.default_rng(10);y=g.integers(0,2,3000)
        x0=y^(g.random(3000)<.1)
        x2=y^(g.random(3000)<.2)
        x=np.column_stack((x0,x0,x2,g.integers(0,2,3000)))
        b,rel=scores_for_bins(x,y,5)
        self.assertEqual(ranked_mrmr(b,rel,2),[0,2])
    def test_reference_methods_match_v3(self):
        x,y,_,_=v3.load_data('breast-cancer')
        selections={ (m,k,b):tuple(s) for m,k,b,s in configurations(x[:200],y[:200]) }
        for method,bins in (('Correlation',0),('Mutual information',5),('mRMR',5)):
            self.assertEqual(selections[method,5,bins],tuple(v3.choose(x[:200],y[:200],method,5)))
        self.assertEqual(selections['All features',0,0],tuple(range(x.shape[1])))
    def test_holm(self):
        p=holm_adjust([.001,.03,.02,.5])
        self.assertTrue(np.allclose(p,[.004,.06,.06,.5]))
    def test_corrected_test_null(self):
        rows=[]
        for i in range(6):
            for m in ('Mutual information','mRMR'):
                rows.append(dict(dataset='d',model='m',method=m,k=5,bins=5,
                                 n_train=80,n_test=20,split=i,accuracy=.8,
                                 logloss_bits=.7))
        s=primary_comparisons(pd.DataFrame(rows),repeats=2,folds=3)
        self.assertEqual(len(s),2)
        self.assertTrue((s.p_raw==1).all())
    def test_v3_local_reference(self):
        p=Path(__file__).resolve().parents[2]/'research_v3'/'outputs'/'fold_results.csv'
        if not p.exists():self.skipTest('V3 fold reference missing')
        baseline=pd.read_csv(p)
        self.assertEqual(len(baseline),240)

if __name__=='__main__':unittest.main()
