"""Unit tests for the independent Research V3 feature-selection benchmark."""
import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from real_world_validation import load_data, quantile_bins, correlation, choose, run

class ValidationTests(unittest.TestCase):
    def test_breast_cancer_shape(self):
        x,y,names,_=load_data("breast-cancer")
        self.assertEqual(x.shape,(569,30))
        self.assertEqual(len(names),30)
        self.assertEqual(int(y.sum()),357)
    def test_ionosphere_shape(self):
        x,y,names,_=load_data("ionosphere")
        self.assertEqual(x.shape,(351,34))
        self.assertEqual(int(y.sum()),225)
        self.assertEqual(int(np.sum(np.ptp(x,axis=0)==0)),1)
    def test_constant_column(self):
        r=np.random.default_rng(10)
        x=np.column_stack((np.ones(120),r.normal(size=(120,3))))
        y=(x[:,2]>0).astype(int)
        self.assertTrue(np.all(quantile_bins(x)[:,0]==0))
        self.assertEqual(correlation(x,y)[0],0.)
    def test_mrmr_rejects_exact_duplicate(self):
        r=np.random.default_rng(2026)
        y=r.integers(0,2,4000)
        x1=y^(r.random(4000)<.1)
        x2=y^(r.random(4000)<.24)
        x=np.column_stack((x1,x1.copy(),x2,r.integers(0,2,4000)))
        self.assertEqual(choose(x,y,"Mutual information",2),[0,1])
        self.assertEqual(choose(x,y,"mRMR",2),[0,2])
    def test_small_run(self):
        raw,stable,profile=run("breast-cancer",repeats=1,folds=2,k=3)
        self.assertEqual(len(raw),16)
        self.assertTrue(raw.accuracy.between(0,1).all())
        self.assertEqual(len(stable),3)
        self.assertEqual(profile["samples"],569)

if __name__=="__main__": unittest.main()
