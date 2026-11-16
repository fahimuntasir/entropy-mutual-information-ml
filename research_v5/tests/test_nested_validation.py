"""Research V5 tests. Run python -m unittest discover -s research_v5/tests -v."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research_v3 import real_world_validation as v3
from research_v5.nested_validation import (
    DATASETS, load_dataset, feature_plans, inner_tune,
    exact_signflip, summarize, model_for
)

class NestedValidationTests(unittest.TestCase):
    def test_five_dataset_shapes(self):
        expected={
            "breast-cancer":(569,30),
            "ionosphere":(351,34),
            "sonar":(208,60),
            "spambase":(4601,57),
            "digits-3-vs-8":(357,64),
        }
        for name in DATASETS:
            x,y,features,info=load_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(x.shape,expected[name])
                self.assertEqual(len(features),x.shape[1])
                self.assertEqual(len(y),x.shape[0])
                self.assertEqual(set(np.unique(y)),{0,1})
                self.assertTrue(np.isfinite(x).all())
                self.assertEqual(info["samples"],len(y))
    def test_spambase_excludes_group_column(self):
        x,y,_,info=load_dataset("spambase")
        self.assertEqual(x.shape[1],57)
        self.assertEqual(int(y.sum()),1813)
        self.assertIn("rgroup",info["ignored_columns"])
    def test_sonar_classes(self):
        _,y,_,_=load_dataset("sonar")
        self.assertEqual(int(y.sum()),111)
    def test_digits_binary_classes(self):
        _,y,_,_=load_dataset("digits-3-vs-8")
        self.assertEqual(int(y.sum()),174)
    def test_reference_selector_compatibility(self):
        x,y,_,_=load_dataset("breast-cancer")
        plans=feature_plans(x[:200],y[:200],(5,),(5,))
        for m,b in (("Correlation",0),("Mutual information",5),("mRMR",5)):
            self.assertEqual(list(plans[(m,5,b)]),v3.choose(x[:200],y[:200],m,5))
    def test_inner_tune_only_train_labels(self):
        x,y,_,_=load_dataset("breast-cancer")
        result, scores=inner_tune(
            x[:140],y[:140],"Logistic regression",outer_id=0,
            inner_folds=2,k_values=(3,),bin_values=(3,))
        self.assertEqual(set(result),set(("All features","Correlation","Mutual information","mRMR")))
        self.assertEqual(len(scores),4)
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))
    def test_signflip_five_datasets_resolution(self):
        val,p=exact_signflip(np.array([1.,2.,3.,4.,5.]))
        self.assertAlmostEqual(p,0.0625)
        self.assertAlmostEqual(val,3.)
    def test_models_return_probabilities(self):
        x,y,_,_=load_dataset("breast-cancer")
        for method in ("Logistic regression","Random forest"):
            m=model_for(method,100).fit(x[:300],y[:300])
            p=m.predict_proba(x[300:310])[:,1]
            self.assertTrue(((p>=0)&(p<=1)).all())

if __name__=="__main__":unittest.main()
