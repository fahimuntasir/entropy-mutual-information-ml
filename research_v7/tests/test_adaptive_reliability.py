"""V7 verification: identification assumptions, oracle, leakage, failure modes."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_reliability import (
    SCENARIOS,WINDOWS,METHODS,P_ANCHOR,Q_TRAIN,CROSSOVER,
    generate,estimate_q_from_unlabeled,posterior,oracle_posterior,
    predictions,score,run_trials,paired_comparisons,planned_tests,
)

class AdaptiveReliabilityTests(unittest.TestCase):
    def test_known_disagreement_formula(self):
        self.assertAlmostEqual(P_ANCHOR+(1-2*P_ANCHOR)*Q_TRAIN,.38)
        self.assertTrue(.39<CROSSOVER<.40)
    def test_data_shape_and_marginal_noise(self):
        p=SCENARIOS["stable"]
        x1,x2,y=generate(80000,p,42)
        self.assertEqual(x1.shape,y.shape)
        self.assertAlmostEqual(np.mean(x1!=y),p["p"],delta=.005)
        self.assertAlmostEqual(np.mean(x2!=y),p["q"],delta=.005)
    def test_unlabeled_agreement_identifies_q_under_assumptions(self):
        x1,x2,_=generate(100000,SCENARIOS["sensor_degraded"],7)
        qhat,disagreement,raw=estimate_q_from_unlabeled(x1,x2)
        self.assertAlmostEqual(qhat,.5,delta=.012)
        self.assertAlmostEqual(disagreement,.5,delta=.012)
    def test_correlation_violates_identifiability(self):
        case=SCENARIOS["dependent_errors"]
        x1,x2,y=generate(200000,case,8)
        self.assertAlmostEqual(np.mean(x2!=y),.5,delta=.005)
        qhat,_,_=estimate_q_from_unlabeled(x1,x2)
        self.assertAlmostEqual(qhat,.31333333333,delta=.012)
        self.assertGreater(abs(qhat-case["q"]),.15)
    def test_anchor_change_biases_q_estimate(self):
        x1,x2,_=generate(150000,SCENARIOS["anchor_drift"],32)
        qhat,_,_=estimate_q_from_unlabeled(x1,x2)
        self.assertAlmostEqual(qhat,.40,delta=.012)
        self.assertNotAlmostEqual(qhat,.30,delta=.05)
    def test_true_oracle_matches_independent_formula(self):
        case=SCENARIOS["sensor_degraded"]
        x1,x2,_=generate(500,case,17)
        a=oracle_posterior(x1,x2,case)
        b=posterior(x1,x2,p=case["p"],q=case["q"],prior=case["prior"])
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    def test_predictor_never_receives_labels(self):
        case=SCENARIOS["sensor_degraded"]
        x1,x2,y=generate(200,case,18)
        methods=predictions(x1,x2,.5,200,case)
        self.assertEqual(set(methods),set(METHODS))
        self.assertTrue(all(len(p)==len(x1) for p in methods.values()))
        self.assertTrue(all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                            for p in methods.values()))
    def test_stale_v6_problem_and_adaptive_remedy(self):
        case=SCENARIOS["sensor_degraded"]
        x1,x2,y=generate(40000,case,25)
        probs=predictions(x1,x2,.50,1000,case)
        loss_fixed=score(y,probs["Fixed fusion"])[0]
        loss_adapt=score(y,probs["Agreement plug-in"])[0]
        loss_fallback=score(y,probs["X1 fallback"])[0]
        self.assertGreater(loss_fixed,loss_fallback+.05)
        self.assertAlmostEqual(loss_adapt,loss_fallback,delta=.008)
    def test_small_full_experiment_and_counts(self):
        raw=run_trials(trials=2,test_size=100,windows=WINDOWS,scenarios=SCENARIOS)
        self.assertEqual(len(raw),7*2*3*6)
        self.assertTrue(raw.logloss_bits.between(0,5).all())
        paired=paired_comparisons(raw)
        self.assertEqual(len(paired),7*3*6)
        contrasts=planned_tests(raw)
        self.assertEqual(len(contrasts),4)
    def test_reproducibility(self):
        subset={"stable":SCENARIOS["stable"]}
        a=run_trials(2,120,windows=(50,),scenarios=subset)
        b=run_trials(2,120,windows=(50,),scenarios=subset)
        self.assertTrue(a.equals(b))

if __name__=="__main__":
    unittest.main()
