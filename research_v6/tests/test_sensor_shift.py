"""V6 tests: exact theory, source selection, shift and reproducibility."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sensor_shift_experiment import (
    P,Q_TRAIN,binary_data,empirical_mi_bits,selectors,bayes_probability,
    exact_expected_logloss_bits,analytic_table,trial_metrics,paired_results)

class SensorShiftTests(unittest.TestCase):
    def test_original_duplicate_structure(self):
        x,y=binary_data(400,P,Q_TRAIN,42)
        self.assertEqual(x.shape,(400,20))
        np.testing.assert_array_equal(x[:,0],x[:,2])
        self.assertEqual(len(y),400)
    def test_empirical_information_identity(self):
        y=np.array([0,0,1,1]*50)
        self.assertAlmostEqual(empirical_mi_bits(y,y),1.,places=12)
        self.assertEqual(empirical_mi_bits(np.zeros_like(y),y),0.)
    def test_selection_recovers_independent_source(self):
        x,y=binary_data(4000,.2,.3,12)
        self.assertEqual(selectors(x,y)["Marginal MI"],[0,2])
        self.assertEqual(selectors(x,y)["Greedy CMI"],[0,1])
    def test_population_expected_logloss(self):
        single,two=exact_expected_logloss_bits(.3)
        self.assertAlmostEqual(single,.7219280948873623,places=12)
        self.assertAlmostEqual(two,.6451769718917554,places=12)
    def test_analytic_crossing(self):
        table,threshold=analytic_table()
        self.assertTrue(.35<threshold<.45)
        self.assertGreater(table.loc[table.q_test==.3,"improvement_bits"].iloc[0],0)
        self.assertLess(table.loc[table.q_test==.5,"improvement_bits"].iloc[0],0)
    def test_target_information_nonnegative(self):
        d,_=analytic_table()
        self.assertTrue((d.target_conditional_mi_bits>=-1e-12).all())
        self.assertAlmostEqual(d.loc[d.q_test==.5,"target_conditional_mi_bits"].iloc[0],0,places=12)
        self.assertGreater(d.loc[d.q_test==.6,"target_conditional_mi_bits"].iloc[0],0)
    def test_fixed_posterior_hard_predictions_do_not_change(self):
        for x1 in (0,1):
            for x2 in (0,1):
                one=bayes_probability(np.array([x1]),None,P,Q_TRAIN)[0]>=.5
                both=bayes_probability(np.array([x1]),np.array([x2]),P,Q_TRAIN)[0]>=.5
                self.assertEqual(bool(one),bool(both))
    def test_repeated_small_run_is_exact(self):
        a=trial_metrics(2,(100,),90)
        b=trial_metrics(2,(100,),90)
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a),48)
        self.assertEqual(len(paired_results(a)),18)

if __name__=="__main__":unittest.main()
