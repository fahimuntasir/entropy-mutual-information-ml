"""Smoke tests for research_v2 experiment.
Run from repository root:
python -m unittest discover -s research_v2/tests -v
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conditional_mi_logloss import (
    binary_entropy_bits, empirical_mi_bits, generate_binary_data,
    joint_binary_code, run_experiment, select_features,
    theoretical_conditional_logloss_bits,
)

class ExperimentTests(unittest.TestCase):
    def test_data_duplicates_and_shape(self):
        x, y = generate_binary_data(200, 0.2, 123)
        self.assertEqual(x.shape, (200, 20))
        self.assertEqual(y.shape, (200,))
        np.testing.assert_array_equal(x[:, 0], x[:, 2])

    def test_mi_exact_binary_copy(self):
        y = np.array([0, 0, 1, 1] * 20)
        self.assertAlmostEqual(empirical_mi_bits(y, y), 1.0, places=10)
        self.assertAlmostEqual(empirical_mi_bits(np.zeros_like(y), y), 0.0, places=10)

    def test_cmi_avoids_duplicate(self):
        x, y = generate_binary_data(4000, 0.2, 12)
        self.assertEqual(select_features(x, y, "MI"), [0, 2])
        self.assertEqual(select_features(x, y, "Conditional MI"), [0, 1])
        np.testing.assert_array_equal(joint_binary_code(x, [0, 2]), x[:, 0] * 3)

    def test_theoretical_probabilistic_gain(self):
        one = binary_entropy_bits(0.2)
        both = theoretical_conditional_logloss_bits(0.2, 0.3)
        self.assertLess(both, one)
        self.assertAlmostEqual(one, 0.7219280948873623, places=12)

    def test_end_to_end(self):
        runs, summary, paired, theory = run_experiment(
            trials=2, test_size=80, train_sizes=(100,), noise_levels=(0.2,)
        )
        self.assertEqual(len(runs), 8)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(theory), 1)
        self.assertTrue(runs["accuracy"].between(0, 1).all())
        self.assertTrue(runs["logloss_bits"].between(0, 4).all())

if __name__ == "__main__":
    unittest.main()
