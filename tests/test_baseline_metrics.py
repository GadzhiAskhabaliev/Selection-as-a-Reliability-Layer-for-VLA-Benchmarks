import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('baseline', Path(__file__).resolve().parents[1] / 'kaggle/openvla_baseline/baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class MetricsTest(unittest.TestCase):
    def test_squared_not_absolute_error(self):
        self.assertEqual(baseline.mse([1, 3], [0, 0]), 5)

    def test_paper_minmax_normalization(self):
        result = baseline.aggregate([1, 2, 5])
        self.assertAlmostEqual(result['amse'], 8 / 3)
        self.assertAlmostEqual(result['namse'], 5 / 12)
        self.assertEqual(result['num_timesteps'], 3)

    def test_no_broadcasting_or_nonfinite_actions(self):
        for a, b in [([1, 2], [1]), ([math.nan], [0]), ([], [])]:
            with self.assertRaises(ValueError):
                baseline.mse(a, b)

    def test_constant_error_is_explicitly_undefined(self):
        result = baseline.aggregate([0, 0])
        self.assertEqual(result['amse'], 0)
        self.assertIsNone(result['namse'])
        self.assertFalse(result['namse_defined'])

    def test_minmax_scale_invariance(self):
        self.assertAlmostEqual(baseline.aggregate([1, 2, 5])['namse'],
                               baseline.aggregate([10, 20, 50])['namse'])


if __name__ == '__main__':
    unittest.main()
