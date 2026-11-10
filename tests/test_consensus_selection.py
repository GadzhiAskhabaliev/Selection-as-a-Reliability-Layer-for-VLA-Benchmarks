import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from consensus_selection import normalize_openvla, select


class SelectionTest(unittest.TestCase):
    def test_singleton_duplicates_and_two_candidate_tie(self):
        for pool in ([[[5.0]]], [[[1.0]]] * 8, [[[0.0]], [[100.0]]]):
            result = select(pool)
            self.assertEqual(result['cs_index'], 0)
            self.assertEqual(result['medoid_index'], 0)
        self.assertEqual(select([[[1.0]]] * 8)['bandwidth'], 1.000001)

    def test_bandwidth_excludes_zero_distances_and_self_pairs(self):
        result = select([[[0.0]], [[0.0]], [[3.0]]])
        self.assertEqual(result['bandwidth'], 3.000001)
        self.assertEqual(result['num_unique'], 2)
        self.assertEqual(result['mean_pairwise_distance'], 2.0)
        h = result['bandwidth']
        self.assertAlmostEqual(result['cs_scores'][0], 2 + math.exp(-9 / (2 * h * h)))

    def test_dense_cluster_vs_distant_outlier(self):
        result = select([[[x]] for x in (0, 1, 2, 3, 100)])
        self.assertEqual(result['cs_index'], 1)  # two central points tie; earliest wins
        self.assertEqual(result['medoid_index'], 2)
        self.assertAlmostEqual(result['cs_scores'][1], 3.5723820338998538)

    def test_whole_prefix_and_translation_invariance(self):
        prefixes = [[[0, 0], [1, 2]], [[2, 0], [1, 2]], [[0, 0], [1, 2]]]
        flat = [[[v for step in prefix for v in step]] for prefix in prefixes]
        shifted = [[[v + 10 for v in step] for step in prefix] for prefix in prefixes]
        self.assertEqual(select(prefixes), select(flat))
        self.assertEqual(select(prefixes), select(shifted))

    def test_invalid_candidates_and_bandwidth(self):
        for pool in ([], [[]], [[[]]], [[[math.nan]]], [[[math.inf]]],
                     [[[0]], [[0, 1]]], [[[0], [1]], [[0]]]):
            with self.assertRaises(ValueError):
                select(pool)
        for eps in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                select([[[0]]], epsilon=eps)

    def test_frozen_inverse_normalizer_mask_and_constant_coordinates(self):
        stats = {'q01': [-0.2, 0, 0], 'q99': [0.2, 0, 1], 'mask': [True, True, False]}
        raw = [[-0.2, 0, -0.7], [0, 0, 0], [0.2, 0, 0.8]]
        self.assertEqual(normalize_openvla(raw, stats), [[-1, 0, -0.7], [0, 0, 0], [1, 0, 0.8]])
        self.assertEqual(raw[0], [-0.2, 0, -0.7])
        # Candidate-dependent scaling would map this to [-1,1]; frozen scaling must not.
        self.assertAlmostEqual(normalize_openvla([[0.1, 0, 0]], stats)[0][0], 0.5)
        with self.assertRaises(ValueError):
            normalize_openvla([[0, 1, 0]], stats)


if __name__ == '__main__':
    unittest.main()
