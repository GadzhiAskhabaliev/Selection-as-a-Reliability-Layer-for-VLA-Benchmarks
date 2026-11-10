import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load('pilot', 'experiments/offline/cs_pilot.py')
baseline = load('baseline_for_pilot_tests', 'kaggle/openvla_baseline/baseline.py')


class PilotTest(unittest.TestCase):
    def test_upstream_decode_mask_and_clipping(self):
        stats = {'q01': [-2, 0, 5], 'q99': [2, 1, 5], 'mask': [True, False, True]}
        result = pilot.decode_tokens([99, 97, 0], 100, [-0.8, 0, 0.8], stats)
        self.assertEqual(result['raw_action'], [-1.6, 0.8, 5])
        self.assertEqual(result['clipped_token_count'], 1)
        with self.assertRaises(ValueError):
            pilot.decode_tokens([99], 100, [-0.8, 0, 0.8], stats)

    def test_pool_seeds_depend_on_observation_not_selector(self):
        self.assertEqual(pilot.sample_seed(5, 6), pilot.sample_seed(5, 6))
        self.assertEqual(len({pilot.sample_seed(ep, step) for ep in range(20) for step in range(20)}), 400)

    def test_paired_summary_keeps_timestep_weights_and_clusters(self):
        errors = {'greedy': [1, 2, 3], 'cs8': [0.5, 1.5, 2.5]}
        episodes = [{'num_steps': 1, 'error_sums': {'greedy': 1, 'cs8': 0.5}},
                    {'num_steps': 2, 'error_sums': {'greedy': 5, 'cs8': 4}}]
        diagnostics = {'sum_unique_candidates': 18, 'cs_medoid_same_index': 2}
        result = pilot.paired_summary(baseline, errors, episodes, diagnostics, 10, 2, 'complete_pilot')
        self.assertEqual(result['methods']['cs8']['delta_amse_vs_paired_greedy'], -0.5)
        self.assertEqual(result['episode_bootstrap']['cs8']['delta_amse_95pct_episode_bootstrap'], [-0.5, -0.5])
        self.assertEqual(result['diagnostics']['mean_unique_candidates'], 6)
        self.assertFalse(result['full_dataset_baseline_reproduced'])


if __name__ == '__main__':
    unittest.main()
