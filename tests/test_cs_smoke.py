import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load('smoke_test_module', 'experiments/offline/cs_smoke.py')
selectors = load('selectors_for_smoke', 'src/consensus_selection/__init__.py')
baseline = load('baseline_for_smoke', 'kaggle/openvla_baseline/baseline.py')
try:
    import numpy as np
except ImportError:
    np = None


class SmokeTest(unittest.TestCase):
    def test_sparse_frames_preserve_real_terminal_and_all_fields(self):
        batch = {'continuous_observation': [list(range(10))], 'action': [list(range(20, 30))],
                 'is_last': [[False] * 9 + [True]], 'unused': None}
        index_map = []
        result = list(smoke.sparse_batches([batch, batch], index_map))
        self.assertEqual(index_map, [[0, 1, 2, 9]] * 2)
        self.assertEqual(result[0]['action'], [[20, 21, 22, 29]])
        self.assertEqual(result[0]['is_last'], [[False, False, False, True]])
        self.assertIsNone(result[0]['unused'])
        self.assertEqual(len(batch['action'][0]), 10)

    @unittest.skipIf(np is None, 'NumPy reference requires the NumPy test environment')
    def test_independent_reference_and_corruption_detection(self):
        candidates = [{'raw_action': [v / 10] * 7, 'token_ids': [31999 - v] * 7,
                       'clipped_token_count': 0} for v in range(8)]
        normalized = [c['raw_action'] for c in candidates]
        selection = selectors.select([[a] for a in normalized])
        indices = {'cs8': selection['cs_index'], 'medoid8': selection['medoid_index'],
                   'random8': 4, 'sample1': 0}
        records = []
        for episode in range(2):
            for step in range(4):
                predictions = {'greedy': [0.2] * 4}
                predictions.update({name: candidates[index]['raw_action'][:4] for name, index in indices.items()})
                target = [step / 7] * 4
                records.append({'episode': episode, 'step': step, 'is_last': step == 3,
                                'candidates': candidates, 'normalized_candidates': normalized,
                                'selection': selection, 'selected_indices': indices,
                                'predictions': predictions, 'target': target,
                                'mse': {name: baseline.mse(a, target) for name, a in predictions.items()}})
        result = {'num_timesteps': 8, 'methods': {
            name: baseline.aggregate([r['mse'][name] for r in records]) for name in records[0]['mse']}}
        checks = smoke.validate_records(records, result, np, [[0, 1, 2, 9]] * 2)
        self.assertIn('PASS', checks['cs_and_medoid_reference'])
        bad = copy.deepcopy(records)
        bad[0]['selection']['cs_scores'][0] += 1
        with self.assertRaisesRegex(AssertionError, 'CS scores'):
            smoke.validate_records(bad, result, np, [[0, 1, 2, 9]] * 2)
        bad = copy.deepcopy(records)
        bad[0]['candidates'][0]['token_ids'][0] = 2
        with self.assertRaisesRegex(AssertionError, 'Non-action token'):
            smoke.validate_records(bad, result, np, [[0, 1, 2, 9]] * 2)
        edge = copy.deepcopy(records)
        edge[0]['candidates'][0]['token_ids'][0] = 31744
        edge[0]['candidates'][0]['clipped_token_count'] = 1
        self.assertIn('PASS', smoke.validate_records(edge, result, np, [[0, 1, 2, 9]] * 2)['action_tokens'])


if __name__ == '__main__':
    unittest.main()
