from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/online')]

import robomonkey_proposals as rmp  # noqa: E402
import selection_rules as rules  # noqa: E402

STATS = {'q01': [-0.03, -0.04, -0.03, -0.08, -0.09, -0.2, 0.0],
         'q99': [0.03, 0.04, 0.04, 0.08, 0.08, 0.2, 1.0],
         'mask': [True] * 6 + [False]}


def pool(actions):
    converter = rmp.TokenActionConverter(STATS)
    actions = np.asarray(actions, dtype=float)
    return np.stack([converter.action_to_token(a) for a in actions]), actions, converter


def run(method, actions, verifier=None, seed=0):
    ids, acts, converter = pool(actions)
    return rules.choose(method, ids, acts, stats=STATS, converter=converter, num_aug=16,
                        rng=np.random.default_rng(seed), verifier=verifier)


class OnlineSelectionTest(unittest.TestCase):
    def test_token_roundtrip_matches_upstream_converter(self):
        converter = rmp.TokenActionConverter(STATS)
        action = np.array([0.01, -0.02, 0.0, 0.05, -0.05, 0.1, 1.0])
        ids = converter.action_to_token(action)
        self.assertTrue(np.all((ids >= 31744) & (ids <= 32000)))
        self.assertTrue(np.allclose(converter.token_to_action(ids)[:6], action[:6], atol=4e-4))

    def test_invalid_tokens_are_dropped_before_selection(self):
        ids, acts, converter = pool([[0.01] * 6 + [1.0]] * 3)
        ids[0, 2] = 100
        _, info = rules.choose('cs', ids, acts, stats=STATS, converter=converter,
                               num_aug=8, rng=np.random.default_rng(0))
        self.assertEqual(info['n_valid'], 2)
        self.assertEqual(info['fallback'], 'single_unique')

    def test_cs_returns_a_sampled_action_at_the_mode(self):
        mode = [0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0]
        action, info = run('cs', [mode, mode, mode, [-0.02, 0.03, 0.03, 0.06, 0.06, 0.15, 0.0]])
        self.assertEqual(info['index'], 0)
        self.assertEqual(action.tolist(), mode)

    def test_majority_gripper_overrides_selected_candidate(self):
        actions = [[0.01, 0, 0, 0, 0, 0, 0.0], [0.01, 0, 0, 0, 0, 0, 1.0],
                   [0.012, 0, 0, 0, 0, 0, 1.0], [-0.02, 0, 0, 0, 0, 0, 1.0]]
        cs_action, _ = run('cs_majgrip', actions)
        mean_action, _ = run('mean', actions)
        self.assertEqual(cs_action[-1], 1.0)
        self.assertEqual(mean_action[-1], 1.0)
        self.assertAlmostEqual(mean_action[0], np.mean([a[0] for a in actions]))

    def test_augmented_methods_share_the_proposal_distribution(self):
        actions = [[0.01, 0.0, 0, 0, 0, 0, 1.0], [0.02, 0.01, 0, 0, 0, 0, 1.0], [0.0, -0.01, 0, 0, 0, 0, 0.0]]
        calls = []

        def verifier(ids):
            calls.append(len(ids))
            return list(range(len(ids)))

        rm_action, rm_info = run('robomonkey', actions, verifier=verifier)
        rand_action, _ = run('random_aug', actions)
        cs_action, _ = run('cs_aug', actions)
        self.assertEqual(calls, [rm_info['n_proposals']])
        self.assertEqual(rm_info['index'], rm_info['n_proposals'] - 1)
        for action in (rm_action, rand_action, cs_action):
            self.assertEqual(action[-1], 1.0)
            self.assertTrue(np.all(action[:6] >= rmp.BRIDGE_MIN[:6]) and np.all(action[:6] <= rmp.BRIDGE_MAX[:6]))

    def test_single_methods_are_rejected_by_pool_dispatch(self):
        with self.assertRaises(ValueError):
            run('greedy', [[0.0] * 7])


if __name__ == '__main__':
    unittest.main()
