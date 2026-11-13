import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gripper_semantics', ROOT / 'experiments/offline/gripper_semantics.py')
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)


class GripperSemanticsTest(unittest.TestCase):
    def test_rel2abs_matches_openvla_rule(self):
        # +1 = close command, -1 = open command, 0 = no change; output +1 closed / -1 open.
        self.assertEqual(gs.rel2abs([0, 0, 1, 0, 0, -1, 0]), [-1, -1, 1, 1, 1, -1, -1])
        # State before the first command is the opposite of that command.
        self.assertEqual(gs.rel2abs([0, -1, 0]), [1, -1, -1])
        # No command at all: open for the whole episode.
        self.assertEqual(gs.rel2abs([0, 0, 0]), [-1, -1, -1])

    def test_targets_are_converted_per_episode(self):
        rows = [{'episode': 0, 'step': 0, 'target': [1.0, 0.1, 0, 0]},
                {'episode': 0, 'step': 1, 'target': [0.0, 0.2, 0, 0]},
                {'episode': 1, 'step': 0, 'target': [0.0, 0.3, 0, 0]}]
        self.assertEqual(gs.corrected_targets(rows), [[1, 0.1, 0, 0], [1, 0.2, 0, 0], [-1, 0.3, 0, 0]])

    def test_continuous_relative_targets_use_the_openvla_threshold(self):
        # NYU Door style: noise around 0 is "no change"; a spike above 0.1 is a closing command.
        self.assertEqual(gs.rel2abs([0.004, -0.01, 0.19, 0.77, 0.01, -0.002]), [-1, -1, 1, 1, 1, 1])

    def test_score_separates_gripper_and_other_dimensions(self):
        result = gs.score([[1, 0.0, 0, 0], [-1, 0.0, 0, 0]], [[1, 0.2, 0, 0], [1, 0.0, 0, 0]])
        self.assertAlmostEqual(result['gripper_part'], 4 / 2 / 4)
        self.assertAlmostEqual(result['other_dims_part'], 0.04 / 2 / 4)
        self.assertAlmostEqual(result['gripper_mismatch_rate'], 0.5)

    def test_score_uses_the_target_dimension(self):
        result = gs.score([[1] + [0.0] * 6], [[-1] + [0.0] * 6])
        self.assertAlmostEqual(result['amse'], 4 / 7)

    def test_state_to_command_against_observed_state(self):
        # predicted closed while observed open -> close (+1); same state -> no change; closed -> open (-1)
        predictions = [[1, 0.5], [1, 0.5], [-1, 0.5]]
        states = gs.observed_states([0.0, 1.0, 1.0], 0.5)
        self.assertEqual(gs.state_to_command(predictions, states), [[1, 0.5], [0, 0.5], [-1, 0.5]])

    def test_constant_predictors(self):
        result = gs.constants([[1, 0.0], [0, 0.0]])
        self.assertAlmostEqual(result['zeros'], 1 / 2 / 2)
        # Per-dimension mean is the MSE-optimal constant: (0.5^2 + 0.5^2) / 2 points / 2 dims.
        self.assertAlmostEqual(result['per_dimension_mean'], 0.25 / 2)
        self.assertLessEqual(result['per_dimension_mean'], min(result.values()))


if __name__ == '__main__':
    unittest.main()
