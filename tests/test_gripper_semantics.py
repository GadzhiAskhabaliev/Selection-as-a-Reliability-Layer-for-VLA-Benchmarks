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

    def test_score_separates_gripper_and_motion(self):
        result = gs.score([[1, 0.0, 0, 0], [-1, 0.0, 0, 0]], [[1, 0.2, 0, 0], [1, 0.0, 0, 0]])
        self.assertAlmostEqual(result['gripper_part'], 4 / 2 / 4)
        self.assertAlmostEqual(result['motion_part'], 0.04 / 2 / 4)
        self.assertAlmostEqual(result['gripper_state_error_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()
