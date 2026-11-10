from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/online'))

import summarize  # noqa: E402


def write_run(root, method, outcomes, select_s=None):
    d = root / method
    d.mkdir()
    with (d / 'episodes.jsonl').open('w') as f:
        for (task_id, trial), success in outcomes.items():
            f.write(json.dumps({'task_id': task_id, 'trial': trial, 'success': success, 'error': None}) + '\n')
    step = {'policy_s': 0.2, **({'select_s': select_s} if select_s is not None else {})}
    (d / 'steps.jsonl').write_text(json.dumps(step) + '\n')


class SummaryTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(summarize.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(summarize.mcnemar_exact(6, 0), 2 / 64)
        self.assertAlmostEqual(summarize.mcnemar_exact(3, 3), 1.0)

    def test_contrast_uses_only_shared_initial_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, 'greedy', {(0, 0): False, (0, 1): True, (1, 0): False})
            write_run(root, 'cs', {(0, 0): True, (0, 1): True}, select_s=0.0003)
            report = summarize.summarize(root, 'greedy')
        contrast = report['contrasts']['cs-greedy']
        self.assertEqual(contrast['paired_episodes'], 2)
        self.assertEqual((contrast['wins'], contrast['losses']), (1, 0))
        self.assertAlmostEqual(contrast['delta'], 0.5)
        self.assertAlmostEqual(report['methods']['cs']['select_ms_mean'], 0.3)
        self.assertAlmostEqual(report['methods']['greedy']['success_rate'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
