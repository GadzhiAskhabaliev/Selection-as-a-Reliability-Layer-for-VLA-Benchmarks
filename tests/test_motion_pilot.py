import copy
from contextlib import nullcontext, redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion = load('motion_test', 'experiments/offline/motion_pilot.py')
selectors = load('motion_selectors_test', 'src/consensus_selection/__init__.py')
baseline = load('motion_baseline_test', 'kaggle/openvla_baseline/baseline.py')
pilot = load('motion_old_pilot_test', 'experiments/offline/cs_pilot.py')
STATS = {'q01': [-1] * 7, 'q99': [1] * 7, 'mask': [True] * 6 + [False]}


class MotionTest(unittest.TestCase):
    def test_decoder_uses_requested_dataset_and_accepts_valid_edge_token(self):
        class Tensor(np.ndarray):
            def cpu(self):
                return self
        tensor = lambda value, **kw: np.array(value).view(Tensor)
        Tensor.device = 'cpu'
        requested = []
        tokens = [31744] * 7
        def dim(dataset):
            requested.append(dataset)
            return 7
        model = types.SimpleNamespace(get_action_dim=dim, get_action_stats=lambda ds: STATS,
            vocab_size=32000, bins=np.linspace(-1, 1, 256),
            bin_centers=(np.linspace(-1, 1, 256)[:-1] + np.linspace(-1, 1, 256)[1:]) / 2,
            generate=lambda ids, **kwargs: tensor(np.concatenate([ids, [tokens]], axis=1)))
        torch = types.SimpleNamespace(all=np.all, tensor=tensor, inference_mode=nullcontext,
                                      cat=lambda values, dim: tensor(np.concatenate(values, axis=dim)))
        runtime = types.SimpleNamespace(torch=torch, model=model)
        protocol = {'dataset': 'nyu_door_opening_surprising_effectiveness',
                    'temperature': 1.0, 'top_k': 10, 'top_p': 1.0}
        result = motion.generate_action(runtime, {'input_ids': tensor([[1, 42]])}, True,
                                        protocol, pilot.decode_tokens)
        self.assertEqual(requested, [protocol['dataset']])
        self.assertEqual(result['clipped_token_count'], 7)
        tokens[0] = 0
        with self.assertRaisesRegex(ValueError, 'Non-action token'):
            motion.generate_action(runtime, {'input_ids': tensor([[1, 29871]])}, True,
                                   protocol, pilot.decode_tokens)

    def test_gripper_cannot_change_motion_selection_and_inputs_are_preserved(self):
        raw = [[x, 0, 0, 0, 0, 0, g] for x, g in [(0, 0), (.1, 1), (.2, -1), (3, .9)]]
        greedy = [.5, .4, .3, .2, .1, 0, .75]
        before = copy.deepcopy(raw)
        first = motion.select_actions(selectors, raw, greedy, STATS, 2)
        changed = [a[:6] + [100 * (i - 1)] for i, a in enumerate(raw)]
        second = motion.select_actions(selectors, changed, greedy, STATS, 2)
        self.assertEqual(first['motion_selection'], second['motion_selection'])
        self.assertEqual(raw, before)
        for name in motion.HYBRIDS:
            action = first['raw_actions'][name]
            self.assertEqual(action[6], greedy[6])
            self.assertEqual(action[:6], raw[first['selected_indices'][name]][:6])

    def test_motion_collapse_ignores_gripper_diversity(self):
        raw = [[0] * 6 + [g] for g in (0, 1, .5)]
        result = motion.select_actions(selectors, raw, [0] * 7, STATS, 1)
        self.assertEqual(result['motion_selection']['num_unique'], 1)
        self.assertEqual(result['motion_selection']['cs_index'], 0)
        self.assertEqual(result['full_selection']['num_unique'], 3)

    def test_two_dataset_complete_loop_with_skipped_episodes_and_full_metrics(self):
        for dataset, dim in [('jaco_play', 4), ('nyu_door_opening_surprising_effectiveness', 7)]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                def save(name, value):
                    (output / name).write_text(json.dumps(value))
                fake_base = types.SimpleNamespace(OUT=output, save_json=save,
                                                  mse=baseline.mse, aggregate=baseline.aggregate)
                protocol = dict(json.loads((ROOT / 'configs/motion/jaco.json').read_text()),
                                dataset=dataset, episode_start=1, episode_count=2,
                                expected_eval_steps=4, eval_action_dim=dim)
                greedy = np.array([.2, .1, -.1, 0, 0, 0, .8])
                def convert(a, ds):
                    self.assertEqual(ds, dataset)
                    return np.array([a[6], *a[:3]]) if dim == 4 else np.array([a[6], *a[:6]])
                batches = []
                for ep in range(3):
                    batches.append({'continuous_observation': [[None, None]],
                        'image_observation': [[np.zeros((2, 2, 3), dtype=np.uint8)] * 2],
                        'text_observation': [[f'episode {ep}', f'episode {ep}']],
                        'action': [[np.zeros(dim), np.zeros(dim)]], 'is_last': [[False, True]]})
                cuda = types.SimpleNamespace(device_count=lambda: 0, synchronize=lambda device: None)
                model = types.SimpleNamespace(get_action_stats=lambda ds: STATS,
                                              generation_config=types.SimpleNamespace(to_dict=lambda: {}))
                runtime = types.SimpleNamespace(np=np, torch=types.SimpleNamespace(cuda=cuda, manual_seed=lambda seed: None),
                    model=model, manifest={}, predict=lambda image, text: greedy.copy(),
                    prepare_inputs=lambda image, text: {}, convert_action=convert,
                    drop_is_terminal_dim=lambda action, ds: action, flat_dir=output,
                    get_openx_dataloader=lambda *a, **kw: iter(batches))
                fake_base.prepare_runtime = lambda kind: runtime
                counter = [0]
                def generated(runtime, inputs, sample, cfg, decoder):
                    counter[0] += int(sample)
                    raw = greedy.copy()
                    if sample:
                        raw[0] += .01 * (counter[0] % 8)
                        raw[6] = float(counter[0] % 2)
                    return {'raw_action': raw.tolist(), 'token_ids': [31900] * 7, 'clipped_token_count': 0}
                with patch.object(motion, 'generate_action', generated), redirect_stdout(io.StringIO()):
                    motion.run(fake_base, selectors, pilot, protocol)
                result = json.loads((output / 'result.json').read_text())
                rows = [json.loads(l) for l in (output / 'candidates.jsonl').read_text().splitlines()]
                self.assertEqual(result['status'], 'complete_motion_pilot')
                self.assertEqual([(r['episode'], r['step']) for r in rows], [(1, 0), (1, 1), (2, 0), (2, 1)])
                self.assertEqual(result['diagnostics']['candidate_actions'], 32)
                self.assertEqual(result['diagnostics']['startup_checks_passed'], 3)
                for name in motion.HYBRIDS:
                    self.assertEqual(result['methods'][name]['gripper_mse'], result['methods']['greedy']['gripper_mse'])
                for name in motion.METHODS:
                    mean = np.mean([np.mean((np.array(r['predictions'][name])-r['target'])**2) for r in rows])
                    self.assertAlmostEqual(result['methods'][name]['amse'], mean)

    def test_seed_uses_dataset_and_original_episode_id(self):
        jaco = {'seed': 7, 'dataset': 'jaco_play'}
        other = {'seed': 7, 'dataset': 'nyu_door_opening_surprising_effectiveness'}
        self.assertEqual(motion.pool_seed(jaco, 20, 0), pilot.sample_seed(20, 0))
        self.assertNotEqual(motion.pool_seed(jaco, 20, 0), motion.pool_seed(other, 20, 0))


if __name__ == '__main__':
    unittest.main()
