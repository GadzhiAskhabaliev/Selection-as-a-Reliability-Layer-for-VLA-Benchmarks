from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('boundaries_test', ROOT / 'experiments/offline/record_boundaries.py')
boundaries = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boundaries
spec.loader.exec_module(boundaries)


class BoundaryMetadataTest(unittest.TestCase):
    def test_double_last_flags_do_not_create_extra_episodes(self):
        # Exact boundary lengths/positions from the first two official NYU records.
        records = [{'num_steps': 37, 'is_first_indices': [0], 'is_last_indices': [35, 36]},
                   {'num_steps': 35, 'is_first_indices': [0], 'is_last_indices': [33, 34]}]
        protocol = {'split_episodes': 2, 'full_split_expected_steps': 72, 'source_episode_lengths': [37, 35]}
        report = boundaries.validate_records(records, protocol)
        self.assertEqual(report['num_source_episodes'], 2)
        self.assertEqual(report['num_source_steps'], 72)
        self.assertEqual(report['raw_is_last_count'], 4)
        self.assertEqual(report['effective_is_last_count'], 2)
        self.assertEqual(report['dropped_steps'], 0)
        with self.assertRaisesRegex(ValueError, 'record/step count'):
            boundaries.validate_records(records, {**protocol, 'full_split_expected_steps': 71})
        with self.assertRaisesRegex(ValueError, 'ordering/lengths'):
            boundaries.validate_records(records, {**protocol, 'source_episode_lengths': [35, 37]})

    def test_bad_first_or_missing_final_marker_is_rejected(self):
        protocol = {'split_episodes': 1, 'full_split_expected_steps': 3, 'source_episode_lengths': [3]}
        for first, last in [([0, 1], [2]), ([], [2]), ([0], [1]), ([0], [])]:
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                boundaries.validate_records([{'num_steps': 3, 'is_first_indices': first,
                                              'is_last_indices': last}], protocol)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow integration environment required')
class TensorFlowBoundaryTest(unittest.TestCase):
    def test_nested_record_repair_roundtrip_preserves_images_and_actions(self):
        import numpy as np
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        lengths = [37, 35]
        def make_episode(index):
            length = tf.gather(tf.constant(lengths, dtype=tf.int64), index)
            def make_step(step):
                return {'is_first': tf.equal(step, 0), 'is_last': step >= length - 2,
                        'is_terminal': step >= length - 2,
                        'observation': {'image': tf.fill([2, 2, 3], tf.cast(index * 100 + step, tf.uint8))},
                        'action': tf.cast(tf.stack([index, step]), tf.float32)}
            return {'steps': tf.data.Dataset.range(length).map(make_step)}
        episodes = tf.data.Dataset.range(2).map(make_episode)
        protocol = {'split_episodes': 2, 'full_split_expected_steps': 72, 'source_episode_lengths': lengths}
        saved = {}
        with redirect_stdout(io.StringIO()):
            corrected = boundaries.restore_boundaries(episodes, tf, protocol, lambda name, value: saved.update({name: value}))
        self.assertTrue(saved['data_boundary_audit.json']['mapped_pipeline_verified'])
        originals = list(episodes.flat_map(lambda ep: ep['steps']).as_numpy_iterator())
        with tempfile.TemporaryDirectory() as temp:
            corrected.flat_map(lambda ep: ep['steps']).save(temp)
            flat = list(tf.data.Dataset.load(temp).as_numpy_iterator())
        self.assertEqual(len(flat), len(originals))
        self.assertEqual([i for i, row in enumerate(flat) if row['is_last']], [36, 71])
        self.assertEqual([i for i, row in enumerate(flat) if row['source_is_last']], [35, 36, 70, 71])
        for old, new in zip(originals, flat):
            np.testing.assert_array_equal(old['observation']['image'], new['observation']['image'])
            np.testing.assert_array_equal(old['action'], new['action'])
            self.assertEqual(old['is_terminal'], new['is_terminal'])
        self.assertEqual([(int(row['source_episode_index']), int(row['source_step_index'])) for row in flat],
                         [(ep, step) for ep, length in enumerate(lengths) for step in range(length)])


if __name__ == '__main__':
    unittest.main()
