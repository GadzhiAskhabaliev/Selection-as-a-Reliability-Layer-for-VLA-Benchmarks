"""Ground-truth action statistics of one MultiNet v0.1 Open-X test split (no model, CPU only).

Targets are produced exactly as in the archived baseline (kaggle/openvla_baseline/baseline.py):
official TFDS shards, the pinned MultiNet data loader and `drop_is_terminal_dim`. The script
reports the gripper target distribution, the AMSE of an all-zeros predictor, the AMSE contribution
of "no change" frames for any +/-1 gripper prediction, and the gripper command blocks per
instruction. It also writes the flattened steps reused by `multinet_gripper_state.py`.

In UR5, NYU Door and PushT the last two steps of each record carry ``is_last=True``, so MultiNet's
loader emits an extra one-step episode per record and the number of loader episodes exceeds the
number of official episodes; coverage is therefore checked on the number of steps.

Usage:
    python multinet_targets.py DATASET --scratch DIR --out DIR [--expected-steps N]
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'kaggle/openvla_baseline/baseline.py'


def load_baseline(scratch: Path, out: Path) -> ModuleType:
    """Import the archived baseline module with its scratch and output directories redirected.

    Args:
        scratch: Directory for downloads, the MultiNet source tree and flattened steps.
        out: Directory for outputs written by the baseline module.

    Returns:
        The loaded `kaggle/openvla_baseline/baseline.py` module.
    """
    spec = importlib.util.spec_from_file_location('multinet_baseline', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRATCH, module.OUT = scratch, out
    return module


def fetch_split(baseline: ModuleType, dataset: str, version: str, split: str, data_dir: Path) -> int:
    """Download the official TFDS files of one split, skipping files that already exist.

    Args:
        baseline: Loaded baseline module (provides the retrying `download`).
        dataset: Open-X dataset name.
        version: TFDS version, e.g. ``0.1.0``.
        split: Split name, e.g. ``test``.
        data_dir: Destination directory.

    Returns:
        Number of episodes in the split according to ``dataset_info.json``.
    """
    base = f'https://storage.googleapis.com/gresearch/robotics/{dataset}/{version}'
    for name in ('dataset_info.json', 'features.json'):
        if not (data_dir / name).exists():
            baseline.download(f'{base}/{name}', data_dir / name)
    info = json.loads((data_dir / 'dataset_info.json').read_text())
    lengths = next(s for s in info['splits'] if s['name'] == split)['shardLengths']
    for i in range(len(lengths)):
        name = f'{dataset}-{split}.tfrecord-{i:05d}-of-{len(lengths):05d}'
        if not (data_dir / name).exists() or (data_dir / name).stat().st_size == 0:
            print('Downloading', name, flush=True)
            baseline.download(f'{base}/{name}', data_dir / name)
    return sum(map(int, lengths))


def multinet_source(baseline: ModuleType, scratch: Path) -> Path:
    """Return the pinned MultiNet source tree, downloading it on first use."""
    root = scratch / 'MultiNet'
    return root if root.exists() else baseline.source_tree()


def blocks(sequence: Sequence[float]) -> str:
    """Summarize nonzero gripper commands of one segment as blocks: C (+1), O (-1), ? (other)."""
    return ''.join({1.0: 'C', -1.0: 'O'}.get(v, '?') for v, _ in itertools.groupby(sequence) if v != 0) or '-'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset')
    p.add_argument('--version', default='0.1.0')
    p.add_argument('--split', default='test')
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gripper-index', type=int, default=0, help='gripper position in the MultiNet target')
    p.add_argument('--expected-steps', type=int, help='published number of evaluated steps')
    args = p.parse_args()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(args.scratch, args.out)
    data_dir = args.scratch / args.dataset / args.version
    episodes_expected = fetch_split(baseline, args.dataset, args.version, args.split, data_dir)

    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], 'GPU')
    flat_dir = args.scratch / f'flat_targets_{args.dataset}'
    if not flat_dir.exists():
        builder = tfds.builder_from_directory(str(data_dir))
        episodes = builder.as_dataset(split=args.split, shuffle_files=False,
                                      read_config=tfds.ReadConfig(interleave_cycle_length=1))
        episodes.flat_map(lambda episode: episode['steps']).save(str(flat_dir))

    root = multinet_source(baseline, args.scratch)
    vla_root = root / 'src/eval/profiling/openvla'
    sys.path[:0] = [str(root), str(vla_root)]
    baseline.minimal_prismatic_packages(vla_root)
    from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
    from src.eval.profiling.openvla.experiments.robot.multinet_openvla_utils import drop_is_terminal_dim
    reference = json.loads((vla_root / 'results/openvla_openx_eval_results.json').read_text())[args.dataset]

    rows, gi = [], args.gripper_index
    with (args.out / 'targets.jsonl').open('w') as log:
        for episode, batch in enumerate(get_openx_dataloader([str(flat_dir)], batch_size=1)):
            for step in range(len(batch['continuous_observation'][0])):
                target = [float(v) for v in drop_is_terminal_dim(batch['action'][0][step], args.dataset)]
                row = {'episode': episode, 'step': step, 'instruction': batch['text_observation'][0][step],
                       'target': target}
                rows.append(row)
                log.write(json.dumps(row) + '\n')

    m, dim = len(rows), len(rows[0]['target'])
    gripper = collections.Counter(r['target'][gi] for r in rows)
    zeros_amse = sum(sum(v * v for v in r['target']) for r in rows) / m / dim
    no_change_floor = gripper.get(0.0, 0) / m / dim
    patterns = collections.Counter()
    for _, group in itertools.groupby(rows, key=lambda r: r['episode']):
        group = list(group)
        patterns[(group[0]['instruction'], blocks([r['target'][gi] for r in group]))] += 1
    summary = {
        'dataset': args.dataset, 'episodes': len({r['episode'] for r in rows}),
        'episodes_expected': episodes_expected, 'timesteps': m, 'timesteps_expected': args.expected_steps,
        'target_dim': dim, 'gripper_target_counts': {str(k): v for k, v in sorted(gripper.items())},
        'no_change_share': gripper.get(0.0, 0) / m,
        'zeros_predictor_amse': zeros_amse,
        'no_change_contribution_for_pm1_predictions': no_change_floor,
        'published_openvla_amse_NOT_OUR_RESULT': reference['avg_dataset_amse'],
        'no_change_contribution_share_of_published_amse': no_change_floor / reference['avg_dataset_amse'],
        'gripper_blocks_by_instruction': [{'instruction': k[0], 'blocks': k[1], 'episodes': v}
                                          for k, v in patterns.most_common()],
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'gripper_blocks_by_instruction'}, indent=2))
    if args.expected_steps and m != args.expected_steps:
        raise SystemExit(f'{m} evaluated steps differ from the published {args.expected_steps}')


if __name__ == '__main__':
    main()
