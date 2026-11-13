"""Re-score MultiNet v0.1 OpenVLA predictions with aligned gripper action semantics.

Three MultiNet v0.1 datasets (Jaco Play, Berkeley UR5, NYU Door Opening) store gripper targets as
relative commands: the change of gripper closedness, with 0 meaning "no change". OpenVLA's data
pipeline converts exactly these datasets to absolute gripper state with
`rel2abs_gripper_actions` (openvla/openvla, prismatic/vla/datasets/rlds/utils/data_utils.py;
vendored in MultiNet at the same path), while MultiNet compares OpenVLA's absolute output
(+1 closed / -1 open after `convert_action`) with the raw relative targets.

Two alignments are computed; all non-gripper dimensions are left unchanged in both:

* Fix 1 (targets -> states): each episode's gripper targets are converted with the same rel2abs
  rule (threshold 0.1) and MultiNet's sign convention.
* Fix 2 (model -> commands): MultiNet's targets are kept and the predicted state is converted into
  a command against the observed gripper state, ``a = clip(s_hat - s_obs, -1, 1)``. This needs a
  proprioceptive gripper state: UR5 ``robot_state[13]`` (``gripper_is_closed``) and the Jaco finger
  joint ``joint_pos[6]`` (binarized at the midpoint of its observed range). NYU Door Opening
  observations contain no robot state.

The gripper is the first coordinate of every MultiNet target and prediction used here.

Episode boundaries: in Berkeley UR5 and NYU Door Opening, the last two steps of every record carry
``is_last=True``, and MultiNet's loader (flatten, then split on ``is_last``) turns the final step of
each record into a separate one-step episode. Per-step metrics are unaffected, but the episode-wise
rel2abs conversion of fix 1 must run on the source records, so one-step segments are merged back
into the preceding segment and the result is checked against the official episode count.

Input files are MultiNet baseline ``predictions.jsonl`` (``episode``, ``step``, ``prediction``,
``target``) or the archived offline CS pilot ``candidates.jsonl`` (per-method ``predictions``).
CPU only, standard library only.

Usage:
    python gripper_semantics.py [FILE ...]
"""
from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = [
    ROOT / 'artifacts/baseline_audit/current/output/predictions.jsonl',
    ROOT / 'artifacts/multinet_audit/berkeley_autolab_ur5/predictions.jsonl',
    ROOT / 'artifacts/multinet_audit/nyu_door_opening_surprising_effectiveness/predictions.jsonl',
    ROOT / 'artifacts/cs_pilot/v2/output/candidates.jsonl',
]
THRESHOLD = 0.1
STATE_DIR = ROOT / 'artifacts/multinet_audit/gripper_state'
OBSERVED_STATE = {'jaco_play': 'joint_pos[6]', 'berkeley_autolab_ur5': 'robot_state[13]'}
# Official test-split episode counts of the datasets whose loader episodes are split in two.
SPLIT_EPISODE_DATASETS = {'berkeley_autolab_ur5': 104, 'nyu_door_opening_surprising_effectiveness': 49}

Action = Sequence[float]


def rel2abs(commands: Sequence[float]) -> list[int]:
    """Port of OpenVLA's ``rel2abs_gripper_actions`` in MultiNet's sign convention.

    Args:
        commands: Relative gripper commands of one episode (+ closes, - opens, ~0 no change).

    Returns:
        Absolute gripper state per step: +1 closed, -1 open. The state before the first command is
        the opposite of that command; an episode without commands is open throughout.
    """
    thresholded = [1 if c < -THRESHOLD else (-1 if c > THRESHOLD else 0) for c in commands]
    first = next((t for t in thresholded if t != 0), 0)
    state = -first if first != 0 else 1
    absolute = []
    for t in thresholded:
        if t != 0:
            state = t
        absolute.append(-state)
    return absolute


def source_episodes(rows: Sequence[dict[str, Any]], expected_episodes: int) -> list[dict[str, Any]]:
    """Merge MultiNet's spurious one-step loader episodes back into their source records.

    Args:
        rows: Prediction records sorted by loader episode and step.
        expected_episodes: Official number of episodes in the split.

    Returns:
        Copies of ``rows`` with ``episode`` and ``step`` renumbered by source record.

    Raises:
        ValueError: If the merged episode count differs from ``expected_episodes``.
    """
    segments = [list(group) for _, group in groupby(rows, key=lambda r: r['episode'])]
    episodes: list[list[dict[str, Any]]] = []
    for segment in segments:
        if len(segment) == 1 and episodes:
            episodes[-1].extend(segment)
        else:
            episodes.append(list(segment))
    if len(episodes) != expected_episodes:
        raise ValueError(f'{len(episodes)} merged episodes, expected {expected_episodes}')
    return [{**r, 'episode': e, 'step': s} for e, episode in enumerate(episodes) for s, r in enumerate(episode)]


def corrected_targets(rows: Sequence[dict[str, Any]]) -> list[list[float]]:
    """Replace each row's gripper target by the episode-wise absolute state (fix 1).

    Args:
        rows: Prediction records sorted by episode and step, each with a ``target`` list.

    Returns:
        Targets with the gripper coordinate converted by :func:`rel2abs`, other coordinates as is.
    """
    out = []
    for _, episode in groupby(rows, key=lambda r: r['episode']):
        episode = sorted(episode, key=lambda r: r['step'])
        states = rel2abs([r['target'][0] for r in episode])
        out.extend([s, *r['target'][1:]] for s, r in zip(states, episode))
    return out


def score(predictions: Sequence[Action], targets: Sequence[Action]) -> dict[str, float]:
    """MultiNet AMSE split into the gripper and the remaining coordinates.

    Args:
        predictions: Per-step predictions, gripper first.
        targets: Per-step targets of the same dimension, gripper first.

    Returns:
        ``amse``, its ``gripper_part`` and ``other_dims_part``, and the share of steps whose gripper
        prediction differs from the target (``gripper_mismatch_rate``).
    """
    m, dim = len(targets), len(targets[0])
    gripper = sum((p[0] - t[0]) ** 2 for p, t in zip(predictions, targets))
    motion = sum(sum((a - b) ** 2 for a, b in zip(p[1:], t[1:])) for p, t in zip(predictions, targets))
    mismatch = sum(p[0] != t[0] for p, t in zip(predictions, targets))
    return {'amse': (gripper + motion) / m / dim, 'gripper_part': gripper / m / dim,
            'other_dims_part': motion / m / dim, 'gripper_mismatch_rate': mismatch / m}


def constants(targets: Sequence[Action]) -> dict[str, float]:
    """AMSE of constant predictors.

    Args:
        targets: Per-step targets.

    Returns:
        AMSE of the all-zeros, always-closed and always-open predictors, and of the per-dimension
        mean of the targets (``per_dimension_mean``), which is the MSE-optimal constant.
    """
    m, dim = len(targets), len(targets[0])
    mean = [sum(t[j] for t in targets) / m for j in range(dim)]
    result = {name: score([[g] + [0.0] * (dim - 1)] * m, targets)['amse']
              for name, g in (('zeros', 0.0), ('always_closed', 1.0), ('always_open', -1.0))}
    result['per_dimension_mean'] = score([mean] * m, targets)['amse']
    return result


def per_dimension_mse(predictions: Sequence[Action], targets: Sequence[Action]) -> list[float]:
    """Mean squared error of each action coordinate over all steps."""
    dim = len(targets[0])
    return [sum((p[j] - t[j]) ** 2 for p, t in zip(predictions, targets)) / len(targets) for j in range(dim)]


def observed_states(values: Sequence[float], threshold: float) -> list[int]:
    """Binarize an observed gripper signal into MultiNet's sign (+1 closed, -1 open)."""
    return [1 if v > threshold else -1 for v in values]


def state_to_command(predictions: Sequence[Action], states: Sequence[int]) -> list[list[float]]:
    """Convert predicted gripper states into relative commands against observed states (fix 2).

    Args:
        predictions: Per-step predictions with the gripper state (+1 closed / -1 open) first.
        states: Observed gripper state per step in the same sign convention.

    Returns:
        Predictions whose gripper coordinate is ``clip(predicted - observed, -1, 1)``.
    """
    return [[max(-1, min(1, p[0] - s))] + list(p[1:]) for p, s in zip(predictions, states)]


def symmetric_alignment(dataset: str, rows: Sequence[dict[str, Any]],
                        predictions: Sequence[Action]) -> dict[str, Any] | None:
    """Score fix 2 for a dataset with an archived, step-aligned observed gripper state.

    Args:
        dataset: MultiNet dataset name.
        rows: Prediction records sorted by episode and step.
        predictions: Model predictions aligned with ``rows``.

    Returns:
        AMSE decomposition of fix 2, the state field and threshold used (and, for Jaco, the
        threshold sensitivity), or ``None`` if no observed gripper state is available.

    Raises:
        ValueError: If the archived state export is not step-aligned with the predictions.
    """
    path = STATE_DIR / f'{dataset}.json'
    if dataset not in OBSERVED_STATE or not path.exists():
        return None
    export = json.loads(path.read_text())
    if export['gripper_command'] != [r['target'][0] for r in rows]:
        raise ValueError(f'{path} is not step-aligned with the predictions')
    values = export['state'][OBSERVED_STATE[dataset]]
    midpoint = (min(values) + max(values)) / 2
    targets = [r['target'] for r in rows]
    result = {'observed_state': OBSERVED_STATE[dataset], 'threshold': midpoint,
              **score(state_to_command(predictions, observed_states(values, midpoint)), targets)}
    if dataset == 'jaco_play':
        result['threshold_sensitivity'] = {
            str(t): score(state_to_command(predictions, observed_states(values, t)), targets)['amse']
            for t in (0.5, 0.75, 1.0)}
    return result


def dataset_of(path: Path | str) -> str:
    """Infer the MultiNet dataset name from an artifact path (Jaco Play by default)."""
    text = str(path)
    return next((d for d in ('berkeley_autolab_ur5', 'nyu_door_opening_surprising_effectiveness')
                 if d in text), 'jaco_play')


def report(path: Path | str) -> dict[str, Any]:
    """Full gripper-semantics report for one predictions file.

    Args:
        path: MultiNet baseline ``predictions.jsonl`` or offline CS pilot ``candidates.jsonl``.

    Returns:
        Original and aligned AMSE per method, constant baselines, per-dimension errors and, where an
        observed gripper state exists, fix 2.
    """
    rows = sorted((json.loads(line) for line in Path(path).read_text().splitlines()),
                  key=lambda r: (r['episode'], r['step']))
    dataset = dataset_of(path)
    if dataset in SPLIT_EPISODE_DATASETS:
        rows = source_episodes(rows, SPLIT_EPISODE_DATASETS[dataset])
    original = [r['target'] for r in rows]
    corrected = corrected_targets(rows)
    no_change = sum(abs(t[0]) < THRESHOLD for t in original) / len(rows)
    if 'predictions' in rows[0]:
        methods = {m: [r['predictions'][m] for r in rows] for m in rows[0]['predictions']}
    else:
        methods = {'openvla_greedy': [r['prediction'] for r in rows]}
    predicted_gripper_values = sorted({p[0] for preds in methods.values() for p in preds})
    symmetric = (symmetric_alignment(dataset_of(path), rows, methods['openvla_greedy'])
                 if 'openvla_greedy' in methods else None)
    return {
        'file': str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
        'dataset': dataset,
        'episodes': len({r['episode'] for r in rows}),
        'state_to_command_alignment': symmetric,
        'timesteps': len(rows), 'target_dim': len(original[0]),
        'no_change_target_share': no_change,
        'predicted_gripper_values': predicted_gripper_values[:10],
        'multinet_targets': {m: score(p, original) for m, p in methods.items()},
        'openvla_semantics_targets': {m: score(p, corrected) for m, p in methods.items()},
        'constant_predictors': {'multinet_targets': constants(original),
                                'openvla_semantics_targets': constants(corrected)},
        'per_dimension_mse_multinet_targets': {
            **{m: per_dimension_mse(p, original) for m, p in methods.items()},
            'zeros': per_dimension_mse([[0.0] * len(original[0])] * len(rows), original)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('files', nargs='*', type=Path, help='prediction files (default: archived files)')
    args = p.parse_args()
    files = args.files or [f for f in DEFAULT_FILES if f.exists()]
    print(json.dumps([report(f) for f in files], indent=2))


if __name__ == '__main__':
    main()
