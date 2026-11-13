"""Re-score MultiNet v0.1 OpenVLA predictions with gripper targets in OpenVLA's training semantics.

Applies to the MultiNet datasets whose gripper targets are relative commands (change of gripper
closedness; 0 = no change): Jaco Play, Berkeley UR5 and NYU Door Opening. OpenVLA's data pipeline
converts exactly these datasets with `rel2abs_gripper_actions` (openvla/openvla,
prismatic/vla/datasets/rlds/utils/data_utils.py; vendored in MultiNet at the same path), while
MultiNet compares OpenVLA's absolute gripper output (+1 closed / -1 open after `convert_action`)
with the raw relative targets. Here each episode's targets are converted with the same rel2abs
rule (threshold 0.1) and sign convention, so prediction and target describe the same quantity.
All other action dimensions are unchanged. The gripper is the first dimension of the MultiNet
target and prediction for all three datasets.

The symmetric alignment keeps MultiNet's targets and converts the model's predicted state into a
relative command against the observed gripper state, a = clip(s_hat - s_obs, -1, 1). It needs a
proprioceptive gripper state, available for UR5 (robot_state[13], `gripper_is_closed`, documented
on the dataset page) and Jaco (finger joint joint_pos[6], binarized at the midpoint of its observed
range) but not for NYU Door Opening, whose observations contain no robot state.

Input: predictions.jsonl from the MultiNet baseline (episode, step, prediction, target) or the
archived CS pilot candidates.jsonl (per-method `predictions`). CPU-only, standard library.
Usage: python gripper_semantics.py FILE [FILE ...]
"""
import argparse
import json
from itertools import groupby
from pathlib import Path

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


def rel2abs(commands):
    """Port of OpenVLA rel2abs_gripper_actions, returned in MultiNet's sign (+1 closed, -1 open)."""
    thresholded = [1 if c < -THRESHOLD else (-1 if c > THRESHOLD else 0) for c in commands]
    first = next((t for t in thresholded if t != 0), 0)
    state = -first if first != 0 else 1
    absolute = []
    for t in thresholded:
        if t != 0:
            state = t
        absolute.append(-state)
    return absolute


def corrected_targets(rows):
    """Replace the gripper target of every row by the episode-wise absolute state."""
    out = []
    for _, episode in groupby(rows, key=lambda r: r['episode']):
        episode = sorted(episode, key=lambda r: r['step'])
        states = rel2abs([r['target'][0] for r in episode])
        out.extend([s, *r['target'][1:]] for s, r in zip(states, episode))
    return out


def score(predictions, targets):
    m, dim = len(targets), len(targets[0])
    gripper = sum((p[0] - t[0]) ** 2 for p, t in zip(predictions, targets))
    motion = sum(sum((a - b) ** 2 for a, b in zip(p[1:], t[1:])) for p, t in zip(predictions, targets))
    mismatch = sum(p[0] != t[0] for p, t in zip(predictions, targets))
    return {'amse': (gripper + motion) / m / dim, 'gripper_part': gripper / m / dim,
            'other_dims_part': motion / m / dim, 'gripper_mismatch_rate': mismatch / m}


def constants(targets):
    """AMSE of constant predictors; `per_dimension_mean` is the MSE-optimal constant."""
    m, dim = len(targets), len(targets[0])
    mean = [sum(t[j] for t in targets) / m for j in range(dim)]
    result = {name: score([[g] + [0.0] * (dim - 1)] * m, targets)['amse']
              for name, g in (('zeros', 0.0), ('always_closed', 1.0), ('always_open', -1.0))}
    result['per_dimension_mean'] = score([mean] * m, targets)['amse']
    return result


def per_dimension_mse(predictions, targets):
    dim = len(targets[0])
    return [sum((p[j] - t[j]) ** 2 for p, t in zip(predictions, targets)) / len(targets) for j in range(dim)]


def observed_states(values, threshold):
    """Observed gripper signal -> MultiNet sign (+1 closed, -1 open)."""
    return [1 if v > threshold else -1 for v in values]


def state_to_command(predictions, states):
    return [[max(-1, min(1, p[0] - s))] + list(p[1:]) for p, s in zip(predictions, states)]


def symmetric_alignment(dataset, rows, predictions):
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


def dataset_of(path):
    text = str(path)
    return next((d for d in ('berkeley_autolab_ur5', 'nyu_door_opening_surprising_effectiveness')
                 if d in text), 'jaco_play')


def report(path):
    rows = sorted((json.loads(line) for line in Path(path).read_text().splitlines()),
                  key=lambda r: (r['episode'], r['step']))
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
        'dataset': dataset_of(path),
        'state_to_command_alignment': symmetric,
        'timesteps': len(rows), 'target_dim': len(original[0]),
        'no_change_target_share': no_change,
        'predicted_gripper_values': predicted_gripper_values[:10],
        'per_dimension_mse_multinet_targets': {
            **{m: per_dimension_mse(p, original) for m, p in methods.items()},
            'zeros': per_dimension_mse([[0.0] * len(original[0])] * len(rows), original)},
        'multinet_targets': {m: score(p, original) for m, p in methods.items()},
        'openvla_semantics_targets': {m: score(p, corrected) for m, p in methods.items()},
        'constant_predictors': {'multinet_targets': constants(original),
                                'openvla_semantics_targets': constants(corrected)},
        'per_dimension_mse_multinet_targets': {
            **{m: per_dimension_mse(p, original) for m, p in methods.items()},
            'zeros': per_dimension_mse([[0.0] * len(original[0])] * len(rows), original)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='*', type=Path)
    args = p.parse_args()
    files = args.files or [f for f in DEFAULT_FILES if f.exists()]
    print(json.dumps([report(f) for f in files], indent=2))


if __name__ == '__main__':
    main()
