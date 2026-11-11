"""Re-score archived Jaco Play predictions with gripper targets in OpenVLA's training semantics.

Jaco targets are relative gripper commands (+1 close, -1 open, 0 no change). OpenVLA is trained on
absolute gripper state obtained with `rel2abs_gripper_actions` (openvla/openvla,
prismatic/vla/datasets/rlds/utils/data_utils.py), and MultiNet maps its output to +1 closed /
-1 open. Here the targets of each episode are converted with the same rel2abs rule and the
same sign convention, so prediction and target describe the same quantity. Motion is unchanged.
Reads only archived outputs; CPU-only, standard library.
"""
import json
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'artifacts/baseline_audit/current/output/predictions.jsonl'
PILOT = ROOT / 'artifacts/cs_pilot/v2/output/candidates.jsonl'
DIM = 4


def rel2abs(commands):
    """Port of OpenVLA rel2abs_gripper_actions, returned in MultiNet's sign (+1 closed, -1 open)."""
    thresholded = [1 if c < -0.1 else (-1 if c > 0.1 else 0) for c in commands]
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
    m = len(targets)
    gripper = sum((p[0] - t[0]) ** 2 for p, t in zip(predictions, targets))
    motion = sum(sum((a - b) ** 2 for a, b in zip(p[1:], t[1:])) for p, t in zip(predictions, targets))
    mismatch = sum(p[0] != t[0] for p, t in zip(predictions, targets))
    return {'amse': (gripper + motion) / m / DIM, 'gripper_part': gripper / m / DIM,
            'motion_part': motion / m / DIM, 'gripper_state_error_rate': mismatch / m}


def main():
    rows = [json.loads(line) for line in BASELINE.read_text().splitlines()]
    rows.sort(key=lambda r: (r['episode'], r['step']))
    original = [r['target'] for r in rows]
    corrected = corrected_targets(rows)
    predictions = [r['prediction'] for r in rows]
    constant = {name: [[g, 0.0, 0.0, 0.0]] * len(rows) for name, g in
                (('always_closed', 1.0), ('always_open', -1.0), ('zeros', 0.0))}
    report = {
        'full_test_openvla_greedy': {'multinet_targets': score(predictions, original),
                                     'openvla_semantics_targets': score(predictions, corrected)},
        'constant_predictors_openvla_semantics': {k: score(v, corrected) for k, v in constant.items()},
        'constant_predictors_multinet_targets': {k: score(v, original) for k, v in constant.items()},
        'closed_share_openvla_semantics': sum(t[0] == 1 for t in corrected) / len(corrected),
    }
    pilot = [json.loads(line) for line in PILOT.read_text().splitlines()]
    pilot.sort(key=lambda r: (r['episode'], r['step']))
    pilot_corrected = corrected_targets(pilot)
    report['pilot_20_episodes_openvla_semantics'] = {
        method: score([r['predictions'][method] for r in pilot], pilot_corrected)
        for method in ('greedy', 'sample1', 'random8', 'medoid8', 'cs8')}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
