"""Decompose archived Jaco Play AMSE into the unreachable gripper floor, gripper sign errors and motion.

MultiNet maps Jaco targets to a ternary gripper {-1, 0, +1} (0 = no movement), while OpenVLA's
binary gripper output is converted to +/-1 only. Frames with target 0 therefore cost exactly 1
in squared gripper error for any OpenVLA prediction and any selector over OpenVLA samples.
Reads only archived outputs; CPU-only, standard library.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'artifacts/baseline_audit/current/output/predictions.jsonl'
PILOT = ROOT / 'artifacts/cs_pilot/v2/output/candidates.jsonl'
DIM = 4


def decompose(pairs):
    m = len(pairs)
    floor = reach = motion = 0.0
    reachable = sign_errors = 0
    for prediction, target in pairs:
        g = (prediction[0] - target[0]) ** 2
        if target[0] == 0:
            floor += g
        else:
            reach += g
            reachable += 1
            sign_errors += prediction[0] != target[0]
        motion += sum((p - t) ** 2 for p, t in zip(prediction[1:], target[1:]))
    return {'amse': (floor + reach + motion) / m / DIM, 'gripper_floor': floor / m / DIM,
            'gripper_reachable': reach / m / DIM, 'motion_xyz': motion / m / DIM,
            'target_no_move_share': 1 - reachable / m, 'sign_error_rate_reachable': sign_errors / reachable}


def main():
    rows = [json.loads(line) for line in BASELINE.read_text().splitlines()]
    targets = [r['target'] for r in rows]
    report = {'full_test_greedy': decompose([(r['prediction'], r['target']) for r in rows]),
              'all_zeros_predictor_amse': sum(sum(v * v for v in t) for t in targets) / len(targets) / DIM}
    pilot = [json.loads(line) for line in PILOT.read_text().splitlines()]
    report['pilot_20_episodes'] = {method: decompose([(r['predictions'][method], r['target']) for r in pilot])
                                   for method in ('greedy', 'sample1', 'random8', 'medoid8', 'cs8')}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
