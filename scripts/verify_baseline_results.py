"""Read-only, standard-library audit of the archived full Jaco baseline.

Recalculates metrics from predictions without importing the inference evaluator.
Run without Python's -O flag: assertions are deliberate audit checks.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/baseline_audit/current/output'


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, atol=1e-12):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=atol), (actual, expected)


def main():
    path = OUTPUT / 'predictions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = read(OUTPUT / 'result.json')
    manifest = read(OUTPUT / 'manifest.json')
    reference = read(OUTPUT / 'upstream_reference_NOT_OUR_RESULT.json')
    audit = read(ROOT / 'artifacts/baseline_audit/verified_jaco_result.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['prediction_log_sha256']
    assert result['status'] == 'complete'
    assert len(rows) == result['num_timesteps'] == manifest['expected_steps'] == 7838
    groups = defaultdict(list)
    errors = []
    for row in rows:
        groups[row['episode']].append(row)
        raw, prediction, target = row['raw_prediction'], row['prediction'], row['target']
        assert len(raw) == 7 and len(prediction) == len(target) == 4
        assert all(math.isfinite(v) for v in [*raw, *prediction, *target])
        grip = 1 if raw[-1] < 0.05 else (-1 if raw[-1] > 0.95 else 0)
        for actual, expected in zip(prediction, [grip, *raw[:3]]):
            close(actual, expected)
        error = math.fsum((p - t) ** 2 for p, t in zip(prediction, target)) / 4
        close(error, row['mse'])
        errors.append(error)
    assert list(groups) == list(range(109))
    assert len(groups) == result['complete_episodes'] == manifest['expected_episodes']
    assert [(r['episode'], r['step']) for r in rows] == [
        (ep, step) for ep, group in groups.items() for step in range(len(group))]
    for group in groups.values():
        assert [r['step'] for r in group] == list(range(len(group)))
        assert sum(r['is_last'] for r in group) == 1 and group[-1]['is_last']
    amse = math.fsum(errors) / len(errors)
    lo, hi = min(errors), max(errors)
    assert hi > lo and result['namse_defined']
    namse = math.fsum((error - lo) / (hi - lo) for error in errors) / len(errors)
    close(lo, result['min_mse'])
    close(hi, result['max_mse'])
    for name, value, key in [('amse', amse, 'avg_dataset_amse'),
                             ('namse', namse, 'normalized_amse')]:
        close(value, result[name])
        close(value, audit['metrics'][name]['ours'])
        close(reference[key], audit['metrics'][name]['upstream'])
        close(value - reference[key], result['delta_' + name + '_vs_published'])
        close(value - reference[key], audit['metrics'][name]['absolute_delta'])
        # Percentage scaling amplifies the saved sum vs. fsum round-off.
        close(100 * (value / reference[key] - 1), audit['metrics'][name]['relative_delta_percent'], atol=1e-10)
    final_matches = sum(group[-1]['prediction'] == group[-1]['target'] for group in groups.values())
    close(100 * final_matches / len(groups), result['offline_final_action_exact_match_percent'])
    print(json.dumps({'status': 'PASS', 'episodes': len(groups), 'steps': len(rows),
                      'amse': amse, 'namse': namse,
                      'note': 'Offline prediction metrics; no online success rate.'}, indent=2))


if __name__ == '__main__':
    main()
