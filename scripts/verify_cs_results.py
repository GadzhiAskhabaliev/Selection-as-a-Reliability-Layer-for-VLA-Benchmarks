"""Independently audit downloaded CS pilot and smoke outputs with NumPy.

Does not run inference or change downloaded outputs. Run in a NumPy environment.
Read-only by default; --write-reports refreshes the derived audit JSON files.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'artifacts/cs_pilot/v2/output'
BASELINE = ROOT / 'artifacts/baseline_audit/current/output'
SMOKE = ROOT / 'artifacts/cs_smoke/v1/output'


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(a, b):
    assert np.allclose(a, b, atol=1e-12, rtol=0), (a, b)


def convert_jaco(action):
    g = action[-1]
    return [1 if g < 0.05 else (-1 if g > 0.95 else 0), *action[:3]]


def main(write_reports=False, include_smoke=False):
    rows = read_rows(PILOT / 'candidates.jsonl')
    result, manifest = read(PILOT / 'result.json'), read(PILOT / 'manifest.json')
    baseline = read_rows(BASELINE / 'predictions.jsonl')
    lookup = {(r['episode'], r['step']): r for r in baseline}
    expected = [r for r in baseline if r['episode'] < 20]
    assert result['status'] == 'complete_pilot'
    assert len(rows) == result['num_timesteps'] == len(expected) == 1487
    assert [(r['episode'], r['step']) for r in rows] == [(r['episode'], r['step']) for r in expected]
    counts = Counter(r['episode'] for r in rows)
    assert len(counts) == result['complete_episodes'] == 20
    assert read(PILOT / 'bundle_sources.json') == read(ROOT / 'kaggle/openvla_cs/source-hashes.json')
    for relative, expected_hash in read(PILOT / 'bundle_sources.json').items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash, relative
    assert hashlib.sha256((PILOT / 'candidates.jsonl').read_bytes()).hexdigest() == read(
        ROOT / 'artifacts/cs_pilot/verified_v2_result.json')['candidate_log_sha256']
    for ep in range(20):
        group = [r for r in rows if r['episode'] == ep]
        assert sum(r['is_last'] for r in group) == 1 and group[-1]['is_last']

    stats = manifest['action_statistics']
    lo, hi, mask = np.array(stats['q01']), np.array(stats['q99']), np.array(stats['mask'])
    span = hi - lo
    bins = np.linspace(-1, 1, 256)
    centres = (bins[:-1] + bins[1:]) / 2
    names = list(result['methods'])
    errors = {name: [] for name in names}
    components = {name: [] for name in names}
    rng = random.Random(7 + 100003)
    upper_edge_tokens = identical_pools = agreement = unique_sum = 0
    for row in rows:
        base = lookup[row['episode'], row['step']]
        assert all(row[k] == base[k] for k in ('image_sha256', 'instruction', 'target', 'is_last'))
        assert row['predictions']['greedy'] == base['prediction']
        seed_key = f'7:jaco_cs8:{row["episode"]}:{row["step"]}'.encode()
        assert row['pool_seed'] == int.from_bytes(hashlib.sha256(seed_key).digest()[:8], 'little') % (2 ** 31)
        raw = np.asarray([c['raw_action'] for c in row['candidates']])
        assert raw.shape == (8, 7) and np.isfinite(raw).all()
        tokens = np.asarray([c['token_ids'] for c in row['candidates']])
        assert tokens.shape == (8, 7) and np.all((tokens >= 31744) & (tokens < 32000))
        upper_edge_tokens += int((tokens == 31744).sum())
        normalized_tokens = centres[np.clip(32000 - tokens - 1, 0, 254)]
        close(raw, np.where(mask, 0.5 * (normalized_tokens + 1) * span + lo, normalized_tokens))
        normalized = np.where(mask, np.divide(2 * (raw - lo), span,
                              out=np.zeros_like(raw), where=span != 0) - 1, raw)
        normalized[:, mask & (span == 0)] = 0
        close(normalized, row['normalized_candidates'])
        d = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)
        positive = d[d > 0]
        h = (float(np.median(positive)) if len(positive) else 1.0) + 1e-6
        scores = np.exp(-0.5 * (d / h) ** 2).sum(axis=1)
        costs = d.sum(axis=1)
        close(scores, row['selection']['cs_scores'])
        close(costs, row['selection']['medoid_costs'])
        close(h, row['selection']['bandwidth'])
        indices = row['selected_indices']
        assert all(0 <= i < 8 for i in indices.values())
        assert indices['sample1'] == 0 and indices['random8'] == rng.randrange(8)
        close(scores[indices['cs8']], scores.max())
        close(costs[indices['medoid8']], costs.min())
        unique = len(np.unique(normalized, axis=0))
        assert row['selection']['num_unique'] == unique
        unique_sum += unique
        identical_pools += int(unique == 1)
        agreement += int(indices['cs8'] == indices['medoid8'])
        for name in names:
            action = row['greedy_raw'] if name == 'greedy' else raw[indices[name]]
            prediction = np.array(convert_jaco(action))
            close(prediction, row['predictions'][name])
            squared = (prediction - np.array(row['target'])) ** 2
            error = float(squared.mean())
            close(error, row['mse'][name])
            errors[name].append(error)
            components[name].append(squared)

    methods = {}
    for name in names:
        x = np.array(errors[name])
        close(x.mean(), result['methods'][name]['amse'])
        close(((x - x.min()) / (x.max() - x.min())).mean(), result['methods'][name]['namse'])
        methods[name] = {**result['methods'][name],
                         'amse_change_vs_greedy_percent': 100 * (x.mean() / np.mean(errors['greedy']) - 1),
                         'coordinate_mse_gripper_x_y_z': np.mean(components[name], axis=0).tolist()}

    # Verify saved episode totals and reproduce the paired bootstrap schedule.
    episode = read(PILOT / 'episode_errors.json')
    for ep, entry in enumerate(episode):
        assert entry['episode'] == ep and entry['num_steps'] == counts[ep]
        for name in names:
            close(entry['error_sums'][name], math.fsum(row['mse'][name] for row in rows if row['episode'] == ep))
    sizes = np.array([counts[ep] for ep in range(20)])
    sums = np.array([[entry['error_sums'][name] for name in names] for entry in episode])
    boot_rng = random.Random(7 + 200003)
    indices = np.array([boot_rng.choices(range(20), k=20) for _ in range(2000)])
    means = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
    greedy = names.index('greedy')
    for name in names[1:]:
        values = np.sort(means[:, names.index(name)] - means[:, greedy])
        close([values[49], values[1949]], result['episode_bootstrap'][name]['delta_amse_95pct_episode_bootstrap'])
    contrasts = {}
    for control in ('greedy', 'sample1', 'random8', 'medoid8'):
        values = np.sort(means[:, names.index('cs8')] - means[:, names.index(control)])
        contrasts['cs8_minus_' + control] = {
            'delta_amse': methods['cs8']['amse'] - methods[control]['amse'],
            '95pct_episode_bootstrap': [float(values[49]), float(values[1949])],
        }
    assert upper_edge_tokens == result['diagnostics']['clipped_action_tokens']
    assert unique_sum == result['diagnostics']['sum_unique_candidates']
    assert identical_pools == result['diagnostics']['collapsed_candidate_pools']
    assert agreement == result['diagnostics']['cs_medoid_same_index']
    report = {
        'status': 'PASS', 'verified_at': datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        'complete_episodes': 20, 'num_timesteps': len(rows), 'candidate_actions': len(rows) * 8,
        'all_paired_greedy_actions_match_full_baseline': True, 'methods': methods,
        'paired_amse_contrasts': contrasts, 'evaluation_minutes': result['elapsed_seconds'] / 60,
        'diagnostics': {**result['diagnostics'], 'invalid_action_tokens': 0,
                        'valid_upper_edge_tokens': upper_edge_tokens,
                        'collapsed_pool_percent': 100 * identical_pools / len(rows),
                        'candidate_pool_to_greedy_latency_ratio': result['diagnostics']['candidate_pool_seconds'] / result['diagnostics']['greedy_seconds']},
        'candidate_log_sha256': hashlib.sha256((PILOT / 'candidates.jsonl').read_bytes()).hexdigest(),
        'limitations': '20 test episodes, one sampling seed and fixed settings; no online success-rate conclusion',
        'conclusion': 'CS does not improve paired greedy AMSE/NAMSE in this pilot; CS AMSE is higher',
    }
    path = ROOT / 'artifacts/cs_pilot/verified_v2_result.json'
    if write_reports:
        path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if include_smoke:
        spec = importlib.util.spec_from_file_location('fixed_smoke', ROOT / 'experiments/offline/cs_smoke.py')
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        smoke_rows = read_rows(SMOKE / 'candidates.jsonl')
        index_map = [[0, 1, 2, sum(r['episode'] == ep for r in baseline) - 1] for ep in range(2)]
        checks = smoke.validate_records(smoke_rows, read(SMOKE / 'result.json'), np, index_map)
        for row in smoke_rows:
            source = lookup[row['episode'], index_map[row['episode']][row['step']]]
            assert all(row[k] == source[k] for k in ('image_sha256', 'instruction', 'target', 'is_last'))
            assert row['predictions']['greedy'] == source['prediction']
        verified = {'status': 'PASS_AFTER_CHECKER_FIX', 'remote_kernel_status': 'ERROR',
                    'original_failure': read(SMOKE / 'failure.json'),
                    'explanation': 'The original checker rejected legal upper-edge action token 31744. '
                                   'All 8 GPU inference steps finished; saved outputs pass the corrected local checker.',
                    'checks': checks, 'source_step_indices': index_map,
                    'validator_sha256': hashlib.sha256((ROOT / 'experiments/offline/cs_smoke.py').read_bytes()).hexdigest(),
                    'candidate_log_sha256': hashlib.sha256((SMOKE / 'candidates.jsonl').read_bytes()).hexdigest()}
        saved = read(ROOT / 'artifacts/cs_smoke/verified_v1_result.json')
        assert verified == saved
        if write_reports:
            (ROOT / 'artifacts/cs_smoke/verified_v1_result.json').write_text(json.dumps(verified, indent=2))
        print('SMOKE REVALIDATION', json.dumps(verified, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-reports', action='store_true')
    parser.add_argument('--include-smoke', action='store_true', help='Also verify optional local integration outputs')
    args = parser.parse_args()
    main(write_reports=args.write_reports, include_smoke=args.include_smoke)
