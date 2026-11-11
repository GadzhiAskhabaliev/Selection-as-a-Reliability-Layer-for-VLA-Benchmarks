"""Independently audit archived motion-selection runs with NumPy; no inference.

Default: verify both completed runs, read-only. --write-reports saves summaries.
Do not use python -O: the audit deliberately uses assertions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'jaco': (ROOT / 'artifacts/overnight_motion/jaco', 'jaco'),
    'nyu': (ROOT / 'artifacts/overnight_motion/nyu_door/v2', 'nyu_door'),
}
PRIMARY = 'motion_cs8_greedy_gripper'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert np.allclose(actual, expected, atol=1e-12, rtol=0), (actual, expected)


def audit(key, write_reports=False):
    directory, config_name = RUNS[key]
    output = directory / 'output'
    result, protocol, manifest = [read(output / f'{name}.json')
                                  for name in ('result', 'protocol', 'manifest')]
    assert read(directory / 'status.json')['status'] == 'COMPLETE'
    assert result['status'] == manifest['status'] == 'complete_motion_pilot'
    assert protocol == read(ROOT / f'configs/motion/{config_name}.json') == manifest['protocol']
    assert read(output / 'bundle_sources.json') == read(ROOT / f'kaggle/motion_{config_name}/source-hashes.json')
    for relative, expected in read(output / 'bundle_sources.json').items():
        assert sha(ROOT / relative) == expected, relative
    assert manifest['model'] == 'openvla/openvla-7b'
    assert manifest['revision'] == '47a0ec7fc4ec123775a391911046cf33cf9ed83f'
    assert manifest['multinet_commit'] == '546a1c77f9a4cc0544c5e4b41565949778d32441'
    assert manifest['quantization'] is None
    assert not result['online_success_rate_measured']
    assert read(output / 'startup_checks.json')['observations_checked'] == 3
    rows = [json.loads(line) for line in (output / 'candidates.jsonl').read_text().splitlines()]
    episodes = read(output / 'episode_errors.json')
    sizes = [ep['num_steps'] for ep in episodes]
    start, count = protocol['episode_start'], protocol['episode_count']
    assert len(rows) == sum(sizes) == result['num_timesteps'] == protocol['expected_eval_steps']
    assert len(episodes) == result['complete_episodes'] == count
    assert [ep['episode'] for ep in episodes] == list(range(start, start + count))
    expected_ids = [(ep['episode'], step) for ep in episodes for step in range(ep['num_steps'])]
    assert [(r['episode'], r['step']) for r in rows] == expected_ids
    size_map = dict(zip(range(start, start + count), sizes))
    baseline = None
    if key == 'jaco':
        baseline = {(r['episode'], r['step']): r for r in map(json.loads,
                    (ROOT / 'artifacts/baseline_audit/current/output/predictions.jsonl').read_text().splitlines())}
    else:
        boundary = read(output / 'data_boundary_audit.json')
        independent = read(directory.parent / 'data_audit/full_split_record_boundaries.json')
        assert boundary['status'] == independent['status'] == 'PASS'
        assert boundary['mapped_pipeline_verified'] and boundary['dropped_steps'] == 0
        assert sizes == protocol['source_episode_lengths'] == boundary['episode_lengths'] == independent['episode_lengths']
        assert boundary['num_source_episodes'] == independent['num_episodes'] == count == 49
        assert boundary['num_source_steps'] == independent['num_steps'] == len(rows) == 2209
        assert boundary['effective_is_last_count'] == count
        assert boundary['raw_is_last_count'] == independent['raw_is_last_count'] == 98
        for ep, (record, length) in enumerate(zip(boundary['records'], sizes)):
            assert record['episode'] == ep and record['num_steps'] == length
            assert record['is_first_indices'] == [0]
            assert record['is_last_indices'] == record['is_terminal_indices'] == [length - 2, length - 1]

    stats = manifest['action_statistics']
    lo, hi, mask = np.array(stats['q01']), np.array(stats['q99']), np.array(stats['mask'])
    span = hi - lo
    bins = np.linspace(-1, 1, 256)
    centres = (bins[:-1] + bins[1:]) / 2
    names = protocol['methods']
    assert set(names) == set(result['methods'])
    errors, components = {n: [] for n in names}, {n: [] for n in names}
    rng = random.Random(protocol['random_selector_seed'])
    unique_sum = collapsed = agreement = edge_tokens = 0
    for row in rows:
        ep, step = row['episode'], row['step']
        assert row['is_last'] == (step == size_map[ep] - 1)
        if baseline is not None:
            base = baseline[ep, step]
            assert all(row[k] == base[k] for k in ('image_sha256', 'instruction', 'target', 'is_last'))
            assert row['greedy_raw'] == base['raw_prediction']
            assert row['predictions']['greedy'] == base['prediction']
        namespace = 'jaco_cs8' if key == 'jaco' else protocol['dataset'] + '_cs8'
        seed_key = f'{protocol["seed"]}:{namespace}:{ep}:{step}'.encode()
        assert row['pool_seed'] == int.from_bytes(hashlib.sha256(seed_key).digest()[:8], 'little') % (2 ** 31)
        raw = np.asarray([c['raw_action'] for c in row['candidates']])
        tokens = np.asarray([c['token_ids'] for c in row['candidates']])
        assert raw.shape == tokens.shape == (8, 7) and np.isfinite(raw).all()
        assert np.all((tokens >= 31744) & (tokens < 32000))
        for candidate, token_row in zip(row['candidates'], tokens):
            assert candidate['clipped_token_count'] == int((token_row == 31744).sum())
        edge_tokens += int((tokens == 31744).sum())
        normalized_tokens = centres[np.clip(32000 - tokens - 1, 0, 254)]
        close(raw, np.where(mask, 0.5 * (normalized_tokens + 1) * span + lo, normalized_tokens))
        normalized = np.where(mask, np.divide(2 * (raw - lo), span,
                              out=np.zeros_like(raw), where=span != 0) - 1, raw)
        normalized[:, mask & (span == 0)] = 0
        close(normalized, row['normalized_candidates'])
        for name, vectors in [('full_selection', normalized), ('motion_selection', normalized[:, :6])]:
            d = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
            positive = d[d > 0]
            h = (float(np.median(positive)) if len(positive) else 1.0) + 1e-6
            scores, costs = np.exp(-0.5 * (d / h) ** 2).sum(axis=1), d.sum(axis=1)
            selection = row[name]
            close(h, selection['bandwidth'])
            close(scores, selection['cs_scores'])
            close(costs, selection['medoid_costs'])
            close(scores[selection['cs_index']], scores.max())
            close(costs[selection['medoid_index']], costs.min())
            assert selection['num_unique'] == len(np.unique(vectors, axis=0))
        full, motion = row['full_selection'], row['motion_selection']
        random_index = rng.randrange(8)
        indices = {'sample1': 0, 'random8': random_index,
                   'cs8': full['cs_index'], 'medoid8': full['medoid_index'],
                   'sample1_greedy_gripper': 0, 'random8_greedy_gripper': random_index,
                   'motion_cs8_greedy_gripper': motion['cs_index'],
                   'motion_medoid8_greedy_gripper': motion['medoid_index']}
        assert row['selected_indices'] == indices
        unique_sum += motion['num_unique']
        collapsed += int(motion['num_unique'] == 1)
        agreement += int(motion['cs_index'] == motion['medoid_index'])
        target = np.asarray(row['target'])
        assert target.shape == (protocol['eval_action_dim'],) and np.isfinite(target).all()
        for name in names:
            action = np.array(row['greedy_raw']) if name == 'greedy' else raw[indices[name]].copy()
            if name.endswith('_greedy_gripper'):
                action[-1] = row['greedy_raw'][-1]
            close(action, row['raw_actions'][name])
            g = action[-1]
            if key == 'jaco':
                prediction = np.array([1 if g < 0.05 else (-1 if g > 0.95 else 0), *action[:3]])
            else:
                prediction = np.array([1 - 2 * g, *action[3:6], *action[:3]])
            close(prediction, row['predictions'][name])
            if name.endswith('_greedy_gripper'):
                close(prediction[0], row['predictions']['greedy'][0])
            squared = (prediction - target) ** 2
            close(squared.mean(), row['mse'][name])
            errors[name].append(float(squared.mean()))
            components[name].append(squared)

    for name in names:
        values, saved = np.array(errors[name]), result['methods'][name]
        close(values.mean(), saved['amse'])
        close(values.min(), saved['min_mse'])
        close(values.max(), saved['max_mse'])
        close(((values - values.min()) / (values.max() - values.min())).mean(), saved['namse'])
        close(np.mean(components[name], axis=0), saved['coordinate_mse'])
        close(saved['coordinate_mse'][0], saved['gripper_mse'])
        close(np.mean(saved['coordinate_mse'][1:]), saved['motion_mse'])
        close(saved['amse'] - result['methods']['greedy']['amse'], saved['delta_amse_vs_greedy'])
    offset = 0
    for ep in episodes:
        for name in names:
            close(math.fsum(errors[name][offset:offset + ep['num_steps']]), ep['error_sums'][name])
        offset += ep['num_steps']
    boot_rng = random.Random(protocol['bootstrap_seed'])
    draws = np.array([boot_rng.choices(range(count), k=count) for _ in range(2000)])
    sizes = np.array(sizes)
    sums = {n: np.array([e['error_sums'][n] for e in episodes]) for n in names}
    for label, contrast in result['paired_contrasts'].items():
        control = label.removeprefix(PRIMARY + '_minus_')
        values = np.sort((sums[PRIMARY] - sums[control])[draws].sum(axis=1) / sizes[draws].sum(axis=1))
        close([values[49], values[1949]], contrast['95pct_episode_bootstrap'])
        close(result['methods'][PRIMARY]['amse'] - result['methods'][control]['amse'], contrast['delta_amse'])
        assert contrast['resamples'] == 2000
    diag = result['diagnostics']
    assert diag['candidate_actions'] == len(rows) * 8
    assert diag['motion_unique_sum'] == unique_sum
    assert diag['motion_collapsed_pools'] == collapsed
    assert diag['motion_cs_medoid_same_index'] == agreement
    assert diag['valid_edge_token_count'] == edge_tokens
    primary, greedy = result['methods'][PRIMARY], result['methods']['greedy']
    reference = read(output / 'upstream_reference_NOT_OUR_RESULT.json')
    report = {
        'status': 'PASS', 'dataset': protocol['dataset'], 'complete_episodes': count,
        'num_timesteps': len(rows), 'candidate_actions': len(rows) * 8,
        'checks': ['coverage_and_boundaries', 'source_hashes_and_protocol', 'action_token_decoding',
                   'normalization', 'full_and_motion_cs_medoid_geometry', 'random_and_pool_seeds',
                   'hybrid_gripper_and_action_conversion', 'all_step_and_aggregate_errors',
                   'episode_totals_and_paired_bootstrap'],
        'paired_greedy_matches_original_baseline': True if baseline is not None else None,
        'amse_change_vs_greedy_percent': 100 * (primary['amse'] / greedy['amse'] - 1),
        'motion_mse_change_vs_greedy_percent': 100 * (primary['motion_mse'] / greedy['motion_mse'] - 1),
        'namse_change_vs_greedy_percent': 100 * (primary['namse'] / greedy['namse'] - 1),
        'methods': result['methods'], 'paired_contrasts': result['paired_contrasts'],
        'evaluation_hours': result['elapsed_seconds'] / 3600,
        'diagnostics': {**diag, 'cs_medoid_agreement_percent': 100 * agreement / len(rows),
                        'mean_unique_motion_candidates': unique_sum / len(rows),
                        'pool_to_greedy_latency_ratio': diag['pool_seconds'] / diag['greedy_seconds']},
        'candidate_log_sha256': sha(output / 'candidates.jsonl'),
        'limitations': 'One sampling seed; offline only; no online success-rate evidence.',
    }
    if key == 'nyu':
        assert reference['num_timesteps'] == len(rows)
        report['greedy_amse_vs_upstream_percent'] = 100 * (greedy['amse'] / reference['avg_dataset_amse'] - 1)
        report['greedy_namse_vs_upstream_percent'] = 100 * (greedy['namse'] / reference['normalized_amse'] - 1)
    if write_reports:
        (directory / 'verified_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('methods', 'diagnostics')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['jaco', 'nyu', 'all'], default='all')
    parser.add_argument('--write-reports', action='store_true')
    args = parser.parse_args()
    for key in RUNS if args.dataset == 'all' else [args.dataset]:
        audit(key, args.write_reports)
