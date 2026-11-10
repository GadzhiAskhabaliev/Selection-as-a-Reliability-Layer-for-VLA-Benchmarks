"""Eight-frame integration test of the unmodified long-run CS evaluator.

Only the data coverage is reduced: first three and final real frame from each
of the first two episodes. Real is_last flags are preserved. The saved scores
are diagnostics, not an evaluation of either complete episode.
"""
import json
import math
from pathlib import Path


def sparse_batches(loader, index_map):
    for episode, batch in enumerate(loader):
        if episode >= 2:
            break
        length = len(batch['continuous_observation'][0])
        if length < 4:
            raise ValueError('Smoke test requires at least four steps per source episode')
        indices = [0, 1, 2, length - 1]
        index_map.append(indices)
        yield {key: [[value[0][i] for i in indices]] if value is not None else None
               for key, value in batch.items()}


def validate_records(records, result, np, index_map, vocab_size=32000, action_bins=256):
    if len(records) != 8 or result['num_timesteps'] != 8 or len(index_map) != 2:
        raise AssertionError('Wrong smoke-test coverage')
    if [(row['episode'], row['step']) for row in records] != [(e, s) for e in range(2) for s in range(4)]:
        raise AssertionError('Unexpected smoke observation IDs')
    if [i for i, row in enumerate(records) if row['is_last']] != [3, 7]:
        raise AssertionError('Real terminal frames were not preserved')
    diverse = 0
    boundary_tokens = 0
    for row in records:
        candidates = row['candidates']
        if len(candidates) != 8 or any(len(c['raw_action']) != 7 or len(c['token_ids']) != 7 for c in candidates):
            raise AssertionError('Expected eight full 7D actions per observation')
        # np.digitize has 256 valid indices but only 255 bin centres. Upstream
        # intentionally maps the valid final edge token to the final centre.
        # That is not a language token or malformed action (ActionTokenizer doc).
        tokens = [t for c in candidates for t in c['token_ids']]
        if any(not vocab_size - action_bins <= t < vocab_size for t in tokens):
            raise AssertionError('Non-action token sampled; inspect before trusting the long run')
        boundary_tokens += sum(t == vocab_size - action_bins for t in tokens)
        if not all(math.isfinite(v) for c in candidates for v in c['raw_action']):
            raise AssertionError('Non-finite generated action')
        x = np.asarray(row['normalized_candidates'], dtype=np.float64)
        d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
        positive = d[np.triu_indices(8, k=1)]
        positive = positive[positive > 0]
        bandwidth = (float(np.median(positive)) if len(positive) else 1.0) + 1e-6
        scores = np.exp(-d ** 2 / (2 * bandwidth ** 2)).sum(axis=1)
        costs = d.sum(axis=1)
        selection = row['selection']
        if not np.allclose(selection['cs_scores'], scores, atol=1e-12, rtol=0):
            raise AssertionError('CS scores differ from independent NumPy reference')
        if not math.isclose(selection['bandwidth'], bandwidth, rel_tol=0, abs_tol=1e-12):
            raise AssertionError('Wrong median-distance bandwidth')
        indices = row['selected_indices']
        if any(not 0 <= index < 8 for index in indices.values()) or indices['sample1'] != 0:
            raise AssertionError('Selection index outside candidate pool')
        if not math.isclose(float(scores[indices['cs8']]), float(scores.max()), abs_tol=1e-12):
            raise AssertionError('CS did not choose a score maximizer')
        if not math.isclose(float(costs[indices['medoid8']]), float(costs.min()), abs_tol=1e-12):
            raise AssertionError('Medoid did not choose a distance minimizer')
        diverse += int(selection['num_unique'] > 1)
        for method, prediction in row['predictions'].items():
            expected = float(np.mean((np.array(prediction) - np.array(row['target'])) ** 2))
            if not math.isclose(row['mse'][method], expected, rel_tol=0, abs_tol=1e-12):
                raise AssertionError('Saved per-step metric differs from independent MSE')
    if diverse == 0:
        raise AssertionError('All eight pools collapsed; stochastic sampling was not demonstrated')
    for method, metric in result['methods'].items():
        values = np.asarray([row['mse'][method] for row in records])
        if not math.isclose(metric['amse'], float(values.mean()), rel_tol=0, abs_tol=1e-12):
            raise AssertionError('Saved AMSE does not match candidate log')
        if values.max() > values.min():
            expected = float(((values - values.min()) / (values.max() - values.min())).mean())
            if not math.isclose(metric['namse'], expected, rel_tol=0, abs_tol=1e-12):
                raise AssertionError('Saved NAMSE does not match candidate log')
        elif metric['namse'] is not None:
            raise AssertionError('Zero-range NAMSE must remain undefined')
    return {
        'coverage': 'PASS: 8 real frames, 2 real terminal frames',
        'greedy_decoder_parity': 'PASS: evaluator checked first 3 frames',
        'sampling': f'PASS: 64 candidates; {diverse}/8 pools have multiple unique actions',
        'action_tokens': f'PASS: 448 valid action tokens; {boundary_tokens} valid upper-edge tokens',
        'cs_and_medoid_reference': 'PASS: independent NumPy distance and kernel calculations',
        'metrics_and_output_roundtrip': 'PASS: JSON readback and independent AMSE/NAMSE',
    }


def run(baseline, selectors, pilot):
    pilot.PROTOCOL = {**pilot.PROTOCOL, 'name': 'jaco_cs8_smoke_v1', 'max_episodes': 2,
                      'episode_selection': 'first 3 and last real step of first 2 episodes',
                      'limitations': 'integration test only; sparse frames, no performance conclusion'}
    original_prepare = baseline.prepare_runtime
    original_save = baseline.save_json
    index_map = []
    runtime_holder = []

    def save(name, value):
        if name in ('manifest.json', 'progress.json', 'result.json'):
            value = {**value, 'scope': '8-frame integration smoke test only',
                     'complete_episodes': 0, 'source_episodes_sampled': len(index_map)}
            if name == 'manifest.json':
                value['kind'] = 'cs_smoke'
                value['expected_steps'] = 8
                value['expected_episodes'] = None
            if value.get('status') == 'complete_pilot':
                value['status'] = 'smoke_inference_finished_checks_pending'
        original_save(name, value)

    def prepare(**kwargs):
        runtime = original_prepare(**kwargs)
        original_loader = runtime.get_openx_dataloader
        runtime.get_openx_dataloader = lambda *a, **k: sparse_batches(original_loader(*a, **k), index_map)
        runtime_holder.append(runtime)
        return runtime

    baseline.prepare_runtime = prepare
    baseline.save_json = save
    pilot.run(baseline, selectors)
    records = [json.loads(line) for line in (baseline.OUT / 'candidates.jsonl').read_text().splitlines()]
    result = json.loads((baseline.OUT / 'result.json').read_text())
    model = runtime_holder[0].model
    checks = validate_records(records, result, runtime_holder[0].np, index_map,
                              vocab_size=model.vocab_size, action_bins=len(model.bins))
    for row in records:
        row['source_step'] = index_map[row['episode']][row['step']]
    with (baseline.OUT / 'candidates.jsonl').open('w') as output:
        for row in records:
            output.write(json.dumps(row, allow_nan=False) + '\n')
    result.update({'status': 'complete_smoke', 'checks': checks, 'source_step_indices': index_map,
                   'complete_episodes': 0, 'source_episodes_sampled': 2,
                   'long_run_guaranteed': False})
    original_save('result.json', result)
    manifest = json.loads((baseline.OUT / 'manifest.json').read_text())
    manifest['status'] = 'complete_smoke'
    original_save('manifest.json', manifest)
    original_save('smoke_checks.json', checks)
    print('SMOKE PASS', json.dumps(checks), flush=True)
