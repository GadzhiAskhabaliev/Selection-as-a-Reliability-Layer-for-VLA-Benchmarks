"""Frozen OpenVLA: select motion, preserve greedy gripper, evaluate full actions.

The original baseline runtime and CS pilot stay byte-for-byte frozen. The bundle
injects them along with a dataset protocol. No labels enter select_actions().
"""
import hashlib
import json
import math
import random
import time

METHODS = ['greedy', 'sample1', 'random8', 'medoid8', 'cs8',
           'sample1_greedy_gripper', 'random8_greedy_gripper',
           'motion_medoid8_greedy_gripper', 'motion_cs8_greedy_gripper']
HYBRIDS = METHODS[5:]


def pool_seed(protocol, episode, step):
    namespace = 'jaco_cs8' if protocol['dataset'] == 'jaco_play' else protocol['dataset'] + '_cs8'
    key = f'{protocol["seed"]}:{namespace}:{episode}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2 ** 31)


def select_actions(selectors, raw, greedy, stats, random_index):
    """Return full and motion-only selections plus full 7D hybrid actions."""
    if len(greedy) != 7 or any(len(a) != 7 for a in raw):
        raise ValueError('This protocol requires 7D raw actions, gripper at index 6')
    if not all(math.isfinite(float(x)) for x in greedy):
        raise ValueError('Non-finite greedy action')
    if not 0 <= random_index < len(raw):
        raise ValueError('Invalid random selector index')
    normalized = selectors.normalize_openvla(raw, stats)
    full = selectors.select([[a] for a in normalized])
    motion = selectors.select([[a[:6]] for a in normalized])
    indices = {'sample1': 0, 'random8': random_index,
               'medoid8': full['medoid_index'], 'cs8': full['cs_index'],
               'sample1_greedy_gripper': 0, 'random8_greedy_gripper': random_index,
               'motion_medoid8_greedy_gripper': motion['medoid_index'],
               'motion_cs8_greedy_gripper': motion['cs_index']}
    actions = {'greedy': list(greedy)}
    for name, index in indices.items():
        actions[name] = list(raw[index])
        if name in HYBRIDS:
            actions[name][6] = float(greedy[6])
    return {'normalized_candidates': normalized, 'full_selection': full,
            'motion_selection': motion, 'selected_indices': indices, 'raw_actions': actions}


def generate_action(runtime, inputs, sample, protocol, decoder):
    torch, model = runtime.torch, runtime.model
    kwargs = dict(inputs)
    ids = kwargs.pop('input_ids')
    if not torch.all(ids[:, -1] == 29871):
        ids = torch.cat((ids, torch.tensor([[29871]], device=ids.device)), dim=1)
    dataset = protocol['dataset']
    dim = model.get_action_dim(dataset)
    if dim != 7:
        raise ValueError('Unexpected action dimensionality')
    sampling = {k: protocol[k] for k in ('temperature', 'top_k', 'top_p')} if sample else {}
    with torch.inference_mode():
        generated = model.generate(ids, **kwargs, max_new_tokens=dim, do_sample=sample, **sampling)
    if generated.shape[0] != 1 or generated.shape[1] - ids.shape[1] != dim:
        raise ValueError('Truncated generated action')
    tokens = generated[0, -dim:].cpu().tolist()
    # 256 valid tokens and 255 bin centres: the legal upper edge is not rejected.
    if any(t < model.vocab_size - len(model.bins) or t >= model.vocab_size for t in tokens):
        raise ValueError('Non-action token sampled; abort rather than hide invalid proposals')
    return decoder(tokens, model.vocab_size, model.bin_centers, model.get_action_stats(dataset))


def independent_startup_check(np, selection, raw, greedy, predictions):
    """Independent NumPy geometry check on the first three real observations."""
    values = np.array(selection['normalized_candidates'])
    for name, vectors in [('full_selection', values), ('motion_selection', values[:, :6])]:
        distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
        positive = distances[distances > 0]
        bandwidth = (float(np.median(positive)) if len(positive) else 1.0) + 1e-6
        scores = np.exp(-0.5 * (distances / bandwidth) ** 2).sum(axis=1)
        costs = distances.sum(axis=1)
        result = selection[name]
        if not (np.allclose(scores, result['cs_scores'], atol=1e-12, rtol=0)
                and np.allclose(costs, result['medoid_costs'], atol=1e-12, rtol=0)
                and np.isclose(bandwidth, result['bandwidth'], atol=1e-12, rtol=0)
                and np.isclose(scores[result['cs_index']], scores.max(), atol=1e-12, rtol=0)
                and np.isclose(costs[result['medoid_index']], costs.min(), atol=1e-12, rtol=0)):
            raise AssertionError('Independent geometry check failed')
    for name in HYBRIDS:
        action = selection['raw_actions'][name]
        assert action[:6] == list(raw[selection['selected_indices'][name]])[:6]
        assert action[6] == float(greedy[6])
        assert predictions[name][0] == predictions['greedy'][0]


def summarize(baseline, protocol, errors, coordinate_sums, episodes, diagnostics, elapsed, status):
    methods = {name: baseline.aggregate(x) for name, x in errors.items()}
    n = len(errors['greedy'])
    for name, result in methods.items():
        coordinates = [v / n for v in coordinate_sums[name]]
        result.update({'delta_amse_vs_greedy': result['amse'] - methods['greedy']['amse'],
                       'coordinate_mse': coordinates, 'gripper_mse': coordinates[0],
                       'motion_mse': math.fsum(coordinates[1:]) / (len(coordinates) - 1)})
    intervals = {}
    if status == 'complete_motion_pilot':
        primary = 'motion_cs8_greedy_gripper'
        controls = ['greedy', 'cs8', 'sample1_greedy_gripper', 'random8_greedy_gripper',
                    'motion_medoid8_greedy_gripper']
        draws = {name: [] for name in controls}
        rng = random.Random(protocol['seed'] + 200003)
        for _ in range(2000):
            chosen = rng.choices(episodes, k=len(episodes))
            count = sum(ep['num_steps'] for ep in chosen)
            for name in controls:
                draws[name].append(math.fsum(ep['error_sums'][primary] - ep['error_sums'][name]
                                             for ep in chosen) / count)
        for name, values in draws.items():
            values.sort()
            intervals[primary + '_minus_' + name] = {
                'delta_amse': methods[primary]['amse'] - methods[name]['amse'],
                '95pct_episode_bootstrap': [values[49], values[1949]], 'resamples': 2000}
    return {'status': status, 'dataset': protocol['dataset'], 'split': protocol['split'],
            'episode_start': protocol['episode_start'], 'complete_episodes': len(episodes),
            'num_timesteps': n, 'elapsed_seconds': elapsed, 'methods': methods,
            'paired_contrasts': intervals, 'diagnostics': diagnostics,
            'scope': protocol['scope'], 'online_success_rate_measured': False}


def run(baseline, selectors, original_pilot, protocol):
    baseline.DATASET = protocol['dataset']
    baseline.DATA_VERSION = protocol['data_version']
    baseline.SPLIT = protocol['split']
    baseline.EXPECTED_EPISODES = protocol['split_episodes']
    baseline.EXPECTED_STEPS = protocol['full_split_expected_steps']
    baseline.SEED = protocol['seed']
    baseline.OUT.mkdir(parents=True, exist_ok=True)
    baseline.save_json('protocol.json', protocol)
    runtime = baseline.prepare_runtime(kind='motion_cs_pilot')
    np, torch = runtime.np, runtime.torch
    stats = runtime.model.get_action_stats(protocol['dataset'])
    runtime.manifest.update({'status': 'evaluating_motion_pilot', 'sampling': True,
                             'protocol': protocol, 'action_statistics': stats,
                             'expected_episodes': protocol['episode_count'],
                             'expected_steps': protocol['expected_eval_steps'],
                             'generation_config_defaults': runtime.model.generation_config.to_dict(),
                             'scope': protocol['scope']})
    baseline.save_json('manifest.json', runtime.manifest)
    errors = {name: [] for name in METHODS}
    coordinate_sums = {name: [0.0] * protocol['eval_action_dim'] for name in METHODS}
    episodes = []
    diagnostics = {'greedy_seconds': 0.0, 'pool_seconds': 0.0, 'selection_seconds': 0.0,
                   'motion_unique_sum': 0, 'motion_collapsed_pools': 0,
                   'motion_cs_medoid_same_index': 0, 'valid_edge_token_count': 0,
                   'candidate_actions': 0, 'startup_checks_passed': 0}
    rng = random.Random(protocol['seed'] + 100003)
    start = time.monotonic()

    def synchronize():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

    def save_progress(status):
        report = summarize(baseline, protocol, errors, coordinate_sums, episodes,
                           diagnostics, time.monotonic() - start, status)
        baseline.save_json('episode_errors.json', episodes)
        baseline.save_json('progress.json', report)
        return report

    print('MOTION PILOT START', json.dumps(protocol), flush=True)
    loader = runtime.get_openx_dataloader([str(runtime.flat_dir)], batch_size=1)
    with (baseline.OUT / 'candidates.jsonl').open('w', buffering=1) as log:
        for episode_idx, batch in enumerate(loader):
            if episode_idx < protocol['episode_start']:
                continue
            if episode_idx >= protocol['episode_start'] + protocol['episode_count']:
                break
            sums = {name: 0.0 for name in METHODS}
            last_count = 0
            steps = len(batch['continuous_observation'][0])
            if not steps:
                raise ValueError('Empty episode')
            for step_idx in range(steps):
                image = batch['image_observation'][0][step_idx]
                instruction = batch['text_observation'][0][step_idx]
                if image is None or not isinstance(instruction, str):
                    raise ValueError('Missing observation/instruction')
                synchronize(); before = time.monotonic()
                greedy = runtime.predict(image, instruction)
                synchronize(); greedy_seconds = time.monotonic() - before
                before = time.monotonic()
                inputs = runtime.prepare_inputs(image, instruction)
                synchronize(); preprocessing_seconds = time.monotonic() - before
                if len(errors['greedy']) < 3:
                    decoded = generate_action(runtime, inputs, False, protocol, original_pilot.decode_tokens)
                    if not np.allclose(greedy, decoded['raw_action'], atol=1e-12, rtol=0):
                        raise AssertionError('Audited decoder differs from upstream greedy')
                seed = pool_seed(protocol, episode_idx, step_idx)
                torch.manual_seed(seed)
                synchronize(); before = time.monotonic()
                candidates = [generate_action(runtime, inputs, True, protocol, original_pilot.decode_tokens)
                              for _ in range(protocol['n_candidates'])]
                synchronize(); pool_seconds = time.monotonic() - before + preprocessing_seconds
                raw = [c['raw_action'] for c in candidates]
                before = time.monotonic()
                selection = select_actions(selectors, raw, greedy, stats, rng.randrange(len(raw)))
                selection_seconds = time.monotonic() - before
                predictions = {name: runtime.convert_action(np.array(action), protocol['dataset']).tolist()
                               for name, action in selection['raw_actions'].items()}
                for name in HYBRIDS:
                    if predictions[name][0] != predictions['greedy'][0]:
                        raise AssertionError('Converted hybrid gripper differs from greedy')
                if len(errors['greedy']) < 3:
                    independent_startup_check(np, selection, raw, greedy, predictions)
                    diagnostics['startup_checks_passed'] += 1
                    baseline.save_json('startup_checks.json', {'status': 'PASS',
                        'observations_checked': diagnostics['startup_checks_passed'],
                        'checks': ['upstream_greedy_parity', 'independent_numpy_geometry', 'greedy_gripper_preserved']})
                    print('STARTUP CHECK PASS', diagnostics['startup_checks_passed'], flush=True)
                # Ground truth is first accessed AFTER action selection and conversion.
                target = np.asarray(runtime.drop_is_terminal_dim(batch['action'][0][step_idx],
                                                                 protocol['dataset'])).tolist()
                if len(target) != protocol['eval_action_dim']:
                    raise ValueError('Unexpected target dimensions')
                mse = {name: baseline.mse(prediction, target) for name, prediction in predictions.items()}
                for name, error in mse.items():
                    errors[name].append(error); sums[name] += error
                    for d, (p, t) in enumerate(zip(predictions[name], target)):
                        coordinate_sums[name][d] += (p - t) ** 2
                last = bool(batch['is_last'][0][step_idx]); last_count += int(last)
                if last != (step_idx == steps - 1):
                    raise ValueError('Episode boundary mismatch')
                motion = selection['motion_selection']
                diagnostics['greedy_seconds'] += greedy_seconds
                diagnostics['pool_seconds'] += pool_seconds
                diagnostics['selection_seconds'] += selection_seconds
                diagnostics['motion_unique_sum'] += motion['num_unique']
                diagnostics['motion_collapsed_pools'] += int(motion['num_unique'] == 1)
                diagnostics['motion_cs_medoid_same_index'] += int(motion['cs_index'] == motion['medoid_index'])
                diagnostics['candidate_actions'] += len(candidates)
                diagnostics['valid_edge_token_count'] += sum(c['clipped_token_count'] for c in candidates)
                record = {'episode': episode_idx, 'step': step_idx, 'pool_seed': seed,
                          'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                          'instruction': instruction, 'greedy_raw': greedy.tolist(),
                          'candidates': candidates, **selection, 'predictions': predictions,
                          'target': target, 'mse': mse, 'is_last': last,
                          'seconds': {'greedy': greedy_seconds, 'pool': pool_seconds,
                                      'selection': selection_seconds}}
                log.write(json.dumps(record, allow_nan=False) + '\n')
            if last_count != 1:
                raise ValueError('Incomplete episode')
            episodes.append({'episode': episode_idx, 'num_steps': steps, 'error_sums': sums})
            progress = save_progress('partial')
            print(json.dumps({'complete_episodes': len(episodes), 'num_timesteps': len(errors['greedy']),
                              'elapsed_seconds': progress['elapsed_seconds'],
                              'amse': {name: result['amse'] for name, result in progress['methods'].items()}}), flush=True)
            if len(episodes) == protocol['episode_count']:
                break
    if len(episodes) != protocol['episode_count'] or len(errors['greedy']) != protocol['expected_eval_steps']:
        save_progress('coverage_mismatch')
        raise ValueError('Coverage differs from preregistered protocol')
    result = save_progress('complete_motion_pilot')
    result['peak_gpu_bytes'] = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    baseline.save_json('result.json', result)
    runtime.manifest['status'] = 'complete_motion_pilot'
    baseline.save_json('manifest.json', runtime.manifest)
    print(json.dumps(result, indent=2), flush=True)
