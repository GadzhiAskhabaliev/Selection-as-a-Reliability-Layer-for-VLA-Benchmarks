"""Paired Jaco Play pilot. All selectors share eight one-step proposals.

The bundled Kaggle entry point injects the frozen baseline runtime and selector
module. Heavy libraries are imported only by baseline.prepare_runtime().
"""
import hashlib
import json
import math
import random
import time

PROTOCOL = {
    'name': 'jaco_cs8_pilot_v1', 'dataset': 'jaco_play', 'split': 'test',
    'max_episodes': 20, 'episode_selection': 'first 20 in baseline TFDS/dataloader order',
    'n_candidates': 8, 'execution_horizon': 1, 'seed': 7,
    'temperature': 1.0, 'top_k': 10, 'top_p': 1.0,
    'sampling_policy': 'fixed exploratory setting; not tuned on validation or test',
    'methods': ['greedy', 'sample1', 'random8', 'medoid8', 'cs8'],
    'normalization': 'inverse frozen OpenVLA q01/q99 mapping; mask=False unchanged; constants zero',
    'selector_space': 'raw 7D actions before MultiNet conversion; H=R=1',
    'bandwidth': 'median positive pairwise Euclidean distance + 1e-6; fallback 1+1e-6',
    'sampling_implementation': '8 sequential single-candidate generate calls; no shared visual encoding',
    'non_action_token_policy': 'same clipping as upstream predict_action; count and save token IDs',
    'early_eos_policy': 'abort if fewer than D new action tokens; no silent reuse of prompt tokens',
    'namse': 'MultiNet per-method min-max over evaluated timesteps; pilot is not full-split reference',
    'limitations': 'single seed, exploratory partial test split, no success-rate claim or hyperparameter selection',
}


def sample_seed(episode, step):
    key = f'{PROTOCOL["seed"]}:jaco_cs8:{episode}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2 ** 31)


def decode_tokens(token_ids, vocab_size, bin_centers, stats):
    """Exactly mirror upstream bin clipping and masked q01/q99 unnormalization."""
    low, high = stats['q01'], stats['q99']
    mask = stats.get('mask', [True] * len(low))
    if not len(token_ids) == len(low) == len(high) == len(mask):
        raise ValueError('Wrong number of action tokens/statistics dimensions')
    indices = [vocab_size - int(t) - 1 for t in token_ids]
    normalized = [float(bin_centers[min(max(i, 0), len(bin_centers) - 1)]) for i in indices]
    actions = [0.5 * (a + 1) * (hi - lo) + lo if use_range else a
               for a, lo, hi, use_range in zip(normalized, low, high, mask)]
    if not all(math.isfinite(v) for v in actions):
        raise ValueError('Non-finite decoded action')
    return {'raw_action': actions, 'token_ids': [int(t) for t in token_ids],
            'clipped_token_count': sum(i < 0 or i >= len(bin_centers) for i in indices)}


def generate_action(runtime, inputs, do_sample):
    """Audit tokens without modifying model weights, logits, or upstream decoding."""
    torch, model = runtime.torch, runtime.model
    kwargs = dict(inputs)
    input_ids = kwargs.pop('input_ids')
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1)
    sampling = {'temperature': PROTOCOL['temperature'], 'top_k': PROTOCOL['top_k'],
                'top_p': PROTOCOL['top_p']} if do_sample else {}
    action_dim = model.get_action_dim('jaco_play')
    with torch.inference_mode():
        generated = model.generate(input_ids, **kwargs, max_new_tokens=action_dim,
                                   do_sample=do_sample, **sampling)
    if generated.shape[0] != 1 or generated.shape[1] - input_ids.shape[1] != action_dim:
        raise ValueError('Model did not generate exactly one full action; refusing truncated candidate')
    return decode_tokens(generated[0, -action_dim:].cpu().tolist(), model.vocab_size,
                         model.bin_centers, model.get_action_stats('jaco_play'))


def paired_summary(baseline, errors, episode_errors, diagnostics, elapsed, completed, status):
    methods = {name: baseline.aggregate(values) for name, values in errors.items()}
    greedy_amse = methods['greedy']['amse']
    for name, result in methods.items():
        result['delta_amse_vs_paired_greedy'] = result['amse'] - greedy_amse
    # Resample whole episodes, preserving all methods' pairing and timestep weights.
    # The interval does not capture variation across sampling seeds or other data.
    confidence = {}
    if status == 'complete_pilot' and completed >= 2:
        rng = random.Random(PROTOCOL['seed'] + 200003)
        boot = {name: [] for name in methods if name != 'greedy'}
        for _ in range(2000):
            chosen = rng.choices(episode_errors, k=completed)
            count = sum(ep['num_steps'] for ep in chosen)
            for name in boot:
                boot[name].append(sum(ep['error_sums'][name] - ep['error_sums']['greedy']
                                      for ep in chosen) / count)
        for name, draws in boot.items():
            draws.sort()
            confidence[name] = {'delta_amse_95pct_episode_bootstrap': [draws[49], draws[1949]],
                                'resamples': 2000, 'conditioning': 'this pilot and one sampling seed'}
    count = len(errors['greedy'])
    return {'status': status, 'complete_episodes': completed, 'num_timesteps': count,
            'elapsed_seconds': elapsed, 'methods': methods,
            'episode_bootstrap': confidence,
            'diagnostics': {**diagnostics,
                            'mean_unique_candidates': diagnostics['sum_unique_candidates'] / count,
                            'cs_medoid_index_agreement': diagnostics['cs_medoid_same_index'] / count},
            'scope': 'first 20 test episodes; compare only methods in this paired pilot',
            'full_dataset_baseline_reproduced': False}


def run(baseline, selectors):
    baseline.OUT.mkdir(parents=True, exist_ok=True)
    baseline.save_json('protocol.json', PROTOCOL)
    runtime = baseline.prepare_runtime(kind='cs_pilot')
    np, torch = runtime.np, runtime.torch
    stats = runtime.model.get_action_stats(baseline.DATASET)
    runtime.manifest.update({'status': 'evaluating_cs_pilot', 'sampling': True,
                             'protocol': PROTOCOL, 'expected_episodes': PROTOCOL['max_episodes'],
                             'expected_steps': None, 'scope': 'partial test split, paired exploratory pilot',
                             'action_statistics': stats,
                             'generation_config': runtime.model.generation_config.to_dict()})
    baseline.save_json('manifest.json', runtime.manifest)
    print('CS PILOT START', json.dumps(PROTOCOL), flush=True)
    errors = {name: [] for name in PROTOCOL['methods']}
    diagnostics = {'sum_unique_candidates': 0, 'cs_medoid_same_index': 0,
                   'cs_medoid_same_action': 0, 'collapsed_candidate_pools': 0,
                   'clipped_action_tokens': 0, 'sampled_action_tokens': 0,
                   'greedy_seconds': 0.0, 'candidate_pool_seconds': 0.0, 'selection_seconds': 0.0}
    episode_errors = []
    random_selector_rng = random.Random(PROTOCOL['seed'] + 100003)
    completed = 0
    started = time.monotonic()

    def synchronize():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

    with (baseline.OUT / 'candidates.jsonl').open('w', buffering=1) as log:
        loader = runtime.get_openx_dataloader([str(runtime.flat_dir)], batch_size=1)
        for episode_idx, batch in enumerate(loader):
            step_count = 0
            sums = {name: 0.0 for name in errors}
            last_count = 0
            for step_idx in range(len(batch['continuous_observation'][0])):
                image = batch['image_observation'][0][step_idx]
                instruction = batch['text_observation'][0][step_idx]
                if image is None or not isinstance(instruction, str):
                    raise ValueError('Missing observation; no silent skipping')
                synchronize()
                before = time.monotonic()
                greedy_raw = runtime.predict(image, instruction)
                synchronize()
                greedy_seconds = time.monotonic() - before

                before = time.monotonic()
                inputs = runtime.prepare_inputs(image, instruction)
                synchronize()
                preprocessing_seconds = time.monotonic() - before
                # Verify our audited decoder against unchanged upstream greedy before sampling.
                if episode_idx == 0 and step_idx < 3:
                    audit = generate_action(runtime, inputs, do_sample=False)
                    if not np.allclose(greedy_raw, audit['raw_action'], atol=1e-12, rtol=0):
                        raise AssertionError('Audited action decoder does not match original predict_action')
                    print('GREEDY DECODER PARITY', step_idx, 'PASS', flush=True)
                seed = sample_seed(episode_idx, step_idx)
                torch.manual_seed(seed)  # Once per pool, never reset to the same seed per candidate.
                synchronize()
                before = time.monotonic()
                candidates = [generate_action(runtime, inputs, do_sample=True)
                              for _ in range(PROTOCOL['n_candidates'])]
                synchronize()
                pool_seconds = time.monotonic() - before + preprocessing_seconds
                raw = [candidate['raw_action'] for candidate in candidates]
                before = time.monotonic()
                normalized = selectors.normalize_openvla(raw, stats)
                selection = selectors.select([[action] for action in normalized])
                indices = {'sample1': 0, 'random8': random_selector_rng.randrange(len(raw)),
                           'medoid8': selection['medoid_index'], 'cs8': selection['cs_index']}
                selection_seconds = time.monotonic() - before
                # Target is accessed only after every selector has chosen its index.
                prediction = {'greedy': runtime.convert_action(greedy_raw.copy(), baseline.DATASET).tolist()}
                prediction.update({name: runtime.convert_action(np.array(raw[i]), baseline.DATASET).tolist()
                                   for name, i in indices.items()})
                target = np.asarray(runtime.drop_is_terminal_dim(batch['action'][0][step_idx],
                                                                baseline.DATASET)).tolist()
                step_errors = {name: baseline.mse(action, target) for name, action in prediction.items()}
                for name, error in step_errors.items():
                    errors[name].append(error)
                    sums[name] += error
                last = bool(batch['is_last'][0][step_idx])
                last_count += int(last)
                step_count += 1
                diagnostics['sum_unique_candidates'] += selection['num_unique']
                diagnostics['collapsed_candidate_pools'] += int(selection['num_unique'] == 1)
                diagnostics['cs_medoid_same_index'] += int(indices['cs8'] == indices['medoid8'])
                diagnostics['cs_medoid_same_action'] += int(raw[indices['cs8']] == raw[indices['medoid8']])
                diagnostics['clipped_action_tokens'] += sum(c['clipped_token_count'] for c in candidates)
                diagnostics['sampled_action_tokens'] += sum(len(c['token_ids']) for c in candidates)
                diagnostics['greedy_seconds'] += greedy_seconds
                diagnostics['candidate_pool_seconds'] += pool_seconds
                diagnostics['selection_seconds'] += selection_seconds
                record = {'episode': episode_idx, 'step': step_idx, 'pool_seed': seed,
                          'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                          'instruction': instruction, 'target': target, 'is_last': last,
                          'greedy_raw': greedy_raw.tolist(), 'candidates': candidates,
                          'normalized_candidates': normalized, 'selection': selection,
                          'selected_indices': indices, 'predictions': prediction, 'mse': step_errors,
                          'seconds': {'greedy': greedy_seconds, 'pool': pool_seconds,
                                      'normalization_and_selectors': selection_seconds}}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if len(errors['greedy']) == 1:
                    print('FIRST CS STEP', json.dumps({'indices': indices, 'mse': step_errors,
                                                      'unique_candidates': selection['num_unique']}), flush=True)
            if not step_count or last_count != 1 or not last:
                raise ValueError('Expected one complete episode per dataloader item')
            completed += 1
            episode_errors.append({'episode': episode_idx, 'num_steps': step_count, 'error_sums': sums})
            report = paired_summary(baseline, errors, episode_errors, diagnostics,
                                    time.monotonic() - started, completed, 'partial')
            baseline.save_json('progress.json', report)
            baseline.save_json('episode_errors.json', episode_errors)
            print(json.dumps(report), flush=True)
            if completed >= PROTOCOL['max_episodes']:
                break
    if completed != PROTOCOL['max_episodes']:
        raise ValueError('Pilot did not cover the planned number of complete episodes')
    result = paired_summary(baseline, errors, episode_errors, diagnostics,
                            time.monotonic() - started, completed, 'complete_pilot')
    result['peak_gpu_bytes'] = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    baseline.save_json('result.json', result)
    runtime.manifest['status'] = 'complete_pilot'
    baseline.save_json('manifest.json', runtime.manifest)
    print(json.dumps(result, indent=2), flush=True)
