"""Closed-loop SIMPLER (WidowX) evaluation of OpenVLA with interchangeable action selectors.

Follows RoboMonkey's openvla-mini `run_simpler_eval.py`: environment seeds 1000, 1001, ...,
150 control steps, centre-cropped JPEG policy input, and verifier image preprocessing.
The VLA (../vla_server.py) and the RoboMonkey verifier (monkey-verifier infer_server.py)
run as separate local HTTP services. Results are appended per episode, so an
interrupted run resumes from the first missing trial.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robomonkey_compat as rmc  # noqa: E402
import robomonkey_proposals as rmp  # noqa: E402
from client import append_jsonl, completed, query_vla, sampling_plan  # noqa: E402
from selection_rules import METHODS, POOL_METHODS, choose  # noqa: E402

TASKS = ('widowx_put_eggplant_in_basket', 'widowx_stack_cube',
         'widowx_carrot_on_plate', 'widowx_spoon_on_towel')
ENV_SEED_START = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--n-initial', type=int, default=5)
    p.add_argument('--n-aug', type=int, default=32)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=150)
    p.add_argument('--seed', type=int, default=1, help='seed for proposal noise and random picks')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--norm-stats', type=Path, required=True, help='bridge_orig action stats JSON')
    p.add_argument('--vla-port', type=int, default=3200)
    p.add_argument('--verifier-port', type=int, default=3100)
    p.add_argument('--verifier-batch', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    import simpler_env

    args.out.mkdir(parents=True, exist_ok=True)
    transfer_dir = (args.out / 'transfer_images').absolute()
    transfer_dir.mkdir(exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    stats = json.loads(args.norm_stats.read_text())
    converter = rmp.TokenActionConverter(stats)
    episodes_path, steps_path = args.out / 'episodes.jsonl', args.out / 'steps.jsonl'
    done_trials = {trial for (trial,) in completed(episodes_path, ['trial'])}
    n, temperature = sampling_plan(args.method, args.n_initial)

    env = simpler_env.make(args.task, renderer_kwargs={'offscreen_only': True})
    instruction = env.get_language_instruction().lower()
    for trial in range(args.trials):
        if trial in done_trials:
            continue
        env_seed = ENV_SEED_START + trial
        rng = np.random.default_rng([args.seed, trial])
        obs, _ = env.reset(seed=env_seed)
        success, records, episode_start = False, [], time.perf_counter()
        for t in range(args.max_steps):
            image, reward_path = rmc.get_simpler_img(env, obs, transfer_dir)
            policy_path = rmc.save_policy_img(image, transfer_dir)
            policy_start = time.perf_counter()
            ids, actions, gpu_s, sample_s = query_vla(args.vla_port, instruction, policy_path, n, temperature)
            record = {'trial': trial, 't': t, 'sample_s': sample_s, 'vla_gpu_s': gpu_s, 'verify_s': 0.0}
            if args.method in POOL_METHODS:
                verify_s = []

                def verifier(candidate_ids):
                    start = time.perf_counter()
                    rewards = rmc.get_rewards(instruction, reward_path, candidate_ids,
                                              args.verifier_port, args.verifier_batch)
                    verify_s.append(time.perf_counter() - start)
                    return rewards

                select_start = time.perf_counter()
                action, info = choose(args.method, ids, actions, stats=stats, converter=converter,
                                      num_aug=args.n_aug, rng=rng, verifier=verifier)
                record['verify_s'] = sum(verify_s)
                record['select_s'] = time.perf_counter() - select_start - record['verify_s']
                record.update(info)
            else:
                action = actions[0]
            action = np.asarray(action, dtype=float)
            record['policy_s'] = time.perf_counter() - policy_start
            record['action'] = action.round(6).tolist()
            records.append(record)
            obs, _, done, _, _ = env.step(rmc.convert_maniskill(action))
            if done:
                success = True
                break
        append_jsonl(steps_path, records)
        policy_s = float(np.mean([r['policy_s'] for r in records]))
        append_jsonl(episodes_path, [{
            'task': args.task, 'method': args.method, 'seed': args.seed, 'trial': trial,
            'env_seed': env_seed, 'success': success, 'steps': len(records),
            'mean_policy_s': policy_s, 'episode_s': time.perf_counter() - episode_start}])
        print(f'{args.task} {args.method} seed={args.seed} trial={trial} success={success} '
              f'steps={len(records)} policy={policy_s * 1e3:.0f}ms', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    main()
