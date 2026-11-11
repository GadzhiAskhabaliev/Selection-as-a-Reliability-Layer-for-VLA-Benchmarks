"""Closed-loop LIBERO-10 (LIBERO-Long) evaluation of OpenVLA with interchangeable action selectors.

Follows openvla/openvla @ c8f03f48 experiments/robot/libero/run_libero_eval.py (MIT License,
Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti): fixed benchmark initial
states, env.seed(0), 10 no-op settling steps, 520 control steps, 180-degree image rotation,
JPEG round-trip + lanczos resize to 224, centre crop 0.9, binarized and inverted gripper.
The policy runs in ../vla_server.py; the image is passed losslessly (PNG).
"""
import argparse
import faulthandler
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robomonkey_proposals as rmp  # noqa: E402
from client import append_jsonl, completed, query_vla, sampling_plan  # noqa: E402
from selection_rules import POOL_METHODS, choose  # noqa: E402

METHODS = ('greedy', 'sample', 'cs', 'cs_majgrip', 'mean')
MAX_STEPS, NUM_STEPS_WAIT, DUMMY_ACTION = 520, 10, [0, 0, 0, 0, 0, 0, -1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--n-pool', type=int, default=8)
    p.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--trials', type=int, default=50, help='initial states per task, in benchmark order')
    p.add_argument('--first-trial', type=int, default=0, help='index of the first initial state')
    p.add_argument('--top-k', type=int, default=0, help='sampling top-k; 0 = full vocabulary')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--norm-stats', type=Path, required=True, help='libero_10 action stats JSON')
    p.add_argument('--vla-port', type=int, default=3201)
    return p.parse_args()


def policy_image(obs, path):
    import tensorflow as tf
    from PIL import Image
    image = obs['agentview_image'][::-1, ::-1]
    image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method='lanczos3', antialias=True)
    image = tf.image.convert_image_dtype(tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8), tf.float32)
    scale = tf.reshape(tf.sqrt(0.9), (1,))
    offset = (1 - scale) / 2
    boxes = tf.stack([offset, offset, offset + scale, offset + scale], axis=1)
    image = tf.image.crop_and_resize(image[None], boxes, tf.range(1), (224, 224))[0]
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    Image.fromarray(image.numpy()).convert('RGB').save(path)
    return str(path)


def to_env_action(action):
    action = np.asarray(action, dtype=float).copy()
    action[-1] = -np.sign(2 * action[-1] - 1)
    return action


def main():
    args = parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    args.out.mkdir(parents=True, exist_ok=True)
    image_path = (args.out / 'policy_input.png').absolute()
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    stats = json.loads(args.norm_stats.read_text())
    converter = rmp.TokenActionConverter(stats)
    episodes_path, steps_path = args.out / 'episodes.jsonl', args.out / 'steps.jsonl'
    done = completed(episodes_path, ['task_id', 'trial'])
    n, temperature = sampling_plan(args.method, args.n_pool)
    suite = benchmark.get_benchmark_dict()['libero_10']()

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(
            get_libero_path('bddl_files'), task.problem_folder, task.bddl_file), camera_heights=256, camera_widths=256)
        env.seed(0)
        for trial in range(args.first_trial, args.first_trial + args.trials):
            if (task_id, trial) in done:
                continue
            rng = np.random.default_rng([args.seed, task_id, trial])
            env.reset()
            obs = env.set_init_state(init_states[trial])
            success, error, records, episode_start = False, None, [], time.perf_counter()
            try:
                for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                    if t < NUM_STEPS_WAIT:
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                        continue
                    path = policy_image(obs, image_path)
                    policy_start = time.perf_counter()
                    ids, actions, gpu_s, sample_s = query_vla(args.vla_port, task.language, path, n, temperature,
                                                              args.top_k)
                    record = {'task_id': task_id, 'trial': trial, 't': t, 'sample_s': sample_s, 'vla_gpu_s': gpu_s}
                    if args.method in POOL_METHODS:
                        select_start = time.perf_counter()
                        action, info = choose(args.method, ids, actions, stats=stats, converter=converter,
                                              num_aug=0, rng=rng)
                        record['select_s'] = time.perf_counter() - select_start
                        record.update(info)
                    else:
                        action = actions[0]
                    record['policy_s'] = time.perf_counter() - policy_start
                    record['action'] = np.asarray(action, dtype=float).round(6).tolist()
                    records.append(record)
                    obs, _, env_done, _ = env.step(to_env_action(action).tolist())
                    if env_done:
                        success = True
                        break
            except Exception as exc:  # upstream logs and ends the episode on simulator errors
                error = repr(exc)
            append_jsonl(steps_path, records)
            policy_s = float(np.mean([r['policy_s'] for r in records])) if records else None
            append_jsonl(episodes_path, [{
                'task_id': task_id, 'task': task.language, 'method': args.method, 'seed': args.seed,
                'trial': trial, 'success': success, 'steps': len(records), 'error': error,
                'mean_policy_s': policy_s, 'episode_s': time.perf_counter() - episode_start}])
            print(f'libero_10 task={task_id} {args.method} trial={trial} success={success} '
                  f'steps={len(records)} policy={(policy_s or 0) * 1e3:.0f}ms', flush=True)
        env.close()


if __name__ == '__main__':
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    main()
