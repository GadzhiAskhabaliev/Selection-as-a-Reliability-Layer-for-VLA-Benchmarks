"""Selection rules compared on a shared pool of OpenVLA samples.

Every pool method starts from the same `n_initial` temperature-1 samples and the
upstream RoboMonkey preprocessing (drop rows with non-action tokens; return the
single action if all valid samples coincide). Only the final rule differs:

  robomonkey   Gaussian proposal + majority gripper, argmax of the learned verifier
  random_aug   same proposal, uniform pick (isolates the verifier's contribution)
  cs_aug       same proposal, Consensus Selection instead of the verifier
  cs           Consensus Selection over the policy samples themselves (7-D)
  cs_majgrip   CS over the 6 motion dimensions + majority gripper
  mean         proposal centre: sample mean + majority gripper
"""
import numpy as np

from consensus_selection import normalize_openvla, select

import robomonkey_proposals as rmc

POOL_METHODS = ('robomonkey', 'random_aug', 'cs_aug', 'cs', 'cs_majgrip', 'mean')
SINGLE_METHODS = ('greedy', 'sample')
METHODS = SINGLE_METHODS + POOL_METHODS


def _normalized(actions, stats):
    return np.array(normalize_openvla(np.asarray(actions, dtype=float).tolist(), stats))


def _cs_index(vectors):
    return select([[list(map(float, v))] for v in vectors])['cs_index']


def majority_gripper(actions):
    return float(np.mean(np.asarray(actions)[:, -1]) >= 0.5)


def choose(method, output_ids, actions, *, stats, converter, num_aug, rng, verifier=None):
    """Return (action, info) for one replan. `verifier(ids) -> rewards` is only used by robomonkey."""
    if method not in POOL_METHODS:
        raise ValueError(f'{method!r} is not a pool method')
    ids, acts = rmc.preprocess_actions(output_ids, actions)
    info = {'n_valid': int(len(acts))}
    if len(acts) == 0:
        info['fallback'] = 'no_valid_sample'
        return np.asarray(actions[0], dtype=float), info
    _, unique = rmc.get_unique_actions(ids, acts)
    info['n_unique'] = int(len(unique))
    if len(unique) == 1:
        info['fallback'] = 'single_unique'
        return unique[0], info

    if method in ('robomonkey', 'random_aug', 'cs_aug'):
        aug_ids, aug = rmc.generate_augmented_samples_from_batch(acts, converter, num_aug, rng)
        if method == 'cs_aug':
            index = _cs_index(_normalized(aug, stats))
            return aug[index], {**info, 'index': int(index)}
        aug_ids, aug = rmc.get_unique_actions(aug_ids, aug)
        info['n_proposals'] = int(len(aug))
        if method == 'random_aug':
            index = int(rng.integers(len(aug)))
            return aug[index], {**info, 'index': index}
        rewards = verifier(aug_ids)
        index = int(np.argmax(rewards))
        return aug[index], {**info, 'index': index, 'reward_max': float(rewards[index])}

    if method == 'cs':
        index = _cs_index(_normalized(acts, stats))
        return acts[index], {**info, 'index': int(index)}
    if method == 'cs_majgrip':
        index = _cs_index(_normalized(acts, stats)[:, :-1])
        action = acts[index].copy()
        action[-1] = majority_gripper(acts)
        return action, {**info, 'index': int(index)}
    action = acts.mean(axis=0)
    action[-1] = majority_gripper(acts)
    return action, info
