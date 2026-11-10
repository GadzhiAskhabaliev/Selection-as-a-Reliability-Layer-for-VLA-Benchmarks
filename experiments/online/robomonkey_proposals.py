"""RoboMonkey action-proposal construction, vendored without behavioural changes.

Source: github.com/robomonkey-vla/openvla-mini @ 8038e8edb0b0c590c777ba303d54b06628fdb8be
(experiments/robot/openvla_utils.py, experiments/robot/token_action_converter.py).
MIT License, Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti; (c) 2025 monkey-vla.
"""
import numpy as np

BRIDGE_MIN = np.array([-0.02872725307941437, -0.04170349963009357, -0.026093858778476715,
                       -0.08092105075716972, -0.09288699507713317, -0.20718276381492615, 0.0])
BRIDGE_MAX = np.array([0.028309678435325586, 0.040855254605412394, 0.040161586627364146,
                       0.08192047759890528, 0.07792850524187081, 0.20382574498653397, 1.0])


class TokenActionConverter:
    def __init__(self, norm_stats, n_action_bins=256, vocab_size=32000):
        bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.vocab_size = vocab_size
        self.q01 = np.array(norm_stats['q01'])
        self.q99 = np.array(norm_stats['q99'])
        self.mask = np.array(norm_stats.get('mask', np.ones_like(self.q01, dtype=bool)))

    def token_to_action(self, output_ids):
        ids = np.array(output_ids)
        disc = np.clip(self.vocab_size - ids - 1, 0, self.bin_centers.shape[0] - 1)
        normalized = self.bin_centers[disc]
        return np.where(self.mask, 0.5 * (normalized + 1) * (self.q99 - self.q01) + self.q01, normalized)

    def action_to_token(self, actions):
        normalized = np.where(self.mask, 2 * (actions - self.q01) / (self.q99 - self.q01) - 1, actions)
        disc = np.array([np.abs(self.bin_centers - v).argmin() for v in normalized])
        ids = self.vocab_size - disc - 1
        return np.where(ids == 31745, 31744, ids)


def preprocess_actions(output_ids, actions):
    output_ids = np.array(output_ids)
    output_ids = np.where(output_ids == 31745, 31744, output_ids)
    actions = np.array(actions)
    keep = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    return output_ids[keep], actions[keep]


def get_unique_actions(output_ids, actions):
    output_ids, actions = np.array(output_ids), np.array(actions)
    _, idx = np.unique(output_ids, axis=0, return_index=True)
    idx = sorted(idx)
    return output_ids[idx], actions[idx]


def generate_augmented_samples_from_batch(batch_actions, converter, num_samples, rng):
    """Gaussian proposal around the initial samples; gripper fixed by majority (mean >= 0.5).

    Upstream draws from the global NumPy RNG; an explicit Generator keeps proposal
    noise reproducible per run without interacting with other random draws.
    """
    mean, var = np.mean(batch_actions, axis=0), np.var(batch_actions, axis=0)
    aug = rng.normal(mean, np.sqrt(var), size=(num_samples, 7))
    aug[:, -1] = float(mean[-1] >= 0.5)
    aug[:, :-1] = np.clip(aug[:, :-1], BRIDGE_MIN[:-1], BRIDGE_MAX[:-1])
    ids = np.stack([converter.action_to_token(a) for a in aug]).astype(np.int64)
    return ids, aug
