"""Export the observed proprioceptive gripper signal of a MultiNet test split, step-aligned.

Reads the flattened TFDS steps written by `multinet_targets.py` (same order as MultiNet's loader and
the baseline predictions) and stores, per step, the raw gripper command and the candidate gripper
state fields, so that predicted gripper states can be converted into relative commands on CPU
(fix 2 in `gripper_semantics.py`).

Fields: Jaco Play finger joints ``joint_pos[6]`` / ``joint_pos[7]`` and the
``end_effector_cartesian_pos[6]`` value that OpenVLA reads as gripper state; Berkeley UR5
``robot_state`` (index 13 = ``gripper_is_closed``, 14 = ``action_blocked`` per the dataset page).
NYU Door Opening has no robot state in its observations.

Usage:
    python multinet_gripper_state.py FLAT_DIR DATASET OUT.json
"""
from __future__ import annotations

import json
import sys

FIELDS: dict[str, list[tuple[str, int]]] = {
    'jaco_play': [('end_effector_cartesian_pos', 6), ('joint_pos', 6), ('joint_pos', 7)],
    'berkeley_autolab_ur5': [('robot_state', i) for i in range(15)],
}


def export(flat_dir: str, dataset: str) -> dict:
    """Read gripper commands and observed state fields from flattened TFDS steps.

    Args:
        flat_dir: Directory saved with ``tf.data.Dataset.save`` by `multinet_targets.py`.
        dataset: ``jaco_play`` or ``berkeley_autolab_ur5``.

    Returns:
        ``{'dataset', 'steps', 'gripper_command', 'state': {field: values}}``.
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    command, state = [], {f'{k}[{i}]': [] for k, i in FIELDS[dataset]}
    for step in tf.data.Dataset.load(flat_dir):
        command.append(float(tf.reshape(step['action']['gripper_closedness_action'], [-1])[0]))
        for key, index in FIELDS[dataset]:
            state[f'{key}[{index}]'].append(float(step['observation'][key][index]))
    return {'dataset': dataset, 'steps': len(command), 'gripper_command': command, 'state': state}


def main() -> None:
    flat_dir, dataset, out = sys.argv[1:4]
    result = export(flat_dir, dataset)
    with open(out, 'w') as f:
        json.dump(result, f)
    print(dataset, result['steps'], 'steps')


if __name__ == '__main__':
    main()
