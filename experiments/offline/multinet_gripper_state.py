"""Export the observed proprioceptive gripper signal of a MultiNet test split, step-aligned.

Reads the flattened TFDS steps written by multinet_targets.py (same order as MultiNet's loader and
the baseline predictions) and stores, per step, the raw gripper command and the candidate gripper
state fields, so that model states can be converted into relative commands on CPU.
Usage: python multinet_gripper_state.py FLAT_DIR DATASET OUT.json
"""
import json
import sys

FIELDS = {
    'jaco_play': [('end_effector_cartesian_pos', 6), ('joint_pos', 6), ('joint_pos', 7)],
    'berkeley_autolab_ur5': [('robot_state', i) for i in range(15)],
}


def main():
    flat_dir, dataset, out = sys.argv[1:4]
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    steps = tf.data.Dataset.load(flat_dir)
    command, state = [], {f'{k}[{i}]': [] for k, i in FIELDS[dataset]}
    for step in steps:
        command.append(float(tf.reshape(step['action']['gripper_closedness_action'], [-1])[0]))
        for key, index in FIELDS[dataset]:
            state[f'{key}[{index}]'].append(float(step['observation'][key][index]))
    json.dump({'dataset': dataset, 'steps': len(command), 'gripper_command': command, 'state': state},
              open(out, 'w'))
    print(dataset, len(command), 'steps')


if __name__ == '__main__':
    main()
