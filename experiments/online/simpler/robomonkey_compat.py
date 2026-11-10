"""RoboMonkey / openvla-mini SIMPLER image and environment helpers, vendored.

Source: github.com/robomonkey-vla/openvla-mini @ 8038e8edb0b0c590c777ba303d54b06628fdb8be
(experiments/robot/openvla_utils.py, experiments/robot/simpler/simpler_utils.py).
MIT License, Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti; (c) 2025 monkey-vla.

Vendored instead of imported because the upstream modules import the full Prismatic/torch
stack at module load, which the simulator process does not need. Image preprocessing and
verifier I/O follow upstream exactly.
"""
import math

import json_numpy as json
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from transforms3d.euler import euler2axangle


def get_rewards(instruction, image_path, output_ids, port, batch_size):
    rewards = []
    for i in range(math.ceil(len(output_ids) / batch_size)):
        payload = {'instruction': instruction, 'image_path': image_path,
                   'action': output_ids[i * batch_size:(i + 1) * batch_size]}
        response = requests.post(f'http://127.0.0.1:{port}/process', data=json.dumps(payload), timeout=600)
        response.raise_for_status()
        rewards.extend(json.loads(response.text)['rewards'])
    return rewards


def _crop_and_resize(image, crop_scale, size):
    expanded = image.shape.ndims == 3
    if expanded:
        image = tf.expand_dims(image, axis=0)
    scale = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(1,))
    offset = (1 - scale) / 2
    boxes = tf.stack([offset, offset, offset + scale, offset + scale], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(1), size)
    return image[0] if expanded else image


def _center_crop_uint8(array, size=(224, 224)):
    image = tf.image.convert_image_dtype(tf.convert_to_tensor(array), tf.float32)
    image = tf.clip_by_value(_crop_and_resize(image, 0.9, size), 0, 1)
    return tf.image.convert_image_dtype(image, tf.uint8, saturate=True).numpy()


def _jpeg_roundtrip_256(image, quality=None):
    encoded = tf.io.encode_jpeg(image) if quality is None else tf.io.encode_jpeg(image, quality=quality)
    image = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256), method='lanczos3', antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


def save_reward_img(image, transfer_dir):
    image = _jpeg_roundtrip_256(_jpeg_roundtrip_256(image), quality=95)
    path = transfer_dir / 'reward_img.jpg'
    Image.fromarray(image.numpy()).save(path)
    # Upstream `process_image`: reload, centre-crop 0.9 and resize to 224, overwrite in place.
    reloaded = np.array(Image.open(path).convert('RGB'))
    Image.fromarray(_center_crop_uint8(reloaded)).convert('RGB').save(path)
    return str(path.absolute())


def get_simpler_img(env, obs, transfer_dir, resize_size=224):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    image = get_image_from_maniskill2_obs_dict(env, obs)
    reward_path = save_reward_img(image, transfer_dir)
    image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (128, 128), method='lanczos3', antialias=True)
    image = tf.image.resize(image, (resize_size, resize_size), method='lanczos3', antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy(), reward_path


def save_policy_img(image, transfer_dir):
    """Upstream get_vla_action with center_crop=True: crop 0.9, resize 224, save JPEG for the server."""
    path = transfer_dir / 'vla_processed_img.jpg'
    Image.fromarray(_center_crop_uint8(np.array(Image.fromarray(image).convert('RGB')))).convert('RGB').save(path)
    return str(path.absolute())


def convert_maniskill(action):
    action = action.copy()
    axis, angle = euler2axangle(action[3], action[4], action[5])
    action[3:6] = axis * angle
    action[-1] = np.sign(2 * action[-1] - 1)
    return action


def dummy_action():
    return np.array([0, 0, 0, 0, 0, 0, -1])
