"""Shared helpers for the closed-loop evaluation drivers."""
import json
import time

import json_numpy
import numpy as np
import requests


def query_vla(port, instruction, image_path, n, temperature, top_k=0):
    """Return (token ids [n, 7], actions [n, 7], server GPU seconds, client wall seconds)."""
    start = time.perf_counter()
    response = requests.post(f'http://127.0.0.1:{port}/sample', timeout=600, data=json_numpy.dumps(
        {'instruction': instruction, 'image_path': image_path, 'n': n, 'temperature': temperature,
         'top_k': top_k}))
    response.raise_for_status()
    body = json_numpy.loads(response.text)
    return np.array(body['output_ids']), np.array(body['actions']), body['seconds'], time.perf_counter() - start


def sampling_plan(method, n_pool):
    """(number of samples, temperature) requested from the VLA for one replan."""
    if method == 'greedy':
        return 1, 0.0
    if method == 'sample':
        return 1, 1.0
    return n_pool, 1.0


def completed(path, key):
    if not path.exists():
        return set()
    return {tuple(json.loads(line)[k] for k in key) for line in path.read_text().splitlines() if line.strip()}


def append_jsonl(path, records):
    with path.open('a') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
