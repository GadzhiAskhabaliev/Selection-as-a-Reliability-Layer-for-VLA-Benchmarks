"""Run the archived MultiNet/OpenVLA baseline (kaggle/openvla_baseline/baseline.py) on another dataset.

The baseline module is executed unchanged; only its dataset constants and directories are set,
already-downloaded files are reused, and this process is capped to a fixed share of GPU memory
so that it cannot grow into memory used by other jobs on a shared GPU. Requires the Python 3.11
environment pinned in baseline.bootstrap().
Usage: python multinet_openvla_baseline.py DATASET --expected-steps N --scratch DIR --out DIR
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multinet_targets import load_baseline  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--expected-steps', type=int, required=True)
    p.add_argument('--version', default='0.1.0')
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hf-home', type=Path, required=True, help='existing cache with the pinned model revision')
    p.add_argument('--max-gpu-gib', type=float, default=22.0)
    args = p.parse_args()

    baseline = load_baseline(args.scratch, args.out)
    info = json.loads((args.scratch / args.dataset / args.version / 'dataset_info.json').read_text())
    lengths = next(s for s in info['splits'] if s['name'] == baseline.SPLIT)['shardLengths']
    baseline.DATASET, baseline.DATA_VERSION = args.dataset, args.version
    baseline.EXPECTED_EPISODES, baseline.EXPECTED_STEPS = sum(map(int, lengths)), args.expected_steps

    download = baseline.download
    baseline.download = lambda url, path: None if path.exists() and path.stat().st_size else download(url, path)
    root = args.scratch / 'MultiNet'
    if root.exists():
        baseline.source_tree = lambda: root
    hf = args.scratch / 'hf'
    if not hf.exists():
        hf.symlink_to(args.hf_home)
    shutil.rmtree(args.scratch / 'flat_steps_0', ignore_errors=True)
    os.environ['GDZ_BASELINE_ENV_READY'] = '1'

    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gib * 2 ** 30 / total), 0)
    baseline.main()


if __name__ == '__main__':
    main()
