"""Figure for the paired LIBERO-Long pilot: success per method and candidate diversity of CS.

Reads a run directory with one sub-directory per method (``episodes.jsonl``, ``steps.jsonl``), as
written by `libero/run_eval.py` and published in the Hugging Face release. Success rates are shown
with 95% Wilson intervals; the right panel shows how many distinct actions the 8 sampled candidates
contain per step (token-level, after dropping invalid tokens). Requires matplotlib.

Usage:
    python plot_libero_pilot.py RUN_DIR [--out assets/libero_long_pilot.png]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METHODS = [('greedy', 'Greedy'), ('sample', 'Single sample\n(T=1)'), ('cs', 'Consensus\nSelection (N=8)'),
           ('cs_majgrip', 'CS + majority-\nvote gripper')]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', type=Path)
    p.add_argument('--out', type=Path, default=ROOT / 'assets/libero_long_pilot.png')
    args = p.parse_args()

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})
    for i, (method, label) in enumerate(METHODS):
        episodes = read_jsonl(args.run_dir / method / 'episodes.jsonl')
        k, n = sum(e['success'] for e in episodes), len(episodes)
        lo, hi = wilson(k, n)
        left.bar(i, k / n, color='#2471a3' if method == 'cs' else '#95a5a6', edgecolor='black', linewidth=0.5)
        left.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color='black', capsize=4, linewidth=1)
        left.text(i, hi + 0.02, f'{100 * k / n:.0f}%', ha='center', fontsize=9)
    left.axhline(0.537, color='#c0392b', linestyle='--', linewidth=1, label='Published OpenVLA result (53.7%)')
    left.legend(loc='upper right', fontsize=7.5, frameon=False)
    left.set_xticks(range(len(METHODS)), [label for _, label in METHODS], fontsize=8)
    left.set_ylim(0, 1)
    left.set_ylabel('Success rate (100 paired episodes)')
    left.set_title('LIBERO-Long, OpenVLA-7B fine-tuned on LIBERO-10', fontsize=10)

    steps = read_jsonl(args.run_dir / 'cs' / 'steps.jsonl')
    counts = collections.Counter(s['n_unique'] for s in steps if 'n_unique' in s)
    total = sum(counts.values())
    xs = range(1, 9)
    right.bar(xs, [counts.get(x, 0) / total for x in xs], color='#2471a3', edgecolor='black', linewidth=0.5)
    mean = sum(x * c for x, c in counts.items()) / total
    right.axvline(mean, color='black', linestyle='--', linewidth=1)
    right.text(mean + 0.1, max(counts.values()) / total * 1.02, f'mean {mean:.1f} of 8', fontsize=8)
    right.set_xticks(list(xs))
    right.set_xlabel('Distinct actions among 8 candidates (T=1.0)')
    right.set_ylabel('Share of control steps')
    right.set_title(f'Candidate diversity under CS ({total:,} steps)', fontsize=10)
    for ax in (left, right):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
