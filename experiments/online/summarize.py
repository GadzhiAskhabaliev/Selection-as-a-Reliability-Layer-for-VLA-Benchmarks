"""Summarize paired closed-loop runs: success per method and task, paired contrasts, latency.

Usage: python summarize.py RUN_DIR [--reference greedy] [--json OUT]
RUN_DIR contains one sub-directory per method with episodes.jsonl and steps.jsonl.
Contrasts use only episodes completed by both methods (same task and initial state).
"""
import argparse
import json
import math
import random
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def episode_key(e):
    return (e.get('task_id', e.get('task')), e['trial'])


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b (method only) and c (reference only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_delta(pairs, draws=10000, seed=0):
    rng = random.Random(seed)
    deltas = sorted(sum(rng.choice(pairs) for _ in pairs) / len(pairs) for _ in range(draws))
    return [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def summarize(run_dir, reference):
    methods = {}
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        episodes = {episode_key(e): e for e in read_jsonl(d / 'episodes.jsonl')}
        if episodes:
            methods[d.name] = {'episodes': episodes, 'steps': read_jsonl(d / 'steps.jsonl')}
    report = {'methods': {}, 'contrasts': {}}
    for name, m in methods.items():
        eps = list(m['episodes'].values())
        per_task = {}
        for e in eps:
            per_task.setdefault(str(e.get('task_id', e.get('task'))), []).append(e['success'])
        policy = [s['policy_s'] for s in m['steps']]
        select = [s['select_s'] for s in m['steps'] if 'select_s' in s]
        report['methods'][name] = {
            'episodes': len(eps), 'success_rate': sum(e['success'] for e in eps) / len(eps),
            'per_task': {k: {'n': len(v), 'success_rate': sum(v) / len(v)} for k, v in sorted(per_task.items())},
            'policy_ms_mean': 1e3 * sum(policy) / len(policy) if policy else None,
            'policy_ms_p95': 1e3 * percentile(policy, 0.95) if policy else None,
            'select_ms_mean': 1e3 * sum(select) / len(select) if select else None,
            'select_ms_p95': 1e3 * percentile(select, 0.95) if select else None,
            'errors': sum(1 for e in eps if e.get('error')),
        }
    if reference in methods:
        ref = methods[reference]['episodes']
        for name, m in methods.items():
            if name == reference:
                continue
            shared = sorted(set(ref) & set(m['episodes']), key=str)
            if not shared:
                continue
            diffs = [int(m['episodes'][k]['success']) - int(ref[k]['success']) for k in shared]
            b, c = diffs.count(1), diffs.count(-1)
            report['contrasts'][f'{name}-{reference}'] = {
                'paired_episodes': len(shared),
                'method_success': sum(m['episodes'][k]['success'] for k in shared) / len(shared),
                'reference_success': sum(ref[k]['success'] for k in shared) / len(shared),
                'delta': sum(diffs) / len(diffs), 'delta_95ci_bootstrap': bootstrap_delta(diffs),
                'wins': b, 'losses': c, 'mcnemar_exact_p': mcnemar_exact(b, c),
            }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', type=Path)
    p.add_argument('--reference', default='greedy')
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    report = summarize(args.run_dir, args.reference)
    for name, m in report['methods'].items():
        sel = f"{m['select_ms_mean']:.3f}" if m['select_ms_mean'] is not None else '-'
        print(f"{name:12s} n={m['episodes']:4d}  success={100 * m['success_rate']:5.1f}%  "
              f"policy={m['policy_ms_mean']:.0f}ms  selector={sel}ms  errors={m['errors']}")
    for name, c in report['contrasts'].items():
        lo, hi = c['delta_95ci_bootstrap']
        print(f"{name:20s} paired={c['paired_episodes']}  delta={100 * c['delta']:+.1f}pp "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  wins/losses={c['wins']}/{c['losses']}  "
              f"McNemar p={c['mcnemar_exact_p']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
