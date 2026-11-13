"""Build the Hugging Face release directory (`hf_release/`) from archived artifacts and rollout logs.

All tables and summary metrics are derived with the same functions that produce the numbers in the
documentation (`experiments/offline/gripper_semantics.py`, `experiments/online/summarize.py`), so
the release cannot drift from the reported results.

Layout:
    multinet_audit/<dataset>.parquet      per-step predictions, targets, aligned targets, gripper state
    multinet_audit/target_statistics/     target statistics of five datasets
    multinet_audit/summary_metrics.json   every audit number of docs/MULTINET_AUDIT.md
    libero_long_consensus/*.parquet       episode and step logs of the paired LIBERO-Long pilot
    libero_long_consensus/summary_metrics.json
    raw/                                  JSONL mirrors readable by the repository scripts
    environment/, figures/, README.md, MANIFEST.json

Usage:
    python scripts/build_hf_release.py --libero-dir RUN_DIR [--out hf_release]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/offline'), str(ROOT / 'experiments/online')]
import gripper_semantics as gs  # noqa: E402
import summarize  # noqa: E402

AUDIT = {
    'jaco_play': ROOT / 'artifacts/baseline_audit/current/output',
    'berkeley_autolab_ur5': ROOT / 'artifacts/multinet_audit/berkeley_autolab_ur5',
    'nyu_door_opening_surprising_effectiveness': ROOT / 'artifacts/multinet_audit/nyu_door_opening_surprising_effectiveness',
}
METHODS = ('greedy', 'sample', 'cs', 'cs_majgrip')
LIBERO_PROTOCOL = {
    'benchmark': 'LIBERO-10 (LIBERO-Long), 10 tasks x initial states 0-9 (100 paired episodes per method)',
    'policy': 'openvla/openvla-7b-finetuned-libero-10 @ 80970322773f81baa2e22fe495d0487b93a05cfa, BF16, greedy decoding '
              'verified token-for-token against the reference generate()',
    'preprocessing': 'OpenVLA run_libero_eval.py: 180-degree rotation, JPEG round-trip, lanczos resize to 224, '
                     'center crop 0.9; 10 settling steps; at most 520 control steps',
    'methods': {
        'greedy': 'argmax decoding, one action per step',
        'sample': 'one ancestral sample at temperature 1.0 over the full vocabulary (top_k=0); distributionally '
                  'identical to a uniformly random choice among 8 i.i.d. candidates',
        'cs': 'Consensus Selection over 8 candidates (temperature 1.0, full vocabulary, top_k=0): Gaussian KDE mode on '
              'the 7-D action normalized with the frozen libero_10 q01/q99 statistics; executes the selected candidate',
        'cs_majgrip': 'CS over the 6 motion dimensions, gripper set by majority vote over the 8 candidates',
    },
    'rendering': 'MuJoCo via OSMesa (CPU) on a compute-only GPU container; methods share the renderer',
    'infra_note': 'three sample episodes (task 2, initial states 3-5) failed on a 600 s server timeout and were re-run; '
                  'the failed attempts are listed in infra_failed_episodes.jsonl and excluded',
}


INSTALL_ROOT = re.compile(r'/[^"\s]*/rm_bench')


def copy_sanitized(source: Path, destination: Path) -> None:
    """Copy a text file, replacing absolute install roots by the documented ``${RM}`` placeholder."""
    destination.write_text(INSTALL_ROOT.sub('${RM}', source.read_text()))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def audit_table(dataset: str, rows: list[dict]) -> pd.DataFrame:
    """Per-step audit table with both gripper alignments for one dataset."""
    corrected = gs.corrected_targets(rows)
    predictions = [r['prediction'] for r in rows]
    raw_state, state, aligned_pred = [None] * len(rows), [None] * len(rows), [None] * len(rows)
    export_path = gs.STATE_DIR / f'{dataset}.json'
    if dataset in gs.OBSERVED_STATE and export_path.exists():
        export = json.loads(export_path.read_text())
        raw_state = export['state'][gs.OBSERVED_STATE[dataset]]
        state = gs.observed_states(raw_state, (min(raw_state) + max(raw_state)) / 2)
        aligned_pred = gs.state_to_command(predictions, state)
    return pd.DataFrame({
        'dataset': dataset,
        'episode_id': [r['episode'] for r in rows],
        'step_id': [r['step'] for r in rows],
        'instruction': [r['instruction'] for r in rows],
        'image_sha256': [r['image_sha256'] for r in rows],
        'raw_openvla_7d': [r['raw_prediction'] for r in rows],
        'converted_multinet': predictions,
        'original_multinet_target': [r['target'] for r in rows],
        'rel2abs_aligned_target': corrected,
        'observed_gripper_raw': raw_state,
        'observed_gripper_state': state,
        'command_aligned_pred': aligned_pred,
        'multinet_step_mse': [r['mse'] for r in rows],
    })


def audit_summary(dataset: str, directory: Path, report: dict) -> dict:
    """Headline audit numbers for one dataset, matching docs/MULTINET_AUDIT.md."""
    result = json.loads((directory / 'result.json').read_text())
    reference = json.loads((directory / 'upstream_reference_NOT_OUR_RESULT.json').read_text())
    original = report['multinet_targets']['openvla_greedy']
    fix1 = report['openvla_semantics_targets']['openvla_greedy']
    fix2 = report['state_to_command_alignment']
    per_dim = report['per_dimension_mse_multinet_targets']
    return {
        'timesteps': report['timesteps'], 'target_dim': report['target_dim'],
        'published_amse_NOT_OUR_RESULT': reference['avg_dataset_amse'],
        'reproduced_amse': result['amse'],
        'reproduced_vs_published_relative': result['amse'] / reference['avg_dataset_amse'] - 1,
        'no_change_target_share': report['no_change_target_share'],
        'gripper_share_of_amse': original['gripper_part'] / original['amse'],
        'zeros_predictor_amse_multinet_targets': report['constant_predictors']['multinet_targets']['zeros'],
        'best_constant_amse_multinet_targets': report['constant_predictors']['multinet_targets']['per_dimension_mean'],
        'fix1_targets_to_states_amse': fix1['amse'],
        'fix1_relative_change': fix1['amse'] / original['amse'] - 1,
        'best_constant_amse_aligned_targets': report['constant_predictors']['openvla_semantics_targets']['per_dimension_mean'],
        'fix2_model_to_commands_amse': fix2['amse'] if fix2 else None,
        'fix2_relative_change': fix2['amse'] / original['amse'] - 1 if fix2 else None,
        'fix2_observed_state': fix2['observed_state'] if fix2 else 'not available in observations',
        'fix2_threshold_sensitivity': fix2.get('threshold_sensitivity') if fix2 else None,
        'per_dimension_mse_openvla': per_dim['openvla_greedy'],
        'per_dimension_mse_zeros': per_dim['zeros'],
        'pose_share_of_amse': original['other_dims_part'] / original['amse'],
    }


def build_audit(out: Path) -> None:
    summaries = {}
    for dataset, directory in AUDIT.items():
        rows = sorted(read_jsonl(directory / 'predictions.jsonl'), key=lambda r: (r['episode'], r['step']))
        audit_table(dataset, rows).to_parquet(out / 'multinet_audit' / f'{dataset}.parquet', index=False)
        raw = out / 'raw/multinet_audit' / dataset
        raw.mkdir(parents=True, exist_ok=True)
        for name in ('predictions.jsonl', 'result.json', 'manifest.json', 'upstream_reference_NOT_OUR_RESULT.json'):
            shutil.copy(directory / name, raw / name)
        summaries[dataset] = audit_summary(dataset, directory, gs.report(raw / 'predictions.jsonl'))
    shutil.copytree(gs.STATE_DIR, out / 'raw/multinet_audit/gripper_state', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'artifacts/multinet_audit/targets', out / 'multinet_audit/target_statistics',
                    dirs_exist_ok=True)
    (out / 'multinet_audit/summary_metrics.json').write_text(json.dumps({
        'multinet_commit': '546a1c77f9a4cc0544c5e4b41565949778d32441',
        'model': 'openvla/openvla-7b @ 47a0ec7fc4ec123775a391911046cf33cf9ed83f, greedy',
        'sign_convention': 'MultiNet: +1 closed, -1 open; relative targets: +1 close, -1 open, 0 no change',
        'datasets': summaries}, indent=2))


def final_attempt_steps(steps: list[dict], episodes: list[dict]) -> list[dict]:
    """Keep only the steps of each episode's final attempt (the last `steps` rows per episode)."""
    expected = {(e['task_id'], e['trial']): e['steps'] for e in episodes}
    grouped = collections.defaultdict(list)
    for s in steps:
        grouped[(s['task_id'], s['trial'])].append(s)
    kept = [s for key, rows in grouped.items() for s in rows[len(rows) - expected[key]:]]
    return sorted(kept, key=lambda s: (s['task_id'], s['trial'], s['t']))


def build_libero(libero_dir: Path, out: Path) -> None:
    raw = out / 'raw/libero_long_consensus'
    episode_frames, step_frames, diversity = [], [], {}
    for method in METHODS:
        episodes = read_jsonl(libero_dir / method / 'episodes.jsonl')
        steps = final_attempt_steps(read_jsonl(libero_dir / method / 'steps.jsonl'), episodes)
        assert sum(e['steps'] for e in episodes) == len(steps), method
        write_jsonl(raw / method / 'episodes.jsonl', episodes)
        write_jsonl(raw / method / 'steps.jsonl', steps)
        copy_sanitized(libero_dir / method / 'config.json', raw / method / 'config.json')
        episode_frames.append(pd.DataFrame(episodes).assign(method=method))
        step_frames.append(pd.DataFrame(steps).assign(method=method))
        unique = [s['n_unique'] for s in steps if 'n_unique' in s]
        if unique:
            diversity[method] = {'steps': len(steps), 'mean_distinct_of_8': sum(unique) / len(unique),
                                 'all_8_identical_share': sum(s.get('fallback') == 'single_unique' for s in steps) / len(steps),
                                 'distinct_count_histogram': dict(sorted(collections.Counter(unique).items()))}
    infra = [e for e in read_jsonl(libero_dir / 'sample' / 'episodes.with_infra_errors.jsonl') if e.get('error')]
    write_jsonl(out / 'libero_long_consensus/infra_failed_episodes.jsonl', infra)
    pd.concat(episode_frames).to_parquet(out / 'libero_long_consensus/episodes.parquet', index=False)
    pd.concat(step_frames).to_parquet(out / 'libero_long_consensus/steps.parquet', index=False)
    report = summarize.summarize(raw, 'greedy')
    (out / 'libero_long_consensus/summary_metrics.json').write_text(json.dumps(
        {'protocol': LIBERO_PROTOCOL, **report, 'candidate_diversity': diversity}, indent=2))


def manifest(out: Path) -> list[dict]:
    entries = []
    for path in sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'):
        rows = None
        if path.suffix == '.parquet':
            rows = len(pd.read_parquet(path))
        elif path.suffix == '.jsonl':
            rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        entries.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'rows': rows,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (out / 'MANIFEST.json').write_text(json.dumps(entries, indent=2))
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--libero-dir', type=Path, required=True, help='LIBERO-Long run directory (one folder per method)')
    p.add_argument('--out', type=Path, default=ROOT / 'hf_release')
    args = p.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    for sub in ('multinet_audit', 'libero_long_consensus', 'figures'):
        (args.out / sub).mkdir(parents=True)
    build_audit(args.out)
    build_libero(args.libero_dir, args.out)
    shutil.copytree(ROOT / 'environment', args.out / 'environment')
    for figure in ('gripper_semantics_comparison.png', 'libero_long_pilot.png'):
        shutil.copy(ROOT / 'assets' / figure, args.out / 'figures' / figure)
    shutil.copy(ROOT / 'scripts/hf_dataset_card.md', args.out / 'README.md')
    for entry in manifest(args.out):
        rows = f"{entry['rows']:>7} rows" if entry['rows'] is not None else ' ' * 12
        print(f"{entry['bytes'] / 1e6:9.2f} MB {rows}  {entry['path']}")


if __name__ == '__main__':
    main()
