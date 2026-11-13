"""Figure for the MultiNet gripper-semantics audit, computed from archived predictions.

Per dataset: all-zeros predictor and OpenVLA on MultiNet's targets, and OpenVLA on targets in
OpenVLA's gripper semantics; OpenVLA bars are split into gripper and other-dimension error, and
the best constant predictor on the aligned targets is marked. Requires matplotlib.
Usage: python plot_gripper_audit.py [--out docs/figures/multinet_gripper_audit.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from gripper_semantics import DEFAULT_FILES, ROOT, report  # noqa: E402

DATASETS = ['Jaco Play', 'Berkeley UR5', 'NYU Door Opening']
GRIPPER, OTHER, ZEROS, ALIGNED = '#c0392b', '#7f8c8d', '#95a5a6', '#2471a3'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'docs/figures/multinet_gripper_audit.png')
    args = p.parse_args()
    reports = [report(f) for f in DEFAULT_FILES[:3]]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    width = 0.26
    for i, (name, r) in enumerate(zip(DATASETS, reports)):
        original = r['multinet_targets']['openvla_greedy']
        aligned = r['openvla_semantics_targets']['openvla_greedy']
        zeros = r['constant_predictors']['multinet_targets']['zeros']
        best_constant = r['constant_predictors']['openvla_semantics_targets']['per_dimension_mean']
        x0, x1, x2 = i - width, i, i + width

        ax.bar(x0, zeros, width, color=ZEROS, edgecolor='black', linewidth=0.5)
        ax.text(x0, zeros + 0.004, f'{zeros:.3f}', ha='center', va='bottom', fontsize=8)

        ax.bar(x1, original['gripper_part'], width, color=GRIPPER, edgecolor='black', linewidth=0.5)
        ax.bar(x1, original['other_dims_part'], width, bottom=original['gripper_part'],
               color=OTHER, edgecolor='black', linewidth=0.5)
        share = original['gripper_part'] / original['amse']
        ax.text(x1, original['amse'] + 0.004, f'{original["amse"]:.3f}\n{share:.1%} gripper',
                ha='center', va='bottom', fontsize=8)

        ax.bar(x2, aligned['amse'], width, color=ALIGNED, edgecolor='black', linewidth=0.5)
        drop = 1 - aligned['amse'] / original['amse']
        ax.text(x2, aligned['amse'] + 0.004, f'{aligned["amse"]:.3f}\n−{drop:.0%}',
                ha='center', va='bottom', fontsize=8)
        ax.hlines(best_constant, x2 - width / 2, x2 + width / 2, colors='black', linestyles='--', linewidth=1.2)
        ax.text(x2 + width / 2, best_constant, f' best constant\n {best_constant:.3f}', ha='left', va='center',
                fontsize=7)

    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_ylabel('AMSE (lower is better)')
    ax.set_ylim(0, 0.3)
    ax.set_title('MultiNet v0.1, OpenVLA: gripper targets are relative commands, OpenVLA predicts absolute state',
                 fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ZEROS, ec='black', lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=GRIPPER, ec='black', lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=OTHER, ec='black', lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=ALIGNED, ec='black', lw=0.5),
        plt.Line2D([0], [0], color='black', ls='--', lw=1.2),
    ]
    labels = ['All-zeros predictor (MultiNet targets)', 'OpenVLA, MultiNet targets: gripper error',
              'OpenVLA, MultiNet targets: other dimensions', 'OpenVLA, targets in OpenVLA gripper semantics',
              'Best constant (per-dimension mean), aligned targets']
    ax.legend(handles, labels, fontsize=7.5, loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.01, 'Greedy OpenVLA-7B through the MultiNet v0.1 pipeline (commit 546a1c7); '
             'published AMSE reproduced within 0.3%. Aligned: targets converted with OpenVLA rel2abs.',
             fontsize=6.5, color='#555555')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
