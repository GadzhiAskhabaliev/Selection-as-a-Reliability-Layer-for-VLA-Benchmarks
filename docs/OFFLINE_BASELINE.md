# Offline baseline: MultiNet v0.1 / OpenVLA on Jaco Play

Completed run: Kaggle `stepanivanov233/multinet-openvla-jaco-baseline`, version 3, status
`COMPLETE` (10 November 2026). One dataset was evaluated.

| Metric | Our run | Published reference | Difference |
| --- | ---: | ---: | ---: |
| AMSE | 0.237987188510227 | 0.23864495486236428 | -0.2756255% |
| NAMSE | 0.2276125782619673 | 0.22824167050949942 | -0.2756255% |

Coverage: **109 episodes / 7838 steps** of the official Jaco Play 0.1.0 test split. Evaluation
time: 5592.59 s (93.21 min). This is a close numerical reproduction of one dataset, not an exact
replication of the full MultiNet table. The gripper-semantics issue found later in this metric is
documented in [MULTINET_AUDIT.md](MULTINET_AUDIT.md).

## Pinned protocol

- MultiNet commit `546a1c77f9a4cc0544c5e4b41565949778d32441`.
- OpenVLA `openvla/openvla-7b`, revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
- MultiNet's own data loader, prompt, 0.9 center-crop area, action conversion and statistics.
- Greedy decoding (`do_sample=False`), seed 7, no quantization.
- 2x Tesla T4, FP16, eager attention; the authors used L4, BF16 and Flash Attention 2.
- AMSE: mean over all steps of the squared error averaged over the 4 Jaco action coordinates.
  NAMSE: mean of the per-step MSEs min-max normalized within this method.

The Jaco Play test split consists of 8 TFRecord shards; their hashes are stored in the manifest.
A coverage error produces `coverage_mismatch` instead of silently saving a partial result.

## Verification

The [raw outputs](../artifacts/baseline_audit/current/output/) contain every prediction. The
[initial audit](../artifacts/baseline_audit/verified_jaco_result.json) is kept unchanged; its
field `cs_saved_steps_matching_baseline=117` refers to the partial CS v1 run at the time of that
check. The later [CS v2 audit](../artifacts/cs_pilot/verified_v2_result.json) confirmed that all
**1487** paired greedy actions match the baseline.

```sh
python scripts/verify_baseline_results.py
```

The script has no ML dependencies. It re-checks SHA-256 hashes, coverage, the episode/step ID
sequence, the real episode boundaries, the Jaco conversion, finiteness of every action, every
per-step MSE, AMSE/NAMSE, and the comparison with the pinned reference.

The field `offline_final_action_exact_match_percent` is a diagnostic: the share of episodes whose
last predicted action equals the demonstration. It is **not** an online success rate.

## Execution history

Version 1 stopped while installing packages for Python 3.13; version 2 stopped on a missing wheel
for `promise`. Neither produced inference results. Version 3 used a separate Python 3.11
environment, built `promise`, and completed the whole test split.

The [exact v3 source](../artifacts/baseline_audit/submitted_v3/) is archived separately. The
current [runtime](../kaggle/openvla_baseline/baseline.py) factors the model and data setup out for
reuse by the CS pilot and by the re-runs on other datasets
(`experiments/offline/multinet_openvla_baseline.py`).

GPU reproduction commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); the joint
interpretation of the baseline and the CS pilot is in [OFFLINE_RESULTS.md](OFFLINE_RESULTS.md).
