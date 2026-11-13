---
license: cc-by-4.0
pretty_name: "VLA test-time selection on LIBERO-Long and a gripper-semantics audit of MultiNet v0.1"
tags:
  - robotics
  - openvla
  - multinet
  - libero
  - open-x-embodiment
  - evaluation
  - vision-language-action
task_categories:
  - robotics
size_categories:
  - 100K<n<1M
configs:
  - config_name: multinet_jaco_play
    data_files: multinet_audit/jaco_play.parquet
  - config_name: multinet_berkeley_autolab_ur5
    data_files: multinet_audit/berkeley_autolab_ur5.parquet
  - config_name: multinet_nyu_door_opening
    data_files: multinet_audit/nyu_door_opening_surprising_effectiveness.parquet
  - config_name: libero_long_episodes
    data_files: libero_long_consensus/episodes.parquet
  - config_name: libero_long_steps
    data_files: libero_long_consensus/steps.parquet
---

# VLA test-time selection on LIBERO-Long and a gripper-semantics audit of MultiNet v0.1

Per-step evaluation artifacts for two studies of the OpenVLA-7B vision-language-action model.
Every number below can be recomputed on a CPU in seconds from the files in this repository; no
GPU, model weights or Open-X RLDS downloads are required.

* **MultiNet v0.1 audit.** MultiNet (commit `546a1c7`) compares OpenVLA's absolute gripper state with
  relative gripper commands on three of its 20 Open-X datasets. Re-running the MultiNet pipeline
  reproduces the published AMSE within 0.3%; the gripper accounts for 98.0-99.9% of it, an all-zeros
  predictor beats the model, and aligning the two semantics removes most of the error.
* **Consensus Selection on LIBERO-Long.** 100 paired closed-loop episodes per method with
  `openvla/openvla-7b-finetuned-libero-10`: greedy, single sample, and Consensus Selection (CS) over
  8 sampled actions.

Code, documentation and the full 20-dataset audit table:
<https://github.com/GadzhiAskhabaliev/Selection-as-a-Reliability-Layer-for-VLA-Benchmarks>
(see `docs/MULTINET_AUDIT.md`).

![OpenVLA AMSE on the three affected MultiNet datasets](figures/gripper_semantics_comparison.png)

## Key numbers

| | Jaco Play | Berkeley UR5 | NYU Door Opening |
| --- | ---: | ---: | ---: |
| Test steps | 7,838 | 10,156 | 2,209 |
| Published OpenVLA AMSE (MultiNet) | 0.2386 | 0.1424 | 0.1209 |
| Reproduced AMSE (MultiNet pipeline, greedy) | 0.2380 (-0.28%) | 0.1424 (+0.001%) | 0.1209 (-0.03%) |
| Gripper share of AMSE | 98.0% | 99.9% | 99.7% |
| All-zeros predictor, MultiNet targets | 0.0699 | 0.0023 | 0.0013 |
| Fix 1: targets converted to states (`rel2abs`) | 0.1508 (-37%) | 0.0222 (-84%) | 0.0345 (-71%) |
| MSE-optimal constant on fix-1 targets | 0.2370 | 0.1332 | 0.1381 |
| Fix 2: model states converted to commands | 0.0709 (-70%) | 0.0190 (-87%) | not possible (no robot state) |

| LIBERO-Long, 100 paired episodes | Success | vs. greedy |
| --- | ---: | --- |
| Greedy | 55% | |
| Single sample (T=1.0) | 58% | +3 pp, 95% CI [-8, +15], McNemar p = 0.74 |
| Consensus Selection, N=8 | 55% | 0 pp, 95% CI [-12, +12], p = 1.00 |
| CS + majority-vote gripper | 47% | -8 pp, 95% CI [-21, +5], p = 0.30 |

Under CS the 8 candidates contain 4.7 distinct actions on average and are all identical in 5.9% of
control steps; the selector adds 0.37 ms per step to a policy call of about 580 ms.

![LIBERO-Long pilot](figures/libero_long_pilot.png)

## Verify the audit in five seconds

```python
from datasets import load_dataset

base = "hf://hackhackhack66666/vla-test-time-and-multinet-audit/multinet_audit"

def amse(pred, target):  # MultiNet AMSE: mean over steps of the per-step mean squared error
    return sum(sum((p - t) ** 2 for p, t in zip(a, b)) / len(b) for a, b in zip(pred, target)) / len(target)

for name in ["jaco_play", "berkeley_autolab_ur5", "nyu_door_opening_surprising_effectiveness"]:
    d = load_dataset("parquet", data_files=f"{base}/{name}.parquet", split="train")
    published = amse(d["converted_multinet"], d["original_multinet_target"])
    fix1 = amse(d["converted_multinet"], d["rel2abs_aligned_target"])
    line = f"{name}: {published:.4f} -> fix 1 {fix1:.4f} ({fix1 / published - 1:+.0%})"
    if d[0]["command_aligned_pred"] is not None:
        fix2 = amse(d["command_aligned_pred"], d["original_multinet_target"])
        line += f", fix 2 {fix2:.4f} ({fix2 / published - 1:+.0%})"
    print(line)
```

Expected output:

```text
jaco_play: 0.2380 -> fix 1 0.1508 (-37%), fix 2 0.0709 (-70%)
berkeley_autolab_ur5: 0.1424 -> fix 1 0.0222 (-84%), fix 2 0.0190 (-87%)
nyu_door_opening_surprising_effectiveness: 0.1209 -> fix 1 0.0345 (-71%)
```

## Files

| Path | Rows | Contents |
| --- | ---: | --- |
| `multinet_audit/jaco_play.parquet` | 7,838 | Per-step audit table (schema below) |
| `multinet_audit/berkeley_autolab_ur5.parquet` | 10,156 | Per-step audit table |
| `multinet_audit/nyu_door_opening_surprising_effectiveness.parquet` | 2,209 | Per-step audit table |
| `multinet_audit/summary_metrics.json` | | All audit numbers, per-dimension errors, threshold sensitivity |
| `multinet_audit/target_statistics/*.json` | | Target statistics for Jaco, UR5, NYU Door, VIOLA and Columbia PushT |
| `libero_long_consensus/episodes.parquet` | 400 | One row per episode and method |
| `libero_long_consensus/steps.parquet` | 155,532 | One row per control step and method |
| `libero_long_consensus/summary_metrics.json` | | Protocol, success rates, paired contrasts, candidate diversity |
| `libero_long_consensus/infra_failed_episodes.jsonl` | 3 | Episodes lost to a server timeout and re-run (excluded) |
| `raw/` | | JSONL mirrors in the layout read by the repository scripts |
| `environment/` | | Pinned package lists per environment and upstream revisions (`pins.json`) |
| `figures/` | | 300 DPI figures |
| `MANIFEST.json` | | Size, row count and SHA-256 of every file |

### `multinet_audit/<dataset>.parquet`

| Column | Type | Description |
| --- | --- | --- |
| `dataset` | string | MultiNet dataset name |
| `episode_id` | int64 | Source episode (TFDS record) index |
| `step_id` | int64 | Step index within the source episode, from 0 |
| `loader_episode_id` | int64 | Episode index in MultiNet's loader, which splits the last step of every UR5 and NYU Door record into a separate episode (duplicated `is_last` markers) |
| `instruction` | string | Language instruction |
| `image_sha256` | string | SHA-256 of the input image |
| `raw_openvla_7d` | list[float64], 7 | OpenVLA output after un-normalization: x, y, z, roll, pitch, yaw, gripper (OpenVLA convention: 1 open, 0 closed) |
| `converted_multinet` | list[float64], 4 or 7 | OpenVLA output after MultiNet's `convert_action`; gripper first (+1 closed, -1 open) |
| `original_multinet_target` | list[float64], 4 or 7 | MultiNet target; gripper first as a relative command (+1 close, -1 open, 0 no move; continuous change for NYU Door) |
| `rel2abs_aligned_target` | list[float64], 4 or 7 | Fix 1: target with the gripper converted per episode by OpenVLA's `rel2abs_gripper_actions` (+1 closed, -1 open) |
| `observed_gripper_raw` | float64 or null | Observed gripper signal: Jaco finger joint `joint_pos[6]`, UR5 `robot_state[13]` (`gripper_is_closed`); null for NYU Door |
| `observed_gripper_state` | int64 or null | Observed gripper state (+1 closed, -1 open); Jaco binarized at the midpoint of the joint range (0.741) |
| `command_aligned_pred` | list[float64] or null | Fix 2: prediction with the gripper converted to `clip(predicted - observed, -1, 1)` |
| `multinet_step_mse` | float64 | MultiNet per-step MSE as computed by the pipeline |

Dimension order: Jaco Play `[gripper, x, y, z]`; UR5 and NYU Door `[gripper, roll, pitch, yaw, x, y, z]`.

### `libero_long_consensus/episodes.parquet`

`method` (`greedy`, `sample`, `cs`, `cs_majgrip`), `task_id` (0-9), `task` (instruction), `trial`
(LIBERO initial-state index 0-9), `success` (bool), `steps` (policy steps), `mean_policy_s`,
`episode_s`, `seed`, `error` (null for all included episodes). Episodes with the same `task_id` and
`trial` share the initial state across methods.

### `libero_long_consensus/steps.parquet`

`method`, `task_id`, `trial`, `t` (environment step, including 10 settling steps before the first
policy call), `sample_s` (client round trip), `vla_gpu_s` (server GPU time), `policy_s`, `select_s`
(selector time, pool methods), `n_valid` (candidates with valid action tokens), `n_unique` (distinct
candidates), `fallback` (`single_unique` when all valid candidates coincide), `index` (selected
candidate), `action` (executed 7-D action before gripper binarization).

## Protocols

**MultiNet audit.** Greedy OpenVLA-7B (`openvla/openvla-7b` @ `47a0ec7`) through MultiNet v0.1 at
commit `546a1c7`, using the archived evaluation code and MultiNet's data loader, prompt, center
crop and `convert_action`. Jaco: 2x T4, FP16; UR5 and NYU Door: H100, BF16. The relative semantics
of the targets is documented by the dataset authors; for UR5 the TFDS feature description reads
"1 if close gripper, -1 if open gripper, 0 if no change".

**LIBERO-Long.** `openvla/openvla-7b-finetuned-libero-10` @ `8097032`, BF16; OpenVLA's LIBERO
preprocessing (rotation, JPEG round-trip, 224 px resize, center crop 0.9), 520 control steps,
10 settling steps, initial states 0-9 of each of the 10 tasks. Sampling uses temperature 1.0 over the
full vocabulary (no top-k); CS uses the frozen `libero_10` q01/q99 statistics to normalize the 7-D
action and executes the Gaussian-KDE mode among the 8 candidates. MuJoCo rendered through OSMesa;
all methods share the renderer.

## Limitations

* The audit re-scores MultiNet's OpenVLA predictions only; GPT-4o and JAT predictions are not public.
* Fix 2 for Jaco depends on binarizing a finger joint (0.090 / 0.071 / 0.102 for thresholds
  0.5 / 0.75 / 1.0); NYU Door has no proprioceptive state, so only fix 1 applies there.
* OpenVLA's pose error is not below the all-zeros predictor on every axis: on UR5 roll and on NYU Door
  (outside OpenVLA's training mixture) it is not. Pose contributes 0.05-2.0% of the AMSE.
* The LIBERO pilot has 100 episodes per method; effects below about 10 percentage points cannot be
  resolved. It evaluates single-action selection only.

## License and attribution

Released under CC BY 4.0. Targets and observations are derived from the Open X-Embodiment datasets
(Jaco Play, Berkeley UR5, NYU Door Opening, VIOLA, Columbia PushT) and the LIBERO benchmark, and
predictions from OpenVLA; please also cite and respect the licenses of those sources.

```bibtex
@article{guruprasad2024benchmarking,
  title   = {Benchmarking Vision, Language, \& Action Models on Robotic Learning Tasks},
  author  = {Guruprasad, Pranav and Sikka, Harshvardhan and Song, Jaewoo and Wang, Yangyue and Liang, Paul Pu},
  journal = {arXiv preprint arXiv:2411.05821},
  year    = {2024}
}
@article{kim2024openvla,
  title   = {OpenVLA: An Open-Source Vision-Language-Action Model},
  author  = {Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan and Lam, Grace and Sanketi, Pannag and Vuong, Quan and Kollar, Thomas and Burchfiel, Benjamin and Tedrake, Russ and Sadigh, Dorsa and Levine, Sergey and Liang, Percy and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2406.09246},
  year    = {2024}
}
@article{liu2023libero,
  title   = {LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author  = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal = {arXiv preprint arXiv:2306.03310},
  year    = {2023}
}
@article{oxe2023,
  title   = {Open X-Embodiment: Robotic Learning Datasets and RT-X Models},
  author  = {{Open X-Embodiment Collaboration}},
  journal = {arXiv preprint arXiv:2310.08864},
  year    = {2023}
}
```
