# Gripper-semantics audit of MultiNet v0.1 (OpenVLA, 20 Open-X datasets)

Audit date: 13 November 2026. Code references are to MultiNet commit
[`546a1c7`](https://github.com/ManifoldRG/MultiNet/tree/546a1c77f9a4cc0544c5e4b41565949778d32441),
the commit behind the published v0.1 OpenVLA results; the gripper conversions are unchanged on
`main` (checked 12 November 2026). OpenVLA's data pipeline is quoted from the copy vendored in
MultiNet (`src/eval/profiling/openvla/prismatic/...`), identical to `openvla/openvla`.

## Finding

For exactly three of the 20 datasets, MultiNet compares OpenVLA's **absolute** gripper output
(open / closed) with **relative** gripper targets (change of closedness, 0 = no change). These are
precisely the three datasets whose OpenVLA/Open-X data transform converts the raw gripper action
with `rel2abs_gripper_actions`. On all three, the gripper dimension accounts for over 98% of
OpenVLA's published AMSE and an all-zeros predictor scores better than the model. Scoring the same
predictions against targets in OpenVLA's semantics lowers AMSE by 37-84%, and the model then beats
every constant predictor. The other 17 datasets are not affected by this mismatch.

## All 20 datasets

Sources: MultiNet target = `definitions/openx.py` (action-space description, line given);
`convert_action` = `src/eval/profiling/openvla/experiments/robot/multinet_openvla_utils.py`;
OpenVLA transform = `src/eval/profiling/openvla/prismatic/vla/datasets/rlds/oxe/transforms.py`.
"Mix" = in OpenVLA's training mixture `oxe_magic_soup_plus_minus`. Status: *verified* = MultiNet
pipeline re-run with OpenVLA and target statistics computed; *targets checked* = target
statistics computed, no model run; *code* = from code and descriptions only.

| Dataset | MultiNet target (gripper) | `convert_action` output (gripper) | OpenVLA `transforms.py` | Status |
| --- | --- | --- | --- | --- |
| `jaco_play` | Relative command {+1 close, -1 open, 0 no move} (L224); data: 75.7% zeros | `-discretize3(g)`, L45: +/-1 only (0 never output, 0/7838) | `rel2abs` (L173); mix | **Mismatch, verified** |
| `berkeley_cable_routing` | No gripper dimension (L230) | Reorder only, L54 | Inverted absolute (L187); mix | N/A, code |
| `nyu_door_opening_surprising_effectiveness` | "Closedness of the gripper" in [-1, 1] (L246); data: per-step change, 93.2% within +/-0.1, spikes on grasp | `-(2g-1)`, L57: continuous absolute closedness | `rel2abs` (L225); not in mix | **Mismatch, verified** |
| `viola` | Absolute {-1 open, +1 closed} (L258); data: only +/-1, open-closed-open | `-sign(2g-1)`, L61 | Clipped, inverted absolute (L242); mix | No mismatch, targets checked |
| `berkeley_autolab_ur5` | Relative command {+1 closing, -1 opening, 0 no change} (L272); data: 98.5% zeros | `-discretize3(2g-1)`, L65: +/-1 only (0/10156) | `rel2abs` (L269); mix | **Mismatch, verified** |
| `toto` | No gripper dimension (L282; boolean `open_gripper` is dropped by the float-only loader) | Reorder only, L71 | `open_gripper` cast (L283); mix | N/A, code |
| `columbia_cairlab_pusht_real` | Absolute {0 open, 1 closed} (L299); data: constant 0 (pusher) | `1 - binarize(g)`, L74 | None; not in mix | No mismatch, targets checked |
| `nyu_rot_dataset_converted_externally_to_rlds` | Absolute {0 closed, 1 open} (L311) | `binarize(g)`, L77 | Inverted absolute (L345); not in mix | No mismatch, code |
| `stanford_hydra_dataset_converted_externally_to_rlds` | Absolute {1 closed, 0 open} (L322) | `1 - binarize(g)`, L81 | Inverted absolute (L351); mix | No mismatch, code |
| `ucsd_kitchen_dataset_converted_externally_to_rlds` | Absolute {0 closed, 1 open} (L360) | `binarize(g)`, L85 | Gripper unchanged (L447); mix | No mismatch, code |
| `ucsd_pick_and_place_dataset_converted_externally_to_rlds` | Gripper torque, continuous (L372) | Raw value, L89 | Inverted, clipped (L453); not in mix | Different quantity (torque vs. state), noted in MultiNet's code comment; not this mismatch |
| `usc_cloth_sim_converted_externally_to_rlds` | Picker activation, >= 0.5 active (L390) | `1 - g`, L93 | None; not in mix | No mismatch, code |
| `utokyo_pr2_opening_fridge_converted_externally_to_rlds` | Absolute {0 closed, 1 open} (L398) | `binarize(g)`, L96 | None; not in mix | No mismatch, code |
| `utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds` | Absolute {0 closed, 1 open} (L410) | `binarize(g)`, L96 | None; not in mix | No mismatch, code |
| `utokyo_xarm_pick_and_place_converted_externally_to_rlds` | Absolute {0 closed, 1 open} (L427) | `binarize(g)`, L104 | No gripper conversion (L526); not in mix | No mismatch, code |
| `stanford_mask_vit_converted_externally_to_rlds` | Absolute {+1 closed, -1 open} (L490) | `-sign(2g-1)`, L108 | No gripper conversion (L575); not in mix | No mismatch, code |
| `eth_agent_affordances` | No gripper dimension (velocities, L529) | First 6 dimensions, L113 | None; not in mix | N/A, code |
| `imperialcollege_sawyer_wrist_cam` | Absolute {1 open, 0 closed} (L539) | `binarize(g)`, L116 | None; not in mix | No mismatch, code |
| `conq_hose_manipulation` | Absolute {0 closed, 1 open} (L592) | `binarize(g)`, L119 | None; not in mix | No mismatch, code |
| `plex_robosuite` | Absolute {+1 closed, -1 open} (L624) | `-sign(2g-1)`, L123 | None; not in mix | No mismatch, code |

`discretize3` = `discretize_range_0_to_pos_1_gripper_action_to_3_values` (L14-37): -1 below 0.05,
+1 above 0.95, 0 in between; `binarize` = `binarize_gripper_action_for_0_1_range`;
`2g-1` = `normalize_gripper_action` (`robot_utils.py` L75-86).

## Verified numbers (MultiNet pipeline re-run, greedy OpenVLA, pinned versions)

| | Jaco Play | Berkeley UR5 | NYU Door Opening |
| --- | ---: | ---: | ---: |
| Test steps (= published) | 7838 | 10156 | 2209 |
| Our AMSE vs. published | 0.23799 vs. 0.23864 (-0.28%) | 0.142428 vs. 0.142427 (+0.001%) | 0.120908 vs. 0.120946 (-0.03%) |
| Hardware | 2x T4, FP16 | H100, BF16 | H100, BF16 |
| Gripper targets "no change" (abs < 0.1) | 75.7% | 98.5% | 93.2% |
| OpenVLA gripper outputs after `convert_action` | +/-1 only | +/-1 only | continuous (not trained on this dataset) |
| Gripper share of OpenVLA's AMSE | 98.0% | 99.9% | 99.7% |
| All-zeros predictor, MultiNet targets | 0.0699 | 0.0023 | 0.0013 |
| OpenVLA AMSE, targets in OpenVLA semantics | 0.1508 (-37%) | 0.0222 (-84%) | 0.0458 (-62%) |
| Best constant, targets in OpenVLA semantics | 0.2592 | 0.1430 | 0.1431 |

"Targets in OpenVLA semantics": each episode's gripper targets converted with the same
`rel2abs_gripper_actions` rule (threshold 0.1) and MultiNet's sign (+1 closed, -1 open); all other
dimensions unchanged. This is one way to align the two semantics; converting the model's states
into commands is another and was not computed. GPT-4o and JAT cannot be re-scored: their per-step
predictions are not available, although their published Jaco AMSE (0.288, 1.237) is also above the
all-zeros predictor (0.070).

The re-runs report `coverage_mismatch` only because MultiNet's loader splits each UR5, NYU Door and
PushT episode into two segments (208/104, 98/49, 28/14 segments/episodes); every test step is
evaluated exactly once and the step counts match the published ones.

## Reproduce

```sh
# CPU, from archived predictions: decomposition and re-scoring for all three datasets
python experiments/offline/gripper_semantics.py
# Target statistics of any split (downloads the official test split; CPU)
python experiments/offline/multinet_targets.py berkeley_autolab_ur5 --scratch DIR --out DIR --expected-steps 10156
# Model re-run with the archived baseline code (GPU, Python 3.11 pinned environment)
python experiments/offline/multinet_openvla_baseline.py berkeley_autolab_ur5 --expected-steps 10156 \
    --scratch DIR --out DIR --hf-home HF_CACHE
```

Archived outputs: `artifacts/baseline_audit/` (Jaco), `artifacts/multinet_audit/` (UR5, NYU Door,
target statistics for Jaco, UR5, NYU Door, VIOLA, PushT).
