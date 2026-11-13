# Experiment archive

Git contains the completed Kaggle baseline v3 and offline CS pilot v2 with raw actions,
provenance and independent audit reports, plus the MultiNet gripper-audit outputs. Every number can
be checked without access to the private Kaggle kernels. Downloaded outputs were not edited.

## Main files

| Path | Contents |
| --- | --- |
| [baseline_audit/current/output/](baseline_audit/current/output/) | All 7838 Jaco predictions, result, manifest, environment freeze, original log |
| [baseline_audit/current/status.json](baseline_audit/current/status.json) | Last saved baseline status, `COMPLETE` |
| [baseline_audit/verified_jaco_result.json](baseline_audit/verified_jaco_result.json) | Historical independent audit of the full baseline |
| [baseline_audit/submitted_v3/](baseline_audit/submitted_v3/) | Source and metadata of exactly the completed baseline v3, downloaded from Kaggle |
| [baseline_audit/source_pin.json](baseline_audit/source_pin.json) | MultiNet commit and hashes of the original files |
| [baseline_audit/jaco_dataset_info.json](baseline_audit/jaco_dataset_info.json) | Metadata of the official Jaco split |
| [cs_pilot/v2/output/](cs_pilot/v2/output/) | 1487 observations x 8 candidates, final metrics, episode totals, protocol, manifest, freeze, log |
| [cs_pilot/submitted_v2/](cs_pilot/submitted_v2/) | Exact submitted CS v2 bundle; its three embedded source modules match the current sources |
| [cs_pilot/v2/status.json](cs_pilot/v2/status.json) | Last saved CS status, `COMPLETE` |
| [cs_pilot/verified_v2_result.json](cs_pilot/verified_v2_result.json) | Independent CS audit: contrasts, per-coordinate errors, diagnostics |
| [checksums.json](checksums.json) | SHA-256 of the archived Kaggle outputs |
| [multinet_audit/](multinet_audit/) | MultiNet pipeline re-runs on Berkeley UR5 and NYU Door Opening (predictions, result, manifest), target statistics for five datasets, step-aligned observed gripper states for Jaco and UR5 |

Files named `upstream_reference_NOT_OUR_RESULT.json` contain MultiNet's **published** results and
must not be presented as new runs. The field `cs_saved_steps_matching_baseline` in the early
baseline audit is 117; the later CS v2 audit confirmed all 1487 frames. The folder name `current`
is the historical name of the completed baseline v3, not an auto-updated state.

## Data formats

### Baseline `predictions.jsonl` (also used by `multinet_audit/*/predictions.jsonl`)

One line per step: `episode`, `step` (zero-based), `image_sha256`, `instruction`,
`raw_prediction` (7-D OpenVLA output), `prediction` (after MultiNet's `convert_action`), `target`
(MultiNet target; 4-D for Jaco, 7-D for UR5 and NYU Door), `mse`, `inference_seconds`, `is_last`.
The gripper is the first coordinate of `prediction` and `target`.

### CS `candidates.jsonl`

One line per observation: the same IDs, image hash, instruction, target and boundary; `greedy_raw`;
eight `candidates` with `raw_action` and `token_ids`; `normalized_candidates`; `selection` with
bandwidth, scores and medoid costs; `selected_indices`; final `predictions`, `mse`, `seconds`;
`pool_seed`.

Raw 7-D order: translation x/y/z, rotation x/y/z, gripper. Jaco evaluation order:
`[gripper, x, y, z]`; rotation is constant in this dataset. MultiNet's Jaco gripper conversion:
`g < 0.05 -> +1`, `g > 0.95 -> -1`, otherwise `0`. Targets are stored for the evaluator and are never
passed to the selector.

`manifest.json` pins the model, revision, dataset, split, source hashes, TFRecord hashes, action
statistics, device map and environment versions. In the CS manifest `generation_config` holds the
model defaults; the actual sampling overrides are in `protocol`. `bundle_sources.json` links the run
to the local modules.

## Integrity checks

```sh
python scripts/verify_artifacts.py
python scripts/verify_baseline_results.py
python scripts/verify_cs_results.py
python experiments/offline/gripper_semantics.py
```

SHA-256 verifies the integrity of the published Kaggle archive; metrics are recomputed separately
from the actions. Weights, images/TFRecords, local PDFs and credentials are not part of this
archive. New experiments go to separate directories, never over this snapshot.
