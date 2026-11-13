# Verification and reproduction

All commands run from the repository root. There are three levels: **recomputing saved results on
CPU** (no accounts, weights, TFRecords or GPUs needed), **re-running offline inference** on a GPU,
and **re-running closed-loop LIBERO evaluation** on a GPU.

## 1. CPU: verify the archive and recompute every reported number

Python 3.11 with NumPy 1.26.4 for the archive checks; the audit and summary scripts use only the
standard library.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py            # SHA-256 of the archived Kaggle outputs
python scripts/verify_baseline_results.py     # Jaco baseline, 7838 steps
python scripts/verify_cs_results.py           # offline CS pilot, 1487 steps
python experiments/offline/gripper_semantics.py   # MultiNet gripper audit (Jaco, UR5, NYU Door)
python experiments/offline/gripper_floor.py       # Jaco AMSE decomposition
```

Do not use `python -O`: some checks are assertions. These commands do not modify results;
`verify_cs_results.py --write-reports` explicitly allows regenerating derived audit JSON and is not
needed to verify the published snapshot. Expected: all unit tests pass; the SHA-256 archive check,
the baseline check and the CS check print `PASS`. GitHub Actions runs the same checks and verifies
the source hashes of the rebuildable Kaggle bundles.

The LIBERO-Long rollout logs and the per-step audit tables are published on Hugging Face (see the
README); `experiments/online/summarize.py RUN_DIR` recomputes the paired success statistics from
the episode logs.

## 2. Where the data is

See the [archive map](../artifacts/README.md). `candidates.jsonl` contains all 1487 observations of
the offline pilot with all eight candidates each, not only the selected actions, so new selection
rules can be replayed without changing the published outputs. Targets are only used to score
selections. These 20 test episodes have already been inspected and are not suitable for
independent confirmation of a new setting.

## 3. Offline inference on Kaggle (original route)

Install the Kaggle CLI and configure credentials **outside the repository**. In copies of
`kaggle/*/kernel-metadata.json`, replace `id` with your own `username/slug`; the historical IDs are
private jobs of the original account and are not needed for the CPU audit.

Verified inference environment: **2x T4**, internet enabled, no quantization. The script creates its
own Python 3.11 environment with CUDA 12.1 / PyTorch 2.2.0, Transformers 4.40.1, NumPy 1.26.4 and
TensorFlow CPU 2.15.1; full pins are in `baseline.py` and the actual freeze is archived. Downloads:
about 14 GB of weights and 1 GB of Jaco test data, none of which is in git.

```sh
kaggle kernels push -p kaggle/openvla_baseline -t 21600
python scripts/build_cs_kernel.py
kaggle kernels push -p kaggle/openvla_cs -t 21600
python scripts/check_kaggle_run.py YOUR_USER/YOUR_SLUG artifacts/new_run
```

`push` creates a new GPU job; use `status` to check on it rather than pushing again, and never write
new outputs over the published archive. `-t` is a runtime limit, not an estimate; measured
evaluation time was 93.21 minutes for the baseline and 158.19 minutes for the pilot, plus setup and
queueing.

## 4. Offline inference on any CUDA machine (other MultiNet datasets)

`experiments/offline/multinet_openvla_baseline.py` executes the archived baseline code unchanged on
another Open-X dataset, with the same pinned environment (Python 3.11, versions as in
`kaggle/openvla_baseline/baseline.py::bootstrap`):

```sh
python experiments/offline/multinet_targets.py berkeley_autolab_ur5 \
    --scratch SCRATCH --out runs/targets/ur5 --expected-steps 10156          # downloads the test split, CPU
python experiments/offline/multinet_openvla_baseline.py berkeley_autolab_ur5 \
    --expected-steps 10156 --scratch SCRATCH --out runs/baseline/ur5 --hf-home HF_CACHE
python experiments/offline/multinet_gripper_state.py SCRATCH/flat_targets_berkeley_autolab_ur5 \
    berkeley_autolab_ur5 runs/gripper_state/berkeley_autolab_ur5.json         # observed gripper state, CPU
```

The baseline reports `coverage_mismatch` for UR5, NYU Door and PushT only because MultiNet's loader
splits each of their episodes into two segments; all test steps are evaluated and the step counts
match the published ones. Peak GPU memory is about 16 GB in BF16.

## 5. Closed-loop LIBERO-Long

Environments are built in user space under `$RM` (default `~/rm_bench`) without root:

```sh
experiments/online/setup/bootstrap.sh            # pinned upstream code, checkpoints, action statistics
experiments/online/setup/setup_envs.sh vla       # OpenVLA server environment
experiments/online/setup/setup_envs.sh libero    # LIBERO client environment
experiments/online/setup/serve.sh vla_libero     # start the server (keep running)
experiments/online/setup/run_libero.sh runs/libero10 10 greedy sample cs cs_majgrip
python experiments/online/summarize.py runs/libero10
```

On machines without the NVIDIA graphics stack, run `experiments/online/setup/software_rendering.sh`
once and set `RM_SOFTWARE_RENDERING=1` (MuJoCo then renders through OSMesa). On a shared GPU,
`gpu_keeper.sh` and `supervise_libero.sh` pause the server when memory runs low and resume
stalled clients from the last completed episode. SIMPLER (`experiments/online/simpler/`) requires
a GPU with Vulkan graphics support.

## 6. What is pinned and what can differ

Model revision and MultiNet commit are in the manifests, together with TFRecord SHA-256,
normalization hashes, device maps and package versions; check them first when re-running. A newer
MultiNet `main` does not replace the pinned commit. The exact submitted v3 baseline source and the
exact CS v2 bundle are archived; `bundle_sources.json` records the hashes of the three CS source
modules and the audit checks them.

Re-running inference can depend on the GPU, dtype, attention implementation and libraries. The CPU
checks reproduce the published metrics exactly from saved actions; closeness of a new GPU run is
checked separately (our re-runs matched the published AMSE within 0.3% on three datasets), without
promising bit-exact equality on every machine.
