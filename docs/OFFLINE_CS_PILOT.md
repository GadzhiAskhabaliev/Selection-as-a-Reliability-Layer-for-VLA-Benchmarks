# Offline CS-8 pilot with OpenVLA (Jaco Play)

Private Kaggle kernel: `stepanivanov233/multinet-openvla-jaco-cs8-pilot`. As of 10 November 2026,
22:56 MSK, **version 2 is complete (`COMPLETE`)**: 20 full episodes, 1487 steps and 11896
candidate actions were evaluated in 158.19 minutes, and an independent check of the downloaded
outputs passed.

## Result

Lower is better. All rows refer to the same first 20 test episodes and must not be compared with
the 109-episode baseline.

| Method | AMSE | NAMSE | AMSE change vs. greedy |
| --- | --- | --- | --- |
| Greedy | 0.2315849 | 0.2270615 | 0% |
| Single stochastic sample | 0.2376200 | 0.2322724 | +2.61% |
| Random-8 | 0.2363625 | 0.2247455 | +2.06% |
| Medoid-8 | 0.2354849 | 0.2308853 | +1.68% |
| CS-8 | 0.2368162 | 0.2321905 | +2.26% |

**CS did not improve on greedy in this pilot.** The AMSE difference CS - greedy is +0.0052313,
with a 95% paired episode-bootstrap interval of [+0.0005037, +0.0114388]. This is one sampling seed
on 20 episodes, not a statement about all tasks. CS - medoid is +0.0013312 on average with an
interval of [-0.0000564, +0.0041245], which includes zero; the CS - Random and CS - single-sample
intervals also include zero.

The per-coordinate decomposition explains the sign of the overall difference: the gripper
contribution to AMSE rose by 0.0053800, while the three motion coordinates together fell by
0.0001487 (a 3.47% lower x/y/z MSE). This is an arithmetic decomposition of the metric, not proof
of a control-error mechanism. **Later finding:** the gripper term of this metric is dominated by a
target-semantics mismatch; with targets in OpenVLA's semantics, CS and greedy have identical
gripper error and CS has slightly lower motion error (AMSE 0.14939 vs. 0.14954). See
[MULTINET_AUDIT.md](MULTINET_AUDIT.md).

CS and medoid picked the same index in 86.42% of frames. On average the pool contained 2.80
distinct normalized actions out of 8, and in 38.40% of frames all candidates were identical.
Generating the pool took about 7.94x the time of greedy decoding; the selection itself is
negligible next to generation.

Random-8 has a lower NAMSE despite a higher AMSE: NAMSE normalizes errors by each method's own
range, so a lower NAMSE alone does not show lower prediction error. Online success rate is not
measured by this experiment.

Checked: coverage, episode boundaries, seeds, decoding of every token, frozen normalization,
CS/medoid scores and selection, the Random RNG, the Jaco conversion, every MSE, aggregation and the
episode bootstrap. All 1487 greedy actions and the corresponding image hashes, instructions and
targets match the completed full baseline. There are no non-action tokens; the 7573 recorded
upstream clipping events are the valid edge action token, not generation errors.

Report: `artifacts/cs_pilot/verified_v2_result.json`. Downloaded outputs:
`artifacts/cs_pilot/v2/output/`. Re-check with `python scripts/verify_cs_results.py`. Version 1 was
stopped; the final version 2 re-ran the same code from the start.

## Fixed protocol

- Data: Jaco Play 0.1.0, the first **20 complete test episodes** in the same TFDS and MultiNet
  loader order as the baseline (1487 steps).
- Model, revision, statistics, center crop, MultiNet conversion, FP16 and eager attention match the
  Kaggle baseline. Model loading and data preparation live in the shared `prepare_runtime`
  function; the baseline loop is unchanged.
- `N=8`, `H=R=1`, temperature 1.0, top-k 10, top-p 1.0, seed 7. This is a pre-chosen exploratory
  setting, not the result of validation tuning. The original OpenVLA generation supports batch
  size 1, so the calls are sequential.
- One pool of 8 stochastic actions per observation. Greedy is computed separately; single-sample is
  the first candidate, Random-8 is a uniform index from a separate RNG, and Medoid-8 and CS-8 use
  the same normalized pool.
- Normalization before selection inverts OpenVLA's frozen q01/q99 unnormalization. Coordinates
  with `mask=False` are already normalized and kept as is; constant coordinates become zero (for
  Jaco this removes the three constant rotation dimensions). Normalization is never estimated from
  the pool or from test labels.
- Selection happens in the raw 7-D action space before the MultiNet conversion; the selected
  candidate is then converted to the 4-D Jaco action for scoring. Targets reach the evaluator only
  after the index has been chosen.

CS follows equations (1), (2) and (S1) of the AAMAS 2027 submission:

```text
d_ij = Euclidean distance between flattened normalized executed prefixes
h = median(positive pairwise d_ij) + 1e-6; if none: 1 + 1e-6
score_i = sum_j exp(-d_ij^2 / (2 h^2))
selected = first index attaining the maximum score
```

Medoid minimizes the sum of Euclidean distances. Exact ties go to the first index for both. This
transfer uses `R=1`; it neither generates future trajectories nor copies the authors' source code.

## What is stored and checked

`candidates.jsonl` contains episode/step, image hash, instruction, seeds, tokens and raw actions of
every candidate, normalized actions, bandwidth, scores, selected indices, greedy, targets, per-method
errors and timings. `protocol.json`, `manifest.json`, `bundle_sources.json` and
`environment-freeze.txt` pin settings, code, data, statistics and environment. `result.json` is
written only after all 20 episodes complete.

At start-up the audited decoding is checked against the original
`model.predict_action(do_sample=False)` on three frames; the number of new tokens and out-of-range
action tokens are checked. Out-of-range tokens are clipped exactly as upstream OpenVLA does and
logged; a short sequence after an early EOS raises an explicit error.

Metrics: AMSE and the original MultiNet NAMSE per method on the same steps, the AMSE difference to
paired greedy, and a 95% episode-bootstrap interval for that difference. The bootstrap does not
reflect variation across sampling seeds.

## Running and sources

To re-run under your own Kaggle account see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build_cs_kernel.py
kaggle kernels push -p kaggle/openvla_cs -t 21600
kaggle kernels status stepanivanov233/multinet-openvla-jaco-cs8-pilot
kaggle kernels output stepanivanov233/multinet-openvla-jaco-cs8-pilot -p artifacts/cs_pilot/output
```

Sources: `src/consensus_selection/__init__.py`, `experiments/offline/cs_pilot.py`, and the shared
runtime in `kaggle/openvla_baseline/baseline.py`. `kaggle/openvla_cs/cs.py` is generated from these
files and records their SHA-256; edit the sources and rebuild the bundle rather than the bundle.

Checked locally: the CS formula and edge cases, the mask and constant coordinates, action-token
decoding, paired aggregation and bootstrap, and bundle import; the baseline evaluation loop is
AST-identical to the original. GPU integration is confirmed by the completed v2 run. At about
0.7 s per greedy call, 20 episodes with 8 candidates take 2-3 hours of compute plus setup and queue.

Generation parameters:
[Transformers 4.40.1](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/text_generation).
