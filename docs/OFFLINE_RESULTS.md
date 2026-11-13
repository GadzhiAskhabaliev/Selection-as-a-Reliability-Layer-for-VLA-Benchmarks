# Report: OpenVLA baseline and the first offline CS-8 pilot

Experiment snapshot: **10 November 2026**. This is an offline evaluation of actions on observations
from Jaco Play demonstrations; no robot or simulator executed the selected actions.

> **Update, 13 November 2026.** The gripper term that dominates the metric below comes from a
> target-semantics mismatch in MultiNet (relative gripper commands compared with OpenVLA's absolute
> gripper state). With targets in OpenVLA's semantics the negative CS result disappears: CS and
> greedy have identical gripper error and CS has slightly lower motion error. See
> [MULTINET_AUDIT.md](MULTINET_AUDIT.md). The closed-loop evaluation on LIBERO-Long is summarized in
> the [README](../README.md). The report below is kept as written for provenance.

## Summary

The baseline is closely reproduced on Jaco Play. In the first transfer of CS to single-step
OpenVLA there is no gain over plain greedy decoding: AMSE is **2.26%** higher. This is a verified
negative result for this specific pilot, useful for diagnosis, not evidence that CS is useless in
general.

The error decomposition points to the gripper as the source of the higher overall error. This
motivates a separate experiment but does not show that changing the gripper handling would raise
control success.

## 1. What was tested

The idea of CS is to generate several actions or chunks and select the existing candidate with the
largest summed Gaussian similarity to the others. The selection rule is transferred from the OAT
action-chunking policies to the OpenVLA family.

In this pilot, standard OpenVLA outputs **one 7-D action per observation**, so `H=R=1`: eight calls
give eight alternatives for one step, not trajectories eight steps into the future. This tests the
single-step case of CS.

| Parameter | Fixed value |
| --- | --- |
| Model | `openvla/openvla-7b` |
| Model revision | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| MultiNet commit | `546a1c77f9a4cc0544c5e4b41565949778d32441` |
| Dataset | `jaco_play/0.1.0`, official `test` split |
| Full baseline | 109 episodes, 7838 steps |
| CS pilot | First 20 episodes in the same TFDS/loader order, 1487 steps |
| Sampling | `N=8`, temperature `1.0`, top-k `10`, top-p `1.0`, seed `7` |
| Hardware | 2x Tesla T4, FP16, eager attention, no quantization |
| Preprocessing | MultiNet's prompt, 0.9 center-crop area and action adapters |
| Normalization | Frozen q01/q99 and mask from the pinned MultiNet commit |

`manifest.generation_config` contains the model's default generation settings, including
`do_sample=false` and `top_k=50`; the actual stochastic settings override them at generation time
and are recorded in `protocol` and above. The sampling settings were chosen in advance for an
exploratory pilot; there was no validation tuning. The first 20 test episodes are not a random
sample of the split.

## 2. Full baseline: reproduction check

| Metric | Our baseline | Published reference | Relative difference |
| --- | ---: | ---: | ---: |
| AMSE | 0.2379871885 | 0.2386449549 | -0.2756% |
| NAMSE | 0.2276125783 | 0.2282416705 | -0.2756% |

All **109 episodes / 7838 steps** completed. Evaluation time was **93.21 minutes**, excluding
environment setup, weight download and the Kaggle queue. The reference numbers come from the pinned
[MultiNet result file](https://github.com/ManifoldRG/MultiNet/blob/546a1c77f9a4cc0544c5e4b41565949778d32441/src/eval/profiling/openvla/results/openvla_openx_eval_results.json),
not from our run, and are marked `NOT_OUR_RESULT` in the archive.

The original environment differs (L4/BF16/Flash Attention 2 vs. T4/FP16/eager). The numbers are
close, but bit-exact replication and reproduction of the whole benchmark are not claimed. The cause
of the small difference was not established.

## 3. CS on the 20 shared episodes

Greedy was computed separately on the same inputs. Single sample, Random-8, Medoid-8 and CS-8 use
**the same pool of eight candidates** for every frame. Single sample is the first candidate;
Random-8 picks a uniform random index with a separate RNG. Greedy is not part of the pool.

| Method | AMSE (lower is better) | NAMSE | AMSE difference to greedy | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.2315849059 | 0.2270614506 | 0 | 0% |
| Single sample | 0.2376199629 | 0.2322724281 | +0.0060350570 | +2.61% |
| Random-8 | 0.2363625107 | 0.2247454599 | +0.0047776048 | +2.06% |
| Medoid-8 | 0.2354849168 | 0.2308852841 | +0.0039000109 | +1.68% |
| CS-8 | 0.2368161663 | 0.2321905308 | +0.0052312604 | **+2.26%** |

**The 109-episode baseline AMSE of 0.2379872 must not be substituted into this table**: comparing
different sets of frames would falsely suggest a CS gain. All 1487 greedy predictions of the pilot
exactly match the corresponding part of the full baseline, as do the image hashes, instructions,
targets and episode boundaries.

### Uncertainty

Paired bootstrap: 2000 resamples of 20 episodes with replacement; AMSE in each resample is weighted
by the number of steps, and the methods are compared on the same resampled episodes. RNG seed
`200010`; bounds are elements 49 and 1949 (zero-based) of the 2000 sorted differences.

| Difference | AMSE difference | 95% episode-bootstrap interval |
| --- | ---: | --- |
| CS - greedy | +0.0052313 | [+0.0005037; +0.0114388] |
| CS - single sample | -0.0008038 | [-0.0062013; +0.0038895] |
| CS - Random-8 | +0.0004537 | [-0.0034655; +0.0038294] |
| CS - Medoid-8 | +0.0013312 | [-0.0000564; +0.0041245] |

In this pilot the CS - greedy interval lies above zero: CS has higher error. The other three
intervals include zero; no advantage of CS over these controls was established. The intervals are
conditional on these 20 episodes and one sampling seed; they do not account for variation across
generation repeats or task selection.

## 4. Diagnostics

After the MultiNet conversion the Jaco action is ordered `[gripper, x, y, z]`. Each column below is
a **contribution to the overall AMSE**, i.e. the corresponding sum of mean squared errors divided by 4.

| Method | Gripper contribution | x/y/z contribution | Total |
| --- | ---: | ---: | ---: |
| Greedy | 0.2273032952 | 0.0042816107 | 0.2315849059 |
| CS-8 | 0.2326832549 | 0.0041329114 | 0.2368161663 |
| CS - greedy | **+0.0053799597** | **-0.0001486993** | **+0.0052312604** |

The summed x/y/z MSE decreased by **3.47%**, but the gripper degradation outweighed it. This is a
decomposition of the obtained result; CS in this run selected over all non-constant coordinates,
not only x/y/z. The mixed scale of a discrete gripper and continuous motion affects AMSE; we keep
the published metric rather than changing it after seeing the result.

| Diagnostic | Value |
| ---: | ---: |
| Distinct normalized actions in the pool | 2.80 of 8 on average |
| All eight candidates identical | 571/1487 frames, 38.40% |
| CS and medoid chose the same index | 1285/1487 frames, 86.42% |
| Invalid action tokens | 0 |
| Pool generation time / greedy time | 7.94x |
| Total pilot evaluation time | 158.19 minutes |

All eight candidates were generated sequentially, without reusing the visual encoding. Pool
generation took 8399.95 s, separate greedy decoding 1058.47 s, and the joint CS/medoid selection
0.20 s. The 7.94x is the ratio of pool time to greedy time, not of the whole study; this is not an
equal-compute comparison. Low diversity limits what selection can do, and the high agreement with
medoid means the specific advantage of Gaussian scores is not shown.

## 5. How the metrics and selection are computed

For M frames and 4-D Jaco actions:

```text
mse_t = sum_d (prediction[t,d] - target[t,d])^2 / 4
AMSE  = mean_t(mse_t)
NAMSE = mean_t((mse_t - min_t mse_t) / (max_t mse_t - min_t mse_t))
```

The min and max in NAMSE are computed **separately for each method over all evaluated frames**,
which is why Random-8 has a lower NAMSE with a higher AMSE: the error range changed too. NAMSE is
not a common normalization of action coordinates. For a zero range the evaluator returns `null`
with `namse_defined=false`.

Selection happens **before** the conversion to 4-D Jaco actions, in the normalized 7-D space. Masked
coordinates invert the q01/q99 unnormalization; constant rotation dimensions become zero;
`mask=False` coordinates, including the gripper, are kept unchanged. Statistics are never fitted to
the pool or to test data.

```text
d_ij = Euclidean distance between normalized candidates
h = median(positive pairwise d_ij) + 1e-6, or 1 + 1e-6 if there are no positive distances
CS     = argmax_i sum_j exp(-d_ij^2 / (2h^2))
Medoid = argmin_i sum_j d_ij
```

Exact ties go to the first index. The selector sees only the candidates, never the targets, which
the evaluator uses after selection. The implementation follows the formulas of the research paper;
it was not copied from the authors' package.

## 6. Result checks

Both final runs, baseline v3 and CS pilot v2, have Kaggle status `COMPLETE`; the partial CS v1 run
is not used. The CPU audit checks coverage and boundaries, raw tokens, normalization, CS/medoid
scores, the Random RNG, the Jaco conversion, every MSE, the final metrics, the bootstrap and the
match of paired greedy to the full baseline. SHA-256 fixes the archived files. This verifies saved
results; it is not a new inference run.

## 7. What can and cannot be concluded

- A close numerical reproduction of the greedy baseline on Jaco is confirmed.
- CS with `N=8, R=1, T=1, top-k=10` did not improve on paired greedy in this pilot.
- This is one dataset, the first 20 test episodes, one sampling seed and one setting; it does not
  generalize to all of OpenVLA, CS or robotic tasks.
- Independence of the Jaco test split from OpenVLA's pre-training is not established; this
  reproduces the benchmark protocol and is not evidence of generalization.
- LIBERO success rate, multi-step planning and CS over future chunks were not measured here.
  Offline final-action exact match is not a success rate.
- The idea "CS for motion only + greedy gripper" arose after seeing these test results and is
  untested; it would first need an exploratory replay of the saved candidates and then confirmation
  on new episodes. Tuning on these 20 episodes must not be presented as a test gain.

Next steps are in the [research plan](RESEARCH_PLAN.md); verification and re-run commands are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Full-precision numbers are in the
[baseline audit JSON](../artifacts/baseline_audit/verified_jaco_result.json) and the
[CS audit JSON](../artifacts/cs_pilot/verified_v2_result.json).
