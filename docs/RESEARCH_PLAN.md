# Research plan

Status as of 13 November 2026. Completed results are in the [README](../README.md),
[MULTINET_AUDIT.md](MULTINET_AUDIT.md) and [OFFLINE_RESULTS.md](OFFLINE_RESULTS.md); the
pre-registered online protocol is in [ONLINE_PLAN.md](ONLINE_PLAN.md).

## Question

When does selecting a consensus candidate help a frozen vision-language-action policy, and how
does offline action error relate to closed-loop success? Transferring CS to another model does not
by itself establish a success-rate gain.

## Done

1. CS / medoid package, normalization and tests.
2. Reproduction of the greedy MultiNet v0.1 baseline on Jaco Play (109 episodes) and re-runs of the
   MultiNet pipeline on Berkeley UR5 and NYU Door Opening, each within 0.3% of the published AMSE.
3. Paired offline CS-8 pilot on 20 Jaco episodes against single-sample, Random-8 and Medoid-8.
4. Audit of the gripper action semantics of all 20 MultiNet v0.1 datasets: a relative-vs-absolute
   mismatch on exactly the three datasets that OpenVLA converts with `rel2abs`, with two alignment
   fixes quantified.
5. Closed-loop LIBERO-Long pilot (100 paired episodes per method): greedy, single sample, CS-8 and
   CS-8 with majority-vote gripper. Greedy reproduces the published OpenVLA result.

## Next

1. **Chunk-level selection.** Single-action CS on OpenVLA does not change episode outcomes. The
   gain in the AAMAS submission was obtained by selecting among sampled action chunks; the next
   test is a stochastic chunking VLA (e.g. pi0-FAST) on LIBERO-Long, with CS on the normalized
   executed prefix and the same splits.
2. **Greedy baseline for the original OAT result.** The AAMAS comparison used a single sample as
   the base; a greedy-decoding baseline should be added there as well.
3. **Single-action ablations on OpenVLA** (stage 1 of the online plan): sampling top-k 10,
   temperature 0.5, 16 candidates, and CS on motion with the greedy gripper, on fresh initial states.
4. **Learned verifier comparison.** Head-to-head with RoboMonkey on SIMPLER (WidowX) on a GPU with
   Vulkan graphics support; code is ready in `experiments/online/simpler/`.
5. **MultiNet.** Report the gripper-semantics mismatch upstream with the reproduction scripts.

Comparisons of selection prefix lengths should keep the execution horizon fixed so that the effect
of CS is not confounded with replanning frequency.
