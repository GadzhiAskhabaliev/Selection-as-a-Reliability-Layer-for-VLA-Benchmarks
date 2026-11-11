# Online evaluation plan: Consensus Selection on LIBERO-10

Fixed on 11 November 2026, before any Stage 0 result was final. Every run listed here is
reported, including null and negative outcomes.

## Question

Does Consensus Selection (CS) over sampled OpenVLA actions raise closed-loop success on
long-horizon tasks (LIBERO-10 / LIBERO-Long), where the offline MultiNet metric could not
measure an effect? If single-action selection does not help, does chunk-level selection?

## Fixed protocol

- Policy: `openvla/openvla-7b-finetuned-libero-10`, unnorm key `libero_10`, served by
  `experiments/online/vla_server.py` (greedy decoding verified token-for-token against
  the reference `generate`).
- Environment: official LIBERO-10 initial states, OpenVLA eval preprocessing
  (`experiments/online/libero/run_eval.py`), 520 control steps, 10 settling steps.
- Rendering: MuJoCo through OSMesa (CPU) on the shared cluster. Absolute success can
  differ from published GPU-rendered numbers; comparisons between methods are paired and
  use the same renderer.
- Primary endpoint: paired success difference against greedy on identical
  (task, initial state) pairs; exact McNemar test; episode bootstrap 95% interval.
- Secondary: per-task success, policy latency, selector latency.

## Data splits (initial states per task, benchmark order)

| Split | States | Episodes per method | Use |
| --- | --- | ---: | --- |
| S0 | 0-9 | 100 | Stage 0 pilot, exploratory |
| S1 | 10-19 | 100 | Stage 1 ablations and configuration choice |
| S2 | 20-49 | 300 | Stage 2 confirmation; untouched until the configuration is frozen |

## Stage 0 - pilot (running on S0)

Greedy, CS-8 (temperature 1, full vocabulary), CS-8 with majority-vote gripper.
Stage 0 is exploratory: whatever it shows, the confirmed claim comes from S2.

Result (100 paired episodes each): greedy 55.0%, CS-8 55.0% (20 wins / 20 losses,
McNemar p = 1.0), CS-8 with majority-vote gripper 47.0% (p = 0.30). Greedy matches the
published OpenVLA LIBERO-Long result (53.7%), which validates the pipeline.

Added after the pilot, on the same S0 states: `sample` (one temperature-1 sample, full
vocabulary, no selection). The AAMAS comparison used a single sample as the base, not greedy
decoding; this control separates the effect of selection from the noise of sampling.

## Stage 1 - one-factor ablations (S1)

Each run changes exactly one factor relative to the Stage 0 CS-8 configuration and is
compared with one shared greedy run on S1. Runs share one VLA server and can execute
concurrently.

| Run | Change | Hypothesis |
| --- | --- | --- |
| greedy | reference | - |
| sample | one temperature-1 sample, no selection | separates sampling noise from selection |
| cs8 | Stage 0 configuration, replicated | estimate of run-to-run variation |
| cs8-topk10 | sampling top-k 10 (as in the AAMAS paper) | less noisy candidates sharpen the mode |
| cs8-t05 | sampling temperature 0.5 | same, via a sharper sampling distribution |
| cs16 | 16 candidates | the paper reports CS-16 > CS-8 on LIBERO-Long |
| cs8-greedygrip | CS over motion, gripper from greedy | tests the offline gripper diagnosis online |

Gripper handling across variants: `cs8`, `cs8-topk10`, `cs8-t05` and `cs16` execute the selected
candidate unchanged, so the gripper command stays coupled to the chosen motion (the paper's
argument against mixing candidates). Stage 0 `cs_majgrip` (majority vote) and `cs8-greedygrip`
are the decoupled alternatives.

After Stage 1, at most one combined configuration (best setting of each factor) is run on
S1 to check for interactions, then frozen.

## Stage 2 - confirmation (S2)

Greedy versus the frozen CS configuration, 300 paired episodes each. This is the only
result presented as a confirmed effect.

## Stage 3 - chunk-level selection

If Stage 1 shows no reliable single-action effect (and as the main extension otherwise):
a stochastic action-chunk policy, pi0-FAST (openpi `pi0_fast_libero`), with CS on the
normalized executed prefix, as in the AAMAS paper. Its LIBERO-10 baseline is re-measured
before any comparison. Same splits and endpoints.

## Not run on this cluster

The RoboMonkey head-to-head (SIMPLER, WidowX) needs a GPU with Vulkan graphics support;
the cluster container exposes compute only. Code is ready in `experiments/online/simpler/`.
