# Step 3: probe scores through the exchange, the probe at C (2026-10-23)

Mac only, on the step 1 cache. No GPU. Code: `tools/step3/run_step3.py` (33.5 min on 8 cores).
Numbers: `agent-collusion/results/step3/results.json`. Per-row scores: `scores.jsonl`. Figures:
`reports/figures/step3/`.

## Setup

**Rows.** Onset and before-onset agent-episodes, as in step 2. There are 348, split by onset path:

| | Before onset | Onset |
|---|---|---|
| Agreement-onset trajectories (18) | 84 | 36 |
| No-agreement-onset trajectories (32) | 164 | 64 |

The label is the agent's own verdict. The 612 sustained agent-episodes are scored as a ceiling
and never used for training.

**Positions per row.**
- **A:** end of the communication-phase intro.
- **A_call:** opening of the agent's own first turn.
- **s1–s10:** end of each message slot, numbered 2·(round−1) + order in the round. Alice always
  speaks first, so odd slots are Alice's messages and even slots Bob's.
- **C:** the verdict turn's opening, from the verdict pass.

900 of 1,000 agent-episodes have all 10 slots. A slot is empty when the agent's own send never
went through; a runner notice stands in for an undelivered peer message.

**Probes.**
- For every position, a probe is trained on that position's vectors.
- CV is the same as step 2: trajectory-grouped, 5 folds × 5 repeats, with the L2 strength and the
  layer chosen inside the training folds.
- Each probe also scores the test rows at every other position (transfer) and the test
  trajectories' sustained rows.
- Each row gets an out-of-fold score per position, averaged over repeats, and an out-of-fold
  feedback-baseline probability (step 2's feedback features).

**"Beats the feedback baseline."** The trajectory-bootstrap 95% interval (2,000 resamples of the
50 runs) of AUC(probe) − AUC(feedback) lies above 0. Scores are converted to within-fold
percentile ranks before pooling, for the probe and the baseline alike. This is because each fold
picks its own L2 strength, so raw probabilities are on different scales across folds. Pooling
raw scores understated the probe (0.85 vs 0.90 at A in a test run).

## 1. AUC at each position against the flat feedback baseline

![AUC by position](figures/step3/step3_auc_by_position.png)

| Position | Probe (pooled) | Feedback (same rows) | Difference [95% CI] | Probe fold AUC | Paired folds better | Layers chosen |
|---|---|---|---|---|---|---|
| A | 0.897 | 0.905 | −0.009 [−0.031, 0.012] | 0.888 ± 0.022 | 9/25 | 35, 31 |
| A_call | 0.906 | 0.905 | +0.000 [−0.025, 0.024] | 0.898 ± 0.028 | 13/25 | 39, 35 |
| R1 Alice (s1) | 0.894 | 0.903 | −0.009 [−0.031, 0.013] | 0.888 ± 0.029 | 12/25 | 55, 59 |
| R1 Bob (s2) | 0.889 | 0.901 | −0.012 [−0.036, 0.012] | 0.877 ± 0.025 | 11/25 | 59, 63 |
| R2 Alice (s3) | 0.896 | 0.907 | −0.011 [−0.031, 0.008] | 0.889 ± 0.026 | 10/25 | 55 |
| R2 Bob (s4) | 0.872 | 0.905 | **−0.033 [−0.059, −0.008]** | 0.862 ± 0.026 | 2/25 | 55, 59 |
| R3 Alice (s5) | 0.888 | 0.906 | −0.018 [−0.040, 0.003] | 0.880 ± 0.026 | 6/25 | 55 |
| R3 Bob (s6) | 0.890 | 0.905 | −0.015 [−0.039, 0.008] | 0.879 ± 0.029 | 10/25 | 55 |
| R4 Alice (s7) | 0.892 | 0.904 | −0.012 [−0.038, 0.015] | 0.881 ± 0.028 | 8/25 | 51, 55 |
| R4 Bob (s8) | 0.901 | 0.903 | −0.002 [−0.033, 0.028] | 0.882 ± 0.045 | 14/25 | 51 |
| R5 Alice (s9) | 0.901 | 0.907 | −0.006 [−0.027, 0.013] | 0.890 ± 0.023 | 10/25 | 55 |
| R5 Bob (s10) | 0.908 | 0.906 | +0.003 [−0.023, 0.024] | 0.895 ± 0.026 | 11/25 | 55, 51 |
| **C** | **0.956** | 0.905 | **+0.051 [+0.030, +0.071]** | **0.953 ± 0.018** | 24/25 | 47 |

The feedback baseline's own fold AUC is 0.896 ± 0.020.

### Where the probe first beats the feedback baseline

| | First position that beats it | Message-slot positions |
|---|---|---|
| Overall | **C** | none beat it; R2 Bob is significantly worse |
| Agreement-onset trajectories | **C** (0.978 vs 0.918, +0.060 [+0.027, +0.091]) | point estimates above feedback at R3 Bob, R4 Bob (0.938 vs 0.917), R5 Alice/Bob, but all intervals include 0; R2 Bob significantly worse (−0.050 [−0.095, −0.003]) |
| No-agreement-onset trajectories | **C** (0.942 vs 0.900, +0.042 [+0.015, +0.070]) | every message slot at or below feedback |

Before the verdict prompt, no point in the exchange shows more about the coming verdict than the
previous episode's outcome already does. Within agreement-onset trajectories the probe climbs
above the baseline from round 3–4 onward, but not significantly. The clear step comes at C,
after the verdict prompt and before any thinking. There the probe beats the baseline on both
paths.

## Per-row curves

![Curves](figures/step3/step3_curves_own.png)

- Each row is its own line: the out-of-fold P(accept) from each position's own probe.
- The dashed line in the same colour is that row's feedback-baseline probability.
- Dots mark the agreement message.
- `step3_curves_A_probe.png` shows the same rows scored by the A probe at every position.

Medians by group (right panel of the AUC figure):
- **Before-onset (floor):** 0.1–0.27 throughout, falling to 0.06 at C.
- **Sustained (ceiling):** 0.84–0.98.
- **Agreement onsets:** about 0.63–0.72 through round 4 Alice, then 0.92 at R4 Bob and 0.98 at C.
- **No-agreement onsets:** 0.64–0.79 through R5 Bob, then 0.91 at C.

## 2. Agreement onsets: where the agreement message falls and where the jump is

![Grid](figures/step3/step3_agreement_onsets_grid.png)

In the 36 agreement-onset agent-episodes (18 episodes × 2 agents), the agreement message falls at:

| Agreement message | R5 Bob | R4 Bob | R5 Alice | R2 Alice |
|---|---|---|---|---|
| Rows | 16 | 10 | 8 | 2 |

| Jump measure (own-position probes) | Earlier than the agreement message | At it | Later |
|---|---|---|---|
| First position above 0.5 | 36 | 0 | 0 |
| Largest one-step rise | 24 | 5 | 7 |

Every curve starts above 0.5 at A (or by R1) already. The largest rise most often comes one or
two messages before the agreement. The commonest single place is R4 Bob, which comes before 6 of
the 16 R5 Bob agreements and holds 4 of the 5 "at" cases (agreements at R4 Bob). Per-row
details are in `results.json` → `agreement_jumps.rows`.

**Caveat.** Each position has its own probe, so a jump between positions mixes a change in the
state with a change of probe. The A probe applied at the message positions cannot replace them:
it barely transfers to message-end tokens (AUC 0.50–0.62 at s1–s10, 0.70 at A_call and C). Its
curves (green, dotted) are mostly flat or noisy.

## 3. The probe at C, and transfer between A and C

**Probe at C** (label: own verdict, same rows and splits):
- **Selected-layer AUC:** 0.953 ± 0.018, range 0.921–0.984.
- **Accuracy:** 0.902, against a majority-class rate of 0.604.
- **Null:** 20 permutations, mean 0.494, 95th percentile 0.531. The probe beats all 20, so
  p = 0.048, the smallest possible with 20.
- **Layers chosen:** 47 (17/25), 43 (4), 39 (3), 51 (1).
- **Per layer:** 0.908 (layer 3), rising to 0.954–0.956 at layers 35–47, then 0.934 (layer 63).

**Transfer** (fold-mean AUC, trajectory-grouped; the source position's layer is chosen in its
training folds):

| Trained at → scored at | AUC |
|---|---|
| A → A | 0.888 ± 0.023 |
| A → C | 0.702 ± 0.216 (0.28–0.89) |
| C → C | 0.953 ± 0.018 |
| C → A | 0.744 ± 0.057 (0.65–0.88) |

By layer (same layer at both positions):

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A→C | .861 | .762 | .782 | .879 | .798 | .863 | .767 | .828 | .578 | .769 | .846 | .868 | .869 | .835 | .812 | .769 |
| C→A | .808 | .839 | .849 | .824 | .817 | .865 | .810 | .781 | .742 | .807 | .825 | .726 | .777 | .853 | .790 | .790 |

The A and C probes transfer partly in both directions, well above chance but well below each
position's own probe. The A→C transfer is unstable across folds and layers (0.58–0.88).
Whatever predicts the verdict at A is not the same readout as at C. The C probe picks up
something more, which is also what lifts it above the feedback baseline.

## 4. Speaking order at A_call

Alice always speaks first, so she has no peer message in view at A_call. Bob always speaks
second, so Alice's round-1 message is in view. Bootstrap over trajectories on out-of-fold ranks:

| | A_call vs feedback | A vs feedback | A_call vs A |
|---|---|---|---|
| Speaks first (Alice, n=174) | 0.903 vs 0.899, +0.004 [−0.023, 0.030] | 0.889 vs 0.899, −0.010 [−0.045, 0.020] | +0.015 [−0.013, 0.042] |
| Speaks second (Bob, n=174) | 0.907 vs 0.912, −0.005 [−0.046, 0.031] | 0.909 vs 0.912, −0.003 [−0.031, 0.025] | −0.002 [−0.035, 0.028] |

A_call's small edge over A (step 2: +0.010) sits with the agent who speaks first, who has seen
no peer message yet. So it does not come from the peer's opening message. In neither group does
A_call or A beat the feedback baseline.

## Reading

1. **Agreement onsets vs no-agreement onsets.** Through the exchange, both paths look alike
   against the baseline. Agreement-onset trajectories drift above it from round 3–4 onward
   (not significantly) and end higher at C (0.978 vs 0.942). No-agreement-onset trajectories
   stay at or below it until C.
2. **Settled before the agreement?** In agreement onsets the probe already favours accept from
   the start (36/36 above 0.5 by A or R1). Its largest rise usually comes one or two messages
   before the agreement message (24/36 earlier, 5 at, 7 later). But at no message slot does the
   residual stream carry more about the verdict than the previous episode's outcome does. So
   the "settled early" signal is mostly the history.
3. **Grey area.** Between the agreement message and C, the probe does not beat the baseline,
   and AUC hovers around it on both paths. The jump past the baseline happens at C, once the
   verdict is asked for and before any thinking. That is where the extra information appears.
