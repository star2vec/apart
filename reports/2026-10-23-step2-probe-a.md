# Step 2: probe at point A (2026-10-23)

Mac only, on the step 1 cache. No GPU. Code: `tools/step2/` (`show_positions.py`,
`probe_lib.py`, `run_step2.py`); full numbers in `agent-collusion/results/step2/results.json`.

## Positions checked before training

`tools/step2/show_positions.py` decodes the saved token and the text before it.

- **A** = the `<|im_end|>` closing the communication-phase intro, which ends "...Only act via tool
  calls. The tool for the communication phase is send_message." No peer message is in view yet.
  Next comes either "Use the send_message tool..." or, for the agent that speaks second, the
  peer's round-1 message.
- **C** = the `\n` after `<|im_start|>assistant` on the verdict turn, right after "The tool for the
  verdict phase is submit_verdict.<|im_end|>". In episodes 1–9 of the verdict pass, the token after C
  is the saved `<tool_call>` (earlier thinking is dropped by the template). In episode 10 it is the
  generation prompt's `<think>`, exactly as vLLM saw it.

## Protocol

- **Unit:** one agent in one episode. The vector is the residual stream at A in the communication pass.
- **Probe set:** onset and before-onset episodes, giving 348 rows from 50 trajectories. Label: the
  agent's own verdict (138 accept, 210 reject). Every onset row is accept by definition; 38 of
  248 before-onset rows are accept.
- **Outer CV:** GroupKFold by trajectory, so all episodes and both agents of a run fall on one
  side of each split. 5 folds × 5 shuffled repeats = 25 test folds.
- **Inner CV:** GroupKFold (4 folds) on the training rows only. It chooses the L2 strength from
  7 values between 1e-5 and 10 by pooled out-of-fold AUC.
- **Selected-layer probe:** in each outer split, the layer with the best inner AUC. Test folds are
  never used to choose anything.
- **Model:** standardised features (on the training fold) and L2 logistic regression. The fit is
  done exactly in the training rows' SVD basis; it matches a full 5,120-feature fit to 4e-7.
- **Baselines:** the same splits and the same inner choice of regularisation.
  - **episode:** one-hot episode number.
  - **relaxation:** the agent's previous reflection's relaxation flag (true, false, or none in
    episode 1).
  - **feedback:** from the previous episode, both verdicts, whether each was right and the agent's
    reward, plus episode number and a first-episode flag.
- **Null:** labels permuted across rows, with 100 permutations. Each layer uses the same
  permutations, and the whole pipeline runs again, including inner selection and layer selection.
- **Scores:** AUC; spread is the SD (and min–max) over the 25 test folds. Majority-class
  accuracy is the accuracy of predicting the training majority on each test fold.

## Positive control: previous episode's reward at A

The previous episode's reward (0, 10 or 20) appears in the prompt as "## Reward". The control is a
3-class probe scored by macro one-vs-rest AUC.

| Rows | n | Selected-layer AUC | Best layers (inner choice) | Majority acc |
|---|---|---|---|---|
| all episodes 2–10 | 900 | **0.924 ± 0.024** (0.865–0.959) | 31 (13/25), 27 (8/25) | 0.59 |
| probe set, episodes 2+ | 248 | 0.735 ± 0.071 (0.586–0.847) | spread out | 0.58 |

The probe set's control is weaker because only 8 of its 248 rows have reward 20.

Control AUC by layer, on all episodes 2–10:

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .802 | .808 | .851 | .897 | .901 | .905 | .930 | .932 | .916 | .915 | .918 | .912 | .912 | .907 | .901 | .881 |

## A probe: own verdict

- **Majority-class accuracy:** 0.604 (± 0.019).
- **Null (selected-layer pipeline):** mean 0.504, 95th percentile 0.558. The real score beats
  all 100 permutations at every layer (p = 0.0099, the smallest possible with 100).

| | AUC (mean ± SD over 25 folds) | min–max | Accuracy |
|---|---|---|---|
| **A probe, layer chosen in training folds** | **0.888 ± 0.023** | 0.836–0.923 | 0.791 |
| Baseline: feedback features | **0.896 ± 0.020** | 0.861–0.930 | 0.822 |
| Baseline: episode number | 0.871 ± 0.027 | 0.817–0.919 | 0.753 |
| Baseline: previous relaxation flag | 0.818 ± 0.031 | 0.757–0.875 | 0.639 |
| Shuffled-label null | 0.504 (95th pct 0.558) | | |

Paired over the same 25 test folds, the selected-layer probe compares with each baseline as follows:

| Baseline | Mean AUC difference | SD | Folds where the probe is better |
|---|---|---|---|
| feedback | −0.008 | 0.027 | 9/25 |
| episode | +0.017 | 0.025 | 17/25 |
| relaxation | +0.070 | 0.039 | 25/25 |

Layers chosen by the inner CV: 35 (9/25), 31 (6/25), 39 (3), 43 (3), 51 (2), 27 (1), 47 (1).

Every layer (descriptive, AUC on the test folds; not used for any choice):

| Layer | AUC | SD | min–max | Null mean / 95th pct |
|---|---|---|---|---|
| 3 | 0.868 | 0.029 | 0.798–0.923 | 0.504 / 0.572 |
| 7 | 0.868 | 0.035 | 0.786–0.934 | 0.502 / 0.566 |
| 11 | 0.884 | 0.031 | 0.829–0.939 | 0.502 / 0.571 |
| 15 | 0.881 | 0.027 | 0.841–0.935 | 0.500 / 0.576 |
| 19 | 0.885 | 0.028 | 0.831–0.938 | 0.501 / 0.565 |
| 23 | 0.886 | 0.026 | 0.842–0.937 | 0.500 / 0.560 |
| 27 | 0.887 | 0.024 | 0.849–0.933 | 0.499 / 0.573 |
| 31 | 0.896 | 0.022 | 0.836–0.941 | 0.501 / 0.571 |
| 35 | 0.899 | 0.019 | 0.857–0.941 | 0.503 / 0.570 |
| 39 | 0.896 | 0.021 | 0.836–0.935 | 0.502 / 0.563 |
| 43 | 0.896 | 0.017 | 0.850–0.923 | 0.501 / 0.560 |
| 47 | 0.894 | 0.018 | 0.854–0.932 | 0.500 / 0.568 |
| 51 | 0.887 | 0.024 | 0.826–0.933 | 0.498 / 0.572 |
| 55 | 0.883 | 0.025 | 0.831–0.923 | 0.496 / 0.560 |
| 59 | 0.885 | 0.025 | 0.836–0.924 | 0.498 / 0.562 |
| 63 | 0.883 | 0.023 | 0.839–0.921 | 0.501 / 0.561 |

## Reading

- The residual stream at A predicts the agent's coming verdict far above chance. It beats the
  episode-number baseline narrowly (17 of 25 folds) and the relaxation flag clearly.
- **It does not beat the previous episode's feedback features** (−0.008 AUC, better in 9 of
  25 folds). On this evidence, what A carries about the verdict is no more than what the
  previous episode's outcome already gives a five-feature logistic regression. This is not yet
  evidence of a settled decision beyond the feedback history.
- Even layer 3 reaches 0.868, about the episode-number baseline. This fits a probe that mostly
  reads where the agent is in the trajectory and what happened last time.
- The probe set mixes two things. Every onset row is accept, and onset follows certain feedback
  patterns, so the label is partly "is this the onset episode".

## Follow-ups

Code: `tools/step2/followups.py`; numbers in `agent-collusion/results/step2/followups.json`. Rows
are sorted by (run, agent, episode), so A and A_call share identical splits (5 × 5 trajectory-grouped
folds, as above). The A probe reproduces the main run: 0.888 ± 0.023.

### Does the probe add anything on top of the feedback features?

Each outer split uses training data only:
- the layer and the L2 strength come from the inner CV;
- the training rows get out-of-fold probe logits over the inner splits;
- a logistic regression (C=1) is fitted on [probe logit + the 7 standardised feedback features];
- the same model on the feedback features alone (C=1) is the comparison.

| | A | A_call |
|---|---|---|
| Probe alone (selected layer) | 0.888 ± 0.023 | 0.898 ± 0.028 |
| Feedback features alone | 0.896 ± 0.019 | 0.896 ± 0.019 |
| **Probe + feedback** | **0.903 ± 0.019** | **0.910 ± 0.019** |
| Gain over feedback (paired) | +0.007 ± 0.016, better in 19/25 folds | +0.013 ± 0.015, better in 19/25 folds |
| Probe weight in the combined model | positive in 25/25 folds (0.18–1.73) | positive in 25/25 folds (0.19–1.76) |

So the residual stream does carry some verdict information that the previous episode's outcome
does not, but not much: about +0.01 AUC on top of 0.896.

### Probe at A_call (the opening of the agent's own first communication turn)

- **Selected-layer AUC:** 0.898 ± 0.028 (range 0.833–0.956).
- **Layers chosen:** 39 (12/25), 35 (10/25).
- **Null:** 20 permutations, mean 0.498, 95th percentile 0.548. The probe beats all 20, so
  p = 0.048, the smallest possible with 20.
- **Majority-class accuracy:** 0.604.
- **Paired against A on the same folds:** +0.010 ± 0.023, better in 19 of 25 folds.

The agent that speaks second sees the peer's first message before A_call, so some of this
difference may come from the peer's opening rather than the agent's own state. Splitting by
speaking order would separate the two.

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_call AUC | .883 | .881 | .885 | .891 | .889 | .883 | .885 | .889 | .906 | .905 | .898 | .895 | .886 | .884 | .883 | .895 |
| fold SD | .029 | .032 | .030 | .028 | .029 | .029 | .027 | .026 | .029 | .029 | .028 | .027 | .024 | .022 | .018 | .016 |

## Overall reading for step 2

- At the start of the communication phase, the agent's coming verdict is mostly predictable
  from what already happened: the previous episode's outcome gives 0.896 AUC.
- The activations reproduce that and add a small, consistent extra (+0.007 at A, +0.013 at
  A_call; better in 19/25 folds each).
- That is weak evidence that something beyond the feedback history is already present before
  any message is sent. It is not evidence of a decision that is settled before the agreement.

## Restricted to agents whose previous verdict was REJECT

Added after steps 2–3: who switches to ACCEPT? Same pipeline, splits, baselines and null
(100 permutations), on the 210 onset and before-onset rows where the agent's own verdict in the
previous episode was reject (episode 1 excluded). 104 of them switch to accept (all 71 onset rows
and 33 of 139 before-onset rows); 106 stay at reject. The rows cover all 50 trajectories. The
majority-class accuracy is 0.477. Numbers: `results_prev_reject.json`, `followups_prev_reject.json`.

| | AUC (mean ± SD over 25 folds) | min–max |
|---|---|---|
| **A probe, layer chosen in training folds** | **0.774 ± 0.064** | 0.628–0.876 |
| Baseline: feedback features | **0.784 ± 0.043** | 0.699–0.854 |
| Baseline: episode number | 0.734 ± 0.055 | 0.627–0.835 |
| Baseline: previous relaxation flag | 0.666 ± 0.044 | 0.601–0.778 |
| Shuffled-label null (selected layer) | 0.500 (95th pct 0.585) | |
| A_call probe | 0.806 ± 0.044 (null 0.494, 95th pct 0.542) | 0.704–0.893 |

Notes on the table:
- **A probe vs baselines (paired):** −0.010 against feedback (better in 12/25 folds), +0.039
  against episode (18/25), +0.107 against relaxation (22/25).
- **Layers chosen:** 35 (9/25) and 39 (6/25) at A; 35 (16/25) and 39 (7/25) at A_call.
- **By layer:** at A, from 0.721 (layer 3) to 0.798 (layer 35), with a fold SD of 0.04–0.07.
  Every layer beats all 100 permutations.

Added value on top of the feedback features (stacked as in the follow-ups above):

| | A | A_call |
|---|---|---|
| Feedback alone (C=1) | 0.788 ± 0.039 | 0.788 ± 0.039 |
| Probe + feedback | 0.805 ± 0.053 | **0.826 ± 0.032** |
| Gain (paired) | +0.017 ± 0.043, better in 18/25 folds | **+0.038 ± 0.049, better in 20/25 folds** |

The A_call probe beats the A probe by +0.033 ± 0.077 (17/25 folds).

**What it shows.** The task is harder here (0.77–0.81 instead of 0.89), because the previous
verdict no longer gives the answer away. The probe still does not clearly beat the previous
episode's outcome on its own. On top of it, the probe adds more than in the full set:
+0.017 at A and +0.038 at A_call, against +0.007 and +0.013 before. The gain is largest at the
agent's own first turn, so the activations carry some information about who is about to switch
that the feedback history does not. It is still modest and varies a lot across folds.
