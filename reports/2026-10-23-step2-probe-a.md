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

## Not done (possible follow-ups, Mac only)

- Probe plus feedback features against feedback features alone (does A add anything on top?).
- The same probe at `A_call` (the agent's own first turn, after the peer's opening message for
  the agent that speaks second).
