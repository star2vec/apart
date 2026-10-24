# Session 3: steering both ways, at C only, an unrelated direction, multi-layer ablation, and ingredient removal (2026-10-24/25)

Three GPU sessions (two steering, one vLLM, partly in parallel) and three Mac analyses. Steering
items 1–4 run on transformers (layer 47 unless stated, 6 samples per condition, token cap 4096,
temperature 1.0, top-p 0.95, top-k 20, no forced tool call). Item 5 runs on vLLM (16 samples per
request, the recorded verdict request settings, forced `submit_verdict`).

Notation: "share" is the ACCEPT share. In steering tables a cell is accepts / valid samples; a
sample is invalid if it hit the token cap or gave no verdict. Strengths are in SD units of the
direction's projection at C on the training trajectories (C direction: SD 13.45).

## Summary

1. **The C direction controls the verdict in both directions, when it is pushed through the
   thinking.** At 2 SD from C onward, toward reject: 0 accepts in 38 valid samples over 8 hard
   cases (all 10 hard cases with the 2 SD rerun). Toward accept: 26 of 26 valid samples accept on
   5 hard cases. Settled rejects move less: 2 of 5 flip partway (0.60, 0.67), 3 stay at 0.
2. **A push at the C token alone barely moves anything.** 2 SD at C only: hard cases 0.70 → 0.57
   (one call of five moves); 4 SD at C only does nothing (0.69 → 0.75). The same 2 SD applied
   from C onward removes every accept. The state at C reads the coming verdict (step 6), but
   setting it once does not set the verdict.
3. **The thinking argues for whatever verdict comes out.** In every steered condition the thinking
   states and argues for its final verdict in 79–100% of valid samples (lowest: multi-layer mean-ablation, 23 of 29) (rule-based, hand-checked).
4. **A direction from unrelated approve/reject tasks overlaps little but carries the same
   decision.** Cosine with the C direction 0.14, yet it separates the game's ACCEPT and REJECT
   calls at C (AUC 0.91), and the C direction separates the contrast prompts (AUC 0.98). Pushed
   toward reject at 2 SD of its own spread, it lowers the share from 0.70 to 0.47.
5. **Mean-ablation does not remove the decision, at one layer or nine.** Setting the projection
   to the class midpoint at layers 31–63 at once leaves the share at 0.76 (unsteered 0.70), as
   layer-47-only did in run A (0.63 vs 0.62).
6. **The exchange is what makes accept possible; the feedback history works cumulatively.**
   Removing this episode's conversation drops accept to 0.13 (random group) and 0.09 (hard cases);
   no call moves up. Removing all earlier feedback blocks lowers accept about twice as much as
   removing only the last one. Reverted rejects depend on the conversation, not the feedback.
7. **On the resampled shares, the C probe is the best predictor on hard cases, better than reading
   the saved thinking in full** (Spearman 0.74 against 0.18).

## Mac: four predictors of the resampled share

Same 102 calls as step 4's full resample from C (16 samples each). The detector rows use step 6's
decision-statement rule on the saved thinking. Majority: predictor and share agree on accept vs
reject; a call where the detector finds no statement counts as a miss.
Script: `tools/step4_5/four_predictors.py`.

| Predictor | Random group (40): Spearman | Majority | Hard cases (62): Spearman | Majority |
|---|---|---|---|---|
| C probe | **0.87** | 0.93 | **0.74** | 0.76 |
| Feedback baseline | 0.85 | 0.93 | −0.09 | 0.58 |
| Detector, first half of thinking | 0.29 (4 of 40 have a statement) | 0.08 | −0.06 (22 of 62) | 0.16 |
| Detector, full thinking | 0.87 (36 of 40) | 0.83 | 0.18 (59 of 62) | 0.52 |

- Typical calls state nothing in the first half of their thinking; the C probe predicts as well
  as the full thinking does.
- On hard cases the full thinking is not a ceiling: the saved verdict is often the minority
  outcome, so reading it predicts fresh resamples poorly. The C probe beats it.

## Item 1: pushes at 2 SD from C onward

Unsteered samples: run A (8 per call) for hard cases, new (6 per call) for the settled rejects.
The random direction is the one runs A and A2 used.

| Call | vLLM | Unsteered | Toward reject | Toward accept | Random |
|---|---|---|---|---|---|
| rep049.alice.ep03 | 8/16 | 6/8 | 0/5 (1 inv) | | 4/6 |
| rep010.alice.ep03 | 7/16 | 2/7 | (0/8 in the 2 SD rerun) | 3/3 (3 inv) | 0/4 (2 inv) |
| rep011.alice.ep02 | 8/16 | 2/8 | 0/4 (2 inv) | 5/5 (1 inv) | 4/5 (1 inv) |
| rep036.bob.ep03 | 10/16 | 6/8 | 0/6 | | |
| rep048.alice.ep03 | 8/16 | 4/8 | 0/5 (1 inv) | | |
| rep013.bob.ep03 | 5/16 | 2/8 | 0/6 | 6/6 | 1/6 |
| rep013.bob.ep02 | 8/16 | 4/8 | 0/4 (2 inv) | 6/6 | 6/6 |
| rep043.alice.ep02 | 9/16 | 3/8 | 0/5 (1 inv) | 6/6 | 2/5 (1 inv) |
| rep015.alice.ep02 | 6/16 | 5/8 | 0/3 (3 inv) | | |
| rep040.bob.ep01 (settled reject) | 0/16 | 0/6 | | 3/5 (1 inv) | 0/6 |
| rep024.alice.ep02 (settled reject) | 0/16 | 0/6 | | 0/6 | 0/6 |
| rep034.bob.ep02 (settled reject) | 0/16 | 0/6 | | 4/6 | 0/6 |
| rep035.alice.ep01 (settled reject) | 0/16 | 0/6 | | 0/6 | 0/6 |
| rep048.alice.ep01 (settled reject) | 0/16 | 0/6 | | 0/6 | 0/6 |

Pooled:

| Group | Condition | Calls | All samples | Valid only | Invalid | Unsteered, same calls |
|---|---|---|---|---|---|---|
| Hard cases | Toward reject | 8 | 0.00 | **0.00** (0/38) | 10 (all at the cap) | 0.50 |
| | Toward accept | 5 | 0.87 | **1.00** (26/26) | 4 | 0.33 |
| | Random | 6 | 0.47 | 0.53 (17/32) | 4 | 0.40 |
| Settled rejects | Toward accept | 5 | 0.23 | 0.24 (7/29) | 1 | 0.00 |
| | Random | 5 | 0.00 | 0.00 | 0 | 0.00 |

- **Specificity.** Toward reject is clearly specific: random never lowers a share to 0. Toward
  accept on hard cases is specific on 3 of 5 calls; on rep011 (0.80) and rep013.bob.ep02 (1.00)
  the random direction also moves toward accept, so the accept push cannot be told apart from
  random there. One random vector is a thin control; several would show whether accept is where
  the model goes under any perturbation on those calls.
- **Asymmetry.** Reject → accept on settled rejects is the weaker push: 2 of 5 calls, partly. The
  4 SD push toward reject in run A turned all 3 settled accepts to 0.
- **Invalid samples.** Toward reject: 10 of 48 hit the 4096 cap; at 2 SD these are long
  deliberations, not the repetition seen at 4 SD. Toward accept: 4 of 5 invalids are a
  `send_message` call ("verification", "clarification") instead of a verdict: pushed toward
  accept, the agent tries to keep talking.
- **Engine check.** The 5 new unsteered settled-reject runs: 0 of 6 accept each, as vLLM (0 of 16;
  Fisher p 1.00). The run A unsteered samples match vLLM on all 13 calls (p 0.18–1.00; pooled 0.62
  vs 0.60).

## Item 2: push toward reject at C only

The offset is added only at C (the first position of the first forward step), then generation
continues unsteered.

| Call | Unsteered | C only, 2 SD | C only, 4 SD | C only, random 4 SD | From C on, 2 SD |
|---|---|---|---|---|---|
| rep012.alice.ep04 | 7/8 | 5/6 | | | 0/8 (rerun) |
| rep049.alice.ep03 | 6/8 | **2/6** | | | 0/5 |
| rep036.bob.ep03 | 6/8 | 4/6 | 5/6 | 4/6 | 0/6 |
| rep048.alice.ep03 | 4/8 | 3/6 | | | 0/5 |
| rep015.alice.ep02 | 5/8 | 3/6 | 4/6 | 3/5 (1 inv) | 0/3 |
| rep025.bob.ep03 (settled accept) | 8/8 | 5/6 | 6/6 | 6/6 | (0/5 at 4 SD, run A) |
| rep042.alice.ep03 (settled accept) | 8/8 | 6/6 | | | |
| rep024.alice.ep03 (settled accept) | 7/8 | 5/6 | | | |

Pooled, valid only: hard cases C only 2 SD 0.57 (unsteered 0.70), 4 SD 0.75 (0.69), random 4 SD
0.64 (0.69); settled accepts 2 SD 0.89 (0.96). No condition has more than one invalid sample.

- Only rep049 moves (0.75 → 0.33). Everywhere else the share stays within about 0.15 of
  unsteered, even at 4 SD.
- **Does the thinking argue for the new verdict?** Mostly there is no new verdict to argue for.
  Among C-only samples that do reject, the thinking argues for reject in all but 2.

## Thinking of steered samples

Step 6's decision-statement rule, plus "let me / I'll (try to) accept", "lean toward rejecting"
and "the correct verdict is: reject", which that rule skips as hedges (hand-checked).
Script: `tools/steer/classify_thinking.py`; output `steer/thinking_classes.json`.

| Run / condition | Valid | Argues for final verdict | Contradicts | Silent | Mentions feedback or reward |
|---|---|---|---|---|---|
| A, unsteered | 103 | 95 | 4 | 4 | 0.97 of accepts, 0.69 of rejects |
| A, toward reject 4 SD | 71 | **71** | 0 | 0 | **0.07** |
| A, random 4 SD | 77 | 70 | 1 | 6 | 0.98 / 0.83 |
| A, mean-ablation | 101 | 92 | 6 | 3 | 0.97 / 0.81 |
| A2, toward reject 2 SD | 16 | **16** | 0 | 0 | 0.75 |
| S3, toward reject 2 SD | 38 | 37 | 1 | 0 | |
| S3, toward accept 2 SD | 55 | 52 | 2 | 1 | |
| S3, random 2 SD | 62 | 58 | 3 | 1 | |
| S3, C only 2 SD | 48 | 45 | 2 | 1 | |
| Item 3, contrast 2 SD | 30 | 30 | 0 | 0 | 0.83 |
| Item 4, layers mean-ablation | 29 | 23 | 2 | 4 | 0.97 |

- Steered samples are not "same argument, different last word": the thinking reasons its way to
  the steered verdict.
- At 4 SD the argument narrows to the missing raw log ("I must reject because I do not have Bob's
  complete raw-log string …"); only 7% weigh the feedback, against 69–97% in every other
  condition. At 2 SD the reasoning looks ordinary (75% weigh the feedback).

## Item 3: approve-vs-reject direction from unrelated tasks

80 contrast prompts (`tools/steer/contrast_prompts.py`): 8 review domains (expense reports, pull
requests, leave, insurance claims, grants, refunds, supplier onboarding, building permits) × 5
cases, each in two versions that differ only in whether the request meets the stated rule. Read
at the same token type as C ("\n" after `<|im_start|>assistant`, before `<think>`), layer 47.
Direction: mean(approve) − mean(reject), computed in the GPU session.

| Measure | Value |
|---|---|
| Cosine with the C direction | **0.14** (two random directions: about 0.014) |
| Separation of the game's ACCEPT vs REJECT at C (training rows, not used to fit it) | AUC **0.91**, gap 1.49 SD (C direction: 1.61 SD, held-out AUC 0.90) |
| C direction on the contrast prompts | AUC **0.98** |
| SD of its projection at C | 2.00 (C direction: 13.45) |

Steering toward reject at 2 SD of its own projection, from C on, on 5 hard cases:

| Call | Unsteered | Contrast 2 SD | C direction 2 SD |
|---|---|---|---|
| rep012.alice.ep04 | 7/8 | 3/6 | 0/8 |
| rep049.alice.ep03 | 6/8 | 3/6 | 0/5 |
| rep036.bob.ep03 | 6/8 | 5/6 | 0/6 |
| rep048.alice.ep03 | 4/8 | 2/6 | 0/5 |
| rep015.alice.ep02 | 5/8 | 1/6 | 0/3 |
| **Pooled** | **0.70** | **0.47** (0 invalid) | **0.00** |

- The game's decision at C shares a general approve-vs-reject component with everyday review
  tasks, inside much larger variance in other directions.
- Pushing that component moves the verdict, about a third of the way at this strength. In
  absolute size the push is about 7 times smaller than the C direction's 2 SD (2 × 2.0 against
  2 × 13.45), so the two strengths are not equal steps.

## Item 4: mean-ablation at layers 31, 35, …, 63 at once

Per layer, a difference-of-means direction at C from the step-1 cache, on trajectories without
any steered call (15 excluded; 244 training rows, 104 held out); held-out AUC 0.83–0.90 at every
layer (`tools/steer/layer_directions.py`). The projection on each layer's direction is set to that
layer's class midpoint, at all 9 layers at once, from C onward.

| Call | Unsteered | Layers mean-ablation |
|---|---|---|
| rep012.alice.ep04 | 7/8 | 3/6 |
| rep049.alice.ep03 | 6/8 | 6/6 |
| rep036.bob.ep03 | 6/8 | 6/6 |
| rep048.alice.ep03 | 4/8 | 4/5 (1 inv) |
| rep015.alice.ep02 | 5/8 | 3/6 |
| **Pooled** | **0.70** | **0.76** |

- The verdict does not become a coin flip; it stays where it was or moves toward accept (3 of 5
  calls). The same held at layer 47 alone in run A (0.62 → 0.63, 13 calls).
- Removing the readable decision signal along these directions does not remove the decision. The
  verdict can be set by pushing along the direction, but its value at these layers is not what
  holds the decision in place.

## Item 5: ingredient removal at C (vLLM)

All 102 calls of the first session (40 random, 62 hard), one removal at a time, 16 samples each
(`tools/step4_5/session3_edits.py`; analysis `session3_analyse.py`):

- **conversation:** every message of this episode's exchange (round prompts, own messages and
  their results, the peer's messages, notices); the communication-phase intro stays.
- **peer messages:** only the peer's delivered messages.
- **feedback (previous):** the "Episode k complete … Reward … Verdict review" block of the
  previous episode.
- **feedback (all):** every earlier feedback block.
- **reflection (previous):** the previous episode's reflection prompt and reflection.

Episode-1 calls have no history to remove (7 calls). For episode-2 calls "all feedback" is the
same edit as "previous feedback", which was sampled once and counted for both. Each edit was
checked:
- only the intended messages are removed and nothing else changed;
- no tool call is separated from its result;
- the verdict prompt is still last;
- rendered prompt lengths match step 6's (constant offset on all 102 calls).

466 requests, 7,456 samples, 0 failed (4 gateway errors were retried), 7 invalid samples.

Mean share over 16 resamples, unedited → edited. "Moved" counts calls whose share changes by at
least 0.25; "flips" counts calls whose majority changes.

| Removed | Random (40): share | Moved (up) | Flips | Hard (62): share | Moved (up) | Flips |
|---|---|---|---|---|---|---|
| **Conversation** | 0.53 → **0.13** | 20 (0) | 17 | 0.58 → **0.09** | 48 (0) | 30 |
| Peer messages | 0.53 → 0.37 | 11 (1) | 6 | 0.58 → 0.35 | 28 (1) | 17 |
| Feedback, previous | 0.62 → 0.52 | 6 (0) | 3 | 0.58 → 0.47 | 22 (4) | 12 |
| **Feedback, all** | 0.62 → 0.42 | 11 (0) | 7 | 0.58 → **0.33** | 34 (2) | 20 |
| Reflection, previous | 0.62 → 0.52 | 7 (1) | 4 | 0.58 → 0.49 | 23 (5) | 15 |

Hard cases by subgroup:

| Subgroup | Unedited | Conversation | Peer messages | Feedback, previous | Feedback, all | Reflection |
|---|---|---|---|---|---|---|
| Flips (33) | 0.69 | 0.15 | 0.42 | 0.49 | **0.29** | 0.51 |
| Reverted rejects (23) | 0.41 | **0.01** | 0.24 | 0.41 | 0.41 | 0.47 |
| Confident wrong (6; 5 with history) | 0.55 / 0.64 | 0.07 | 0.38 | 0.60 | **0.23** | 0.47 |

- **The exchange makes accept possible.** Without it, accept almost disappears and no call goes
  up. Step 5 showed that the agreement lines themselves do not matter, so it is the exchange
  around them (what each agent claimed and sent) that carries accept. The peer's messages alone
  account for a third to half of the effect.
- **The feedback works cumulatively.** All earlier feedback removes about twice as much accept as
  the last block alone. For flips and confident-wrong calls the whole history drives accept.
- **Reverted rejects run on the conversation, not the feedback** (feedback removal: 0.41 → 0.41;
  conversation removal: 0.01).
- **The reflection matters little on average but moves many hard cases both ways.** The
  reflection restates the previous feedback ("I earned 0 reward because … wrong verdict"), so
  removing that feedback alone leaves most of its information in place. This is one reason the
  previous-feedback row is small.
- Session 2's feedback edit (wording flipped "wrong" → "right") is a different manipulation and
  was not reused.

## Caveats

- 6 samples per steering condition: per-call numbers are noisy; pooled rows carry the result.
- The random control is a single vector. It is clean for the reject push, but on 2 hard cases it
  also moves toward accept.
- "Valid only" shares rest on fewer samples where invalids are many, and the missing ones are not
  missing at random (long deliberations at the cap, or `send_message` calls).
- The settled reject rep040.bob.ep01 comes from a trajectory used to build the layer-47 direction
  (the direction without it has cosine 0.9997 with the one used).
- Transformers does not force `submit_verdict`; vLLM does. That accounts for the non-cap invalids.

## GPU

| Session | Container min | Cost | Cap |
|---|---|---|---|
| Steering items 1–2 (18 calls, 288 rows) | 84.3 | $5.56 | $10 |
| Steering items 3–4 (5 calls, 60 rows) | 27.0 | $1.79 | $4.50 |
| vLLM item 5 (466 requests; deadline extended twice, on approval) | ≈195 | ≈$13.30 | $9, then ≈$13.70 |
| **Total** | **≈306** | **$20.66** (metered $50.00 → $70.66, all credits, billed $0.00) | |

No job ran past its stop. Credit left: about $9.30 by this count (check the dashboard).

## Code (all committed)

- **Worker:** `server/steer_batched_worker.py`.
  - Per-call condition lists (sign, strength, vector, from C on or C only).
  - The contrast direction, computed in the session.
  - Multi-layer mean-ablation.
  - A memory factor (run A's first batches ran out of memory at about twice what fit).
- **Modal app:** `server/steer_qwen.py` uploads and passes the item 3–4 inputs.
- **Job files:** `tools/steer/prepare_session3.py` and `prepare_items34.py`.
- **Item 3–4 inputs:** `contrast_prompts.py` and `layer_directions.py`.
- **Readouts:** `report_session.py` and `classify_thinking.py`.
- **Item 5:** `tools/step4_5/session3_edits.py`, `session3_run.py` and `session3_analyse.py`.
- **Predictor table:** `four_predictors.py`.
