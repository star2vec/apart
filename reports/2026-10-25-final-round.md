# Final round: five predictors, the contrast direction at A_call, own messages removed, feedback reversed (2026-10-25)

Three Mac analyses (no GPU) and one vLLM session (16 samples per request, the recorded verdict
request settings, forced `submit_verdict`). The vLLM session ended early when the Modal credit ran
out (see GPU): item 4 is complete, item 5 has 55 of its 95 calls.

## Summary

1. **The C probe's lead on hard cases does not come from the confidently wrong calls.** Without
   them it still predicts the resampled shares best (Spearman 0.71–0.74, against −0.28 to −0.20
   for the feedback baseline and 0.28–0.30 for the full saved thinking).
2. **A direction from unrelated approve/reject tasks, never fitted on the game, comes second**
   (Spearman 0.74 on the random group, 0.58–0.61 on hard cases).
3. **At A_call that direction leans toward the coming verdict but never beats the feedback
   baseline** (AUC 0.72 vs 0.90; on previous-REJECT rows 0.62 vs 0.78; 0 of 25 folds better).
4. **The agent's own messages matter least for accept.** Removing them lowers accept a little
   (0.53 → 0.49 random, 0.57 → 0.47 hard). Removing the peer's messages does about three times as
   much, and removing both (the whole conversation) cuts accept to about 0.1.
5. **Reversing the previous feedback moves the verdict more than deleting it, and only one way.**
   Turning "you reached the wrong verdict" into "right" lowers accept (0.59 → 0.37). The reverse
   edit changes nothing (0.53 → 0.53).

## 1–2. Five predictors of the resampled share

Target: the share of 16 fresh resamples from C that accept (first session, 102 calls).
Script: `tools/step4_5/five_predictors.py`; output `step4_5/five_predictors.json`.

Predictors:
- **C probe:** step 6's out-of-fold probe at C.
- **Feedback baseline:** the previous episode's outcome, out of fold.
- **Verdict stated in the first half / the full saved thinking:** step 6's decision-statement rule.
- **Contrast direction:** the projection of the call's C activation (layer 47, step-1 cache) on
  steering item 3's approve-vs-reject direction from unrelated review tasks. It is not fitted on
  the game. Its majority threshold is the midpoint of the class means on the other trajectories'
  onset and before-onset rows (leave one trajectory out).

The analysis tags 10 hard-case calls as confidently wrong: 6 have that as their only group, and 4
are flips that also carry the tag. All 10 were selected on probe errors on the saved verdict, so
hard cases are shown with and without them.

Each cell is the Spearman correlation with the share, with majority agreement in brackets.

| Predictor | Random (40) | Hard (62) | Hard without the 6 (56) | Hard without all 10 (52) |
|---|---|---|---|---|
| C probe | **0.87** (0.93) | **0.74** (0.76) | **0.74** (0.75) | **0.71** (0.73) |
| Feedback baseline | 0.85 (0.93) | −0.09 (0.58) | −0.20 (0.55) | −0.28 (0.56) |
| Verdict stated, first half | 0.29 (0.08) | −0.06 (0.16) | −0.01 (0.18) | −0.10 (0.14) |
| Verdict stated, full thinking | 0.87 (0.83) | 0.18 (0.52) | 0.28 (0.55) | 0.30 (0.56) |
| Contrast direction | 0.74 (0.83) | 0.58 (0.68) | 0.61 (0.70) | 0.59 (0.67) |

For the two detector rows, a call with no stated verdict counts as a majority miss. Coverage (calls
with a statement), first half / full: random 4 / 36; hard 22 / 59; without the 6, 21 / 53; without
all 10, 17 / 49.

- The confidently wrong calls were picked for probe errors on the saved verdict, yet the probe
  still tracks their resampled shares; leaving them out barely changes its score.
- The contrast direction beats the feedback baseline and the full saved thinking on hard cases,
  with no fitting on the game at all.

## 3. The contrast direction at A_call

The direction is read at A_call (layer 47, communication pass: the opening of the agent's first
communication turn, the same token type as C) from the step-1 cache. Rows are as in step 2:
onset and before-onset episodes, labelled with the agent's own verdict, and the previous-REJECT
subset (who switches to accept). Both are scored on the same trajectory-grouped test folds
(5 folds × 5 repeats): the contrast direction as a fixed projection, the feedback baseline fitted
out of fold. Script: `tools/steer/contrast_at_a_call.py`; output `steer/contrast_at_a_call.json`.

| Rows | Contrast direction AUC | Feedback baseline AUC | Contrast better (folds) | C direction at A_call (reference) |
|---|---|---|---|---|
| Onset and before-onset (348 rows, 50 trajectories) | 0.72 ± 0.05 | **0.90** ± 0.02 | 0 of 25 | 0.86 |
| Previous REJECT (210 rows) | 0.62 ± 0.05 | **0.78** ± 0.04 | 0 of 25 | 0.70 |

- At A_call the general approve/reject component already leans toward the coming verdict, but
  much less than the feedback does, including on the rows where the question is who switches.
- The C-direction reference was fitted on C rows of most of these trajectories, so it is partly
  in-sample here.

## 4. Own messages removed at C

All 102 calls. The edit removes every `send_message` call of the agent in this episode with its
tool result, including failed attempts. The peer's delivered messages, round prompts and notices
stay. Checks, as before:
- only those messages are removed;
- no tool call is separated from its result;
- the verdict prompt is still last;
- rendered prompt lengths are recorded before and after.

Script: `tools/step4_5/session4_edits.py`.

102 requests, 1,632 samples, 0 invalid. Mean share over 16 resamples, on the same calls. The
peer-removed and conversation-removed shares are item 5 of session 3. Each edit column gives the
calls whose share moves by at least 0.25 down / up, then the majority flips.

| Group | Unedited | Own messages removed | Down / up | Flips | Peer's messages removed | Whole conversation removed |
|---|---|---|---|---|---|---|
| Random (40) | 0.53 | **0.49** | 5 / 0 | 2 | 0.37 | 0.13 |
| Hard (62) | 0.57 | **0.47** | 18 / 5 | 10 | 0.35 | 0.09 |
| Flips (33) | 0.69 | 0.55 | 12 / 1 | 7 | 0.42 | 0.15 |
| Reverted rejects (23) | 0.41 | 0.34 | 6 / 3 | 3 | 0.24 | 0.01 |
| Confident wrong (6) | 0.55 | 0.53 | 0 / 1 | 0 | 0.38 | 0.07 |

- What the peer says carries more of accept than what the agent sent.
- Neither side alone accounts for the effect of removing the conversation. The two partial removals
  add up to far less than the whole, so accept depends on the exchange as a whole.

## 5. Previous feedback reversed at C (cut short: 55 of 95 calls)

The previous episode's feedback block keeps its place, with the outcome reversed:
- both verdict-review lines flip ("reached the wrong verdict" ↔ "right");
- the reward becomes 20 minus the old one, following the reward rule (+10 when the peer's verdict
  on you is right, +10 when yours on the peer is right). A +10 episode stays +10, with its lines
  swapped.

The reflection that follows still describes the original outcome. Episode-1 calls have no previous
episode. Checks: exactly one reward line and two review lines change, and nothing else.

55 requests, 880 samples, 0 invalid. The other 40 calls were not run (see GPU). They are the later
ones in the request order (which alternated random and hard calls), so these 55 are not a random
half.

| Group | Calls | Unedited | **Feedback reversed** | Down / up | Flips | Feedback removed (session 3) |
|---|---|---|---|---|---|---|
| Random | 28 | 0.65 | **0.46** | 8 / 1 | 4 | 0.55 |
| Hard | 27 | 0.50 | **0.36** | 10 / 4 | 8 | 0.39 |
| Flips | 12 | 0.64 | 0.29 | 8 / 0 | 3 | 0.39 |
| Reverted rejects | 11 | 0.31 | **0.46** | 0 / 3 | 2 | 0.30 |
| Confident wrong | 4 | 0.61 | 0.33 | 2 / 1 | 3 | 0.64 |

By what the original feedback said about the agent's own verdict:

| Original | Edited to | Calls | Unedited | Reversed |
|---|---|---|---|---|
| "You reached the **wrong** verdict" | "right" | 39 | 0.59 | **0.37** |
| "You reached the **right** verdict" | "wrong" | 16 | 0.53 | 0.53 |

- Reversing the feedback moves the verdict more than deleting it does (random: reversed 0.46,
  removed 0.55). The feedback works as a signal to change course, not only as context.
- The effect runs one way. Telling an agent its earlier verdict was right, when it was wrong,
  lowers accept, which fits the main finding that "you were wrong (to reject)" is what pushes
  toward accept. Telling it wrong when it was right changes nothing here.
- Reverted rejects go the other way (0.31 → 0.46, 11 calls).
- Session 2's feedback edit was a different manipulation (it changed one line on onset calls,
  0.88 → 0.33).

## GPU

| Session | Container time | Cost |
|---|---|---|
| vLLM, items 4–5 (02:54 to about 04:05 EEST; deadline extended once, on approval) | ≈70 min | **$4.80** |

The metered total went from $70.66 to $75.46. **The credit ran out mid-session:** total credits
were about $75.39, so about $4.70 was left at the start, not the $9.30 estimated from metered
totals. The dashboard figure was never confirmed, and the session's cap ($6, then about $6.60
after the extension) was sized against a balance that wasn't there. Modal disabled the workspace
("workspace … is disabled") instead of billing, so the billed cost is $0.00. The vLLM endpoint
returned 404 from then on, and the runner was stopped. Finishing item 5 (40 requests) would take
about 20–25 minutes and about $1.75, after the workspace is re-enabled. **It was skipped by
decision.**

## Code (all committed)

- Predictor table: `tools/step4_5/five_predictors.py` (renamed from `four_predictors.py`).
- Contrast direction at A_call: `tools/steer/contrast_at_a_call.py`.
- Edits: `tools/step4_5/session4_edits.py`.
- Runner: `tools/step4_5/session3_run.py`, now with `--edits`, `--priority` and `--out-name`.
- Analysis: `tools/step4_5/session4_analyse.py`.
