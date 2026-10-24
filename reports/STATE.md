# State of the project (2026-10-24)

## The story so far

In the paper's 50 Qwen 3.6 trajectories, the previous episode's feedback mostly decides the coming
verdict before the exchange starts.
- At A and at every message, a probe is no better than a feedback baseline (0.89 vs 0.90 AUC).
- Flipping one line of feedback ("you reached the wrong verdict" → "right") removes most onset
  accepts (share 0.88 → 0.33).

The activation at C (the verdict turn, before any thinking) reads the decision. It predicts how
often fresh resamples accept (ρ 0.87). On typical calls the feedback baseline does just as well
(ρ 0.85). On the hard cases (reverted rejects, flips, confidently wrong calls) the C probe is the
only predictor that works (ρ 0.74, against −0.09 for feedback, 0.19–0.23 for word counts).

The reasoning mostly ratifies a verdict already set at C:
- typical calls are settled at C (31 of 40 random calls);
- with thinking off, shares stay close (ρ 0.91);
- where thinking does change things, it pushes toward accept (flips: 0.37 accept without
  thinking, 0.69 with), and an early "I must reject" is usually overturned.

The agreement text is not load-bearing. Removing the agreement lines barely moves the C probe
(0.976 → 0.940) or the resampled verdict (0.93 accept). Agreement and no-agreement onsets look
alike until C.

Steering along the C direction is not shown yet. A 3-call pilot pushed one hard case to reject,
but random directions were inconclusive, and the "ablation" condition was confounded (it was a
push toward accept).

## Reports, in order

| Report | Contents |
|---|---|
| `2026-10-23-overnight.md` | Setup S7: two 10-episode trajectories on our server. One completes (mutual ACCEPT from episode 4); one crashed on a 64k-token runaway reply. First resampling plan. |
| `2026-10-23-step2-probe-a.md` | Step 2: probe at A against the feedback, episode and relaxation baselines; shuffled null; positive control. Follow-ups: added value on top of feedback, A_call, and agents whose previous verdict was reject. |
| `2026-10-23-step3-trace-and-c.md` | Step 3: per-position probes through the exchange (first beats the baseline at C); agreement-onset curves; probe at C; A↔C transfer; speaking order; last message vs C with text baselines; the C probe traced everywhere; the previous-reject rows. |
| `2026-10-23-step6-projections.md` | Step 6: the C probe through each verdict call's own thinking; separation by tenth (C probe and fresh probes); lock-in point vs the first verdict statement; edited prompts at C; reverted calls as case studies. |
| `2026-10-24-step4-5-resampling.md` | Steps 4–5, vLLM session 1: full resamples from C, continuations from cuts of the saved thinking, flips, edited prompts (variant B) resampled. |
| `2026-10-24-session2-and-steering-pilot.md` | Baselines vs resampled shares, flip base rates, steering direction, predictors by cut. vLLM session 2: thinking off, feedback edit, from A with the peer replayed. The session-2 cost overrun and its fix. Steering pilot and the sizing of the main run. |

## Where the data lives

Everything below is under `agent-collusion/results/` (git-ignored) unless noted.

| What | Path |
|---|---|
| Paper traces, labels, rebuilt verdict requests | `paper/traces_qwen36_main/`, `paper/labels/`, `paper/paper_wire_verdict.jsonl` |
| Activation cache (step 1): jobs, index, episode labels | `step1_cache/jobs.jsonl`, `index.jsonl`, `episodes.jsonl`; raw download in `step1_cache/acts-v1/acts/`; Modal Volume `qwen36-activations:/cache-v1` |
| Packed layers (layer 3, 7, …, 63, row = index row, bf16 bits) | `step1_cache/packed/layer_XX.npy` (load with `tools/step1_cache/pack.py:load_layer`) |
| Probe results | `step2/` (`results*.json`, `followups*.json`), `step3/` (`results.json`, `scores.jsonl`, `trace_c_probe.json`, `text_baselines.json`), `step3_prev_reject/` |
| Step 6 projections | `step6/acts/` (per-token C-probe logits, tenth vectors), `step6/probes.npz` (fold C probes, raw weights), `calls.jsonl`, `per_call.jsonl`, `verdict_statements.jsonl`, `analysis.json` |
| Edited transcripts (step 5) | `step5/edits.jsonl` |
| Resampling, session 1 | `step4_5/plan.json`, `responses.jsonl`, `analysis.json`, `flips.json`, `baselines_vs_resampling.json`, `predictors_by_cut.json` |
| Resampling, session 2 | `step4_5/session2_plan.json`, `session2_single.jsonl` (thinking off, feedback edit), `session2_from_a.jsonl`, `from_a_llm_calls.jsonl` |
| Steering direction and calls | `steer/direction.npz` (unit, sd, abl_mean), `random_direction.npz`, `steer_calls.json`, `direction.json`, `jobs_*.json` |
| Steering pilot and check outputs | `steer/pilot/out/`, `steer/check/out/` |
| Figures (in git) | `reports/figures/step3/`, `step3_prev_reject/`, `step4_5/`, `step6/` |

## Unsteered check and the mean-ablation value

- **Forward pass:** transformers matches vLLM's per-token log-probs on 5 paper replies to within
  0.003. For example, −0.2648 vs −0.2659, and −0.3536 vs −0.3542.
- **Shares:** 16 unsteered samples per call, transformers vs vLLM: 13/16 vs 8/16, 6/16 vs 9/16,
  9/16 vs 6/16. That's consistent within sampling noise (Fisher p 0.14–0.5, differences go both
  ways).
- **Path difference:** the only one is that vLLM forced the `submit_verdict` call; no sample was
  invalid.
- **Mean-ablation target:** the projection on the C direction is set to **−13.07**, the midpoint of
  the class means on the training trajectories (−2.27 ACCEPT, −23.86 REJECT; SD 13.44).

## Next: steering design A (waits for approval)

13 calls (10 hard cases first, then 3 settled accepts), strength 4 SD, 8 samples per condition.
Conditions: unsteered (in batch), target direction, random direction, mean-ablation. All
conditions of a call share batched `generate` calls. **Cap $11**, hard stop 165 min, estimate
$8.40–10.20.

```
caffeinate -i modal run server/steer_qwen.py --jobs agent-collusion/results/steer/jobs_reduced.json \
  --tag steer-A --direction agent-collusion/results/steer/direction.npz \
  --random-direction agent-collusion/results/steer/random_direction.npz \
  --batched --strength 4 --samples 8 --in-batch-unsteered --max-new-tokens 6144 \
  --timeout-min 165 --go
```

The Modal function timeout (165 min) is the hard stop. The worker also checks its deadline before
every call and every batch.

## Modal

$46 credit left (dashboard). Metered so far: **$33.85**, all covered by credits, billed $0.00
(`modal billing summary`).

## Rules

- Ask before any GPU job.
- Report GPU minutes and cost after each one.
- Every GPU-backed runner checks its deadline before every call (and batch), has a process-level
  hard stop, and runs under `caffeinate`.
- No Co-Authored-By lines in commits; only the user appears on GitHub.
