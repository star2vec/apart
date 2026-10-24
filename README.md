# Apart AI Collusion Research Sprint (Oct 23–25, 2026)

Solo entry. Runs the two-agent collusion environment against a self-hosted
Qwen3.6-27B served with vLLM on Modal, and studies the paper's 50 released Qwen 3.6
trajectories from the inside: residual-stream probes, resampling and edited transcripts.

**Question.** Does collusion that starts with an explicit agreement look different from
collusion that starts without one? Is the decision to collude already settled before the
agreement? Is there a grey area between the agreement and the ACCEPT verdict?

## Findings so far (steps 1–6)

Points in an episode: **A** = start of the communication phase; **C** = the verdict turn,
right before its thinking. "Feedback baseline" = a logistic regression on the previous
episode's outcome (both verdicts, whether each was right, reward, episode number).

1. **Before the verdict prompt, the activations add little beyond the history.** At A and at
   every message of the exchange, a probe for the agent's coming verdict is level with the
   feedback baseline (0.89 vs 0.90 AUC). On top of the baseline it adds about +0.01, or +0.04
   for agents switching from reject at their first turn.
2. **The verdict becomes readable when it is asked for.** At C, before any thinking, the probe
   beats the baseline (0.956 vs 0.905, interval above 0) on both agreement and no-agreement
   paths. It also beats the last message, which has seen the same conversation (+0.05), and
   word-count models of the messages (+0.05–0.07). For the agent who wrote the last message,
   the verdict is already about as readable in its own final message, mostly via its words.
3. **No difference between the paths until C.** Agreement and no-agreement onsets look alike
   through the exchange. The C probe stays flat through the messages and rises at C, after the
   agreement message, on all 36 agreement-onset curves.
4. **For a typical verdict call, the outcome is set at C.** 16 fresh samples from C reproduce the
   saved verdict 91% of the time, and the C probe predicts the resampled share (ρ = 0.87). Calls
   where the probe looked confidently wrong are calls where the paper's single sample was the
   unlikely outcome: resampling follows the probe in 9 of 10.
5. **The grey area is in a minority of calls.** Reverted rejects (after onset) and "flips"
   (the thinking first says reject, then accepts; 22–26% of accepts) are open at C. The saved
   thinking moves them towards their outcome gradually, through to the end. An early
   "I must reject" in a flip is not binding: 64% of continuations from just after it still
   accept.
6. **Removing the agreement lines barely changes the verdict.** That covers the judge-quoted
   lines plus every "accept…"/"agree…" sentence. The C-probe score at C goes from 0.976 to
   0.940, and resampled ACCEPT stays at 0.93 overall (one large drop in 36).

Reports: `reports/` (one per step); numbers and per-call data in `agent-collusion/results/`
(git-ignored).

## Layout

```text
.
├── agent-collusion/         # environment (copy, see Source below)
├── server/                  # Modal apps
│   ├── pins.py              # model revision, vLLM version, max-model-len, GPU
│   ├── download_weights.py  # CPU only: fills the weights Volume, verifies symlinks
│   ├── serve_qwen.py        # GPU: OpenAI-compatible vLLM endpoint
│   ├── replay_qwen.py       # GPU: transformers replay, saves residual-stream vectors
│   └── replay_worker.py     #   runs inside the image: whole or chunked passes, per-token
│                            #   probe scores, early stop after the last needed layer
├── reports/                 # one report per setup run / step, figures in reports/figures/
└── tools/
    ├── smoke_completion.py  # one LiteLLM call with the repo's tools (setup S3)
    ├── run_trajectory.py    # one trajectory, hard timeout, server.json + summary.json
    ├── teacher_force.py     # paper replies' log-probs under our server vs our samples
    ├── log_prefix_check.py  # per-agent prefix check on a run's llm_calls.jsonl
    ├── replay_prepare.py    # rebuild a prompt, pick token positions for replay
    ├── resample.py          # resample verdicts from fixed histories, for step 4 (plan unless --go)
    ├── setup_s6.sh          # setup S6 in one GPU start
    ├── step1_cache/         # step 1: rebuild comm prompts, positions, index; check and pack the download
    ├── step2/               # step 2: probe at A, baselines, null, control, follow-ups
    ├── step3/               # step 3: probe traced through the exchange, probe at C, transfer
    ├── step5/               # step 5: agreement lines removed from agreement-onset prompts
    ├── step6/               # step 6: probes, jobs, verdict statements, analysis of the projections
    ├── step4_5/             # steps 4-5: flips, call list, vLLM resampling session, analysis
    ├── paper_prompts/       # rebuild and render the paper's verdict calls
    └── requirements.txt     # analysis venv (tools/.venv), torch-free: tokenizer, probes, plots
```

## Setup (2026-10-23)

Setup steps S1–S7 (serving, logging, rendering, timing). Steps 1–6 below are the experiments.

| Setup | Status |
| --- | --- |
| S1. Modal vLLM server | done; boots at 262,144 context in ~2.5 min, KV cache 310,575 tokens |
| S2. Environment, `.env` | done |
| S3. LiteLLM smoke test (tools, `tool_choice=required`) | passed: tool call returned, `reasoning_content` populated |
| S4. Logging | `run.json` written after every episode; per-call wire log added |
| S5. Prompt rendering | on the paper's data: 1,000/1,000 rebuilt verdict prompts match logged `input_tokens`; earlier thinking is dropped by the template; history is a strict prefix, prompt + own reply is not |
| S6. Boot, teacher forcing, timing | 5 paper verdict replies score −0.256/token vs −0.223 for our own samples (no token below −10); 2 episodes in 6.0 + 5.3 min, 0 failures, no pruning; ~$0.40 per early episode |
| S7. Overnight: 2 × 10-episode trajectories in parallel | sequence 2 complete (37 min, mutual ACCEPT from episode 4); sequence 1 crashed in episode 1 on a 64k-token degenerate reply (1,882 repeated tool calls). See `reports/2026-10-23-overnight.md` |

## Experiments: steps 1–6

On the paper's 50 Qwen 3.6 main trajectories and their judge labels.

| Step | Status |
| --- | --- |
| 1. Activation cache | done 2026-10-23: 200 passes (per run and agent: the last episode's communication-phase history and its last verdict prompt, each with its own phase's tool list), 14.26M tokens, whole passes (longest 93,645 tokens, peak 69.0 GiB); residual stream after layers 3, 7, …, 63 at 32,834 positions (A, A_call, every message end, C, verdict, feedback, reflection start/end, runner notices), 5.0 GiB bf16; index in `index.jsonl`. 39.6 H100-minutes, ~$2.61. Modal Volume `qwen36-activations:/cache-v1` |
| 2. Probe at A | done (Mac): own verdict from A, 348 onset/before-onset rows, trajectory-grouped nested CV; selected-layer AUC 0.888 ± 0.023 vs feedback-features baseline 0.896 ± 0.020 (probe better in 9/25 folds), episode number 0.871, relaxation flag 0.818, shuffled null 0.504; positive control (previous reward) 0.924; probe + feedback 0.903 vs feedback 0.896 (+0.007, 19/25 folds), at A_call 0.910 (+0.013, 19/25). See `reports/2026-10-23-step2-probe-a.md` |
| 3. Message-by-message probe, probe at C | done (Mac): per-position probes at A, each of 10 message slots and C, out-of-fold per row with the feedback baseline as reference; the probe first beats the feedback baseline at C (0.956 vs 0.905, +0.051 [0.030, 0.071]), overall and on both paths; no message slot does. C probe 0.953 ± 0.018; A↔C transfer partial (A→C 0.70, C→A 0.74); A_call's edge over A sits with the first speaker. Additions: last message vs C +0.048 [0.027, 0.069] (Alice +0.081, Bob +0.017 n.s.); C probe (layer 47) traced everywhere is flat until C, largest rise at C on 36/36 agreement onsets; agents with previous verdict reject (210 rows): first beats feedback at C (+0.110), A probe 0.774 vs feedback 0.784, A_call + feedback +0.038. See `reports/2026-10-23-step3-trace-and-c.md` |
| 4. Resampling session (vLLM) | done 2026-10-24 with step 5's resampling: 830 requests, 11,680 samples, no capped replies, 44 invalid (task-phase tool calls on the completions endpoint); ≈134 H100-min, ≈$8.80 (cap raised to $15). Typical calls are decided at C: resampling reproduces the saved verdict 91% and tracks the C probe (ρ 0.87); "confidently wrong" probe calls follow the probe 9/10; reverted rejects and flips settle gradually through the thinking. See `reports/2026-10-24-step4-5-resampling.md` |
| 5. Agreement lines removed | edited prompts (`tools/step5/`); C-probe score at C barely moves (step 6); variant B resampled at C: share ACCEPT 0.927 (11 paired calls 0.847 vs 0.966 unedited) |
| 6. Projections session (transformers) | done 2026-10-23 (Volume `qwen36-activations:/step6`): 388 verdict calls + own thinking and 72 edited prompts, C probe (layer 47) per token, 27.0 H100-min (~$1.78). ACCEPT vs REJECT separation: C 0.942, first tenths of thinking 0.86 (fresh probe), last tenth 0.99; lock-in median at 92% of the thinking, about where the text first states the verdict; agreement lines removed barely move the C score (0.976 → 0.940); reverted calls often lean accept at C and reject in the thinking. See `reports/2026-10-23-step6-projections.md` |

GPU spend to date: about 455–490 H100-minutes (~$30–32), including one failed 31-minute start and a session-2 overrun (see `reports/2026-10-24-session2-and-steering-pilot.md`).
Run outputs (`agent-collusion/results/`) are git-ignored.

### Session 2 and steering pilot (2026-10-24)

On the random group the feedback baseline predicts resampled shares as well as the C probe
(ρ 0.85 vs 0.87); the probe only wins on hard cases (0.74 vs −0.09). Telling an agent its previous
verdict was right removes most onset accepts (0.88 → 0.33). With thinking off, flips mostly
reject (0.69 → 0.37). Rerunning the communication phase from A with the peer replayed gives
about the same shares as resampling from C (ρ 0.80). Steering pilot (3 calls, 4 samples):
the C direction pushes one hard case to reject, random directions are inconclusive; the
"ablation" condition was a push toward accept (fix: mean-ablation). See
`reports/2026-10-24-session2-and-steering-pilot.md`.

### Next (each GPU job waits for approval)

- **D.** Mid-layer rerun of the thinking projections (a layer in 23–31), fresh probes per tenth,
  vectors saved at a newline token in each tenth (transformers).
- **Steering main run** (cap $10): 10 hard cases, 5 settled accepts, 3 settled rejects;
  proposed as one batched generate per call with per-row steering and mean-ablation.

## Source

`agent-collusion/` is a copy of
[SALT-NLP/agent-collusion](https://github.com/SALT-NLP/agent-collusion) at commit
`5afb2c595aa3eff6b9b829a032f4f117b36ee54d` (2026-09-23), from Shi, Zhang & Yang (2026),
*Emergent Collusion in Long-Horizon LLM Agent Interaction*. It is MIT-licensed; its
license is kept at [agent-collusion/LICENSE](agent-collusion/LICENSE). Every change made
for this project is listed below.

### Changes to agent-collusion

1. **Per-call wire log** (`experiments/llm.py`, enabled in `experiments/runner.py`,
   noted in `run_config.llm_call_log` by `experiments/output.py`). Each run directory gets
   `llm_calls.jsonl` next to `run.json`: for every LiteLLM call, a `request` record with
   the exact JSON body sent to the server (messages including replayed
   `reasoning_content`, tools, `tool_choice`, sampling), then a `response` record with the
   raw server response before LiteLLM's conversion (including vLLM's `reasoning` field),
   or an `error` record. Records are fsync'd as each call happens, so they survive a run
   that dies mid-episode. No credentials or headers are written. Writes are serialised
   with a lock so concurrent callers (`tools/resample.py`) cannot interleave lines.

## Model and serving

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, bf16, no quantisation |
| Server | vLLM 0.31.0 (`vllm/vllm-openai:v0.31.0` pinned by digest) on one Modal H100 80 GB |
| Served name | `qwen3.6-27b` (LiteLLM route `openai/qwen3.6-27b`) |
| Parsers | reasoning `qwen3`, tool calls `qwen3_coder` with auto tool choice |
| `--max-model-len` | 262,144, native (64,000 output + up to 198,144 prompt tokens; the paper's Qwen 3.6 prompts reach 94,483) |
| `--max-num-seqs` | 64 (default 1024 exceeds the 361 Mamba state blocks that fit) |
| Sampling | temperature 1.0, max output tokens 64,000 (paper); top_p 0.95, top_k 20 from the model's generation config |

### Deviations from the paper's setup

1. **Serving engine version.** We serve with vLLM 0.31.0. The paper does not record its
   server, but none of its 13,037 Qwen 3.6 usage records has
   `completion_tokens_details`, which vLLM reports for reasoning models from v0.28.0
   (2026-08-26) on, with the same litellm 1.94.1 on both sides. Their server was most
   likely vLLM 0.19–0.27 or another engine. Prompt rendering is unaffected: our template
   reproduces their logged `input_tokens` for 1,000 of 1,000 rebuilt verdict calls
   (`tools/paper_prompts/`). Kernels, sampling code and tool-call parsing may differ.
2. **Model revision is inferred, not recorded.** Their runs fall between 2026-07-30
   (litellm 1.94.1) and 2026-09-21 (code release). Every Hugging Face revision from
   `a1bdd8d` (2026-04-22) to `6a9e13bd` (2026-04-24, still the latest) has identical
   weights, tokenizer, template and configs; we pin `6a9e13bd`.

### Choices in the analysis (not in the paper)

1. **Activations come from transformers, not from the serving engine.** We use transformers 5.17
   with flash-linear-attention 0.5.2 for the Gated DeltaNet layers, inside the vLLM image (bf16,
   whole passes). The prompts are the paper's, rebuilt token for token. The C vectors from two
   separate runs (steps 1 and 6) are identical.
2. **Positions.** Earlier thinking is dropped from the history by the chat template, so each
   saved position equals what the model computed in the agent's later real calls.
   - Comm-phase positions are read with the comm-phase tool list, and C with the verdict-phase
     one (the difference is small: median cosine 0.9997).
   - C is the `\n` after `<|im_start|>assistant`, the last token shared by the verdict call's
     prompt and all later prompts.
3. **Resampling (steps 4–5).** Output is capped at 24,000 tokens (the paper used 64,000); no
   sample reached the cap.
   - Full verdicts and edited prompts use the recorded chat request with the forced
     `submit_verdict`.
   - Continuations from partial thinking use the raw completions endpoint, which has no forced
     tool call. The verdict is read from the model's own tool-call text, and 44 of 9,472 such
     samples called a task-phase tool instead and are counted as invalid.
4. **Probes.** L2 logistic regression with cross-validation grouped by trajectory (5 folds × 5
   repeats). The L2 strength and the layer are chosen inside the training folds. Scores are
   out of fold.

Secrets live in Modal (`huggingface`, `vllm-api-key`) and in `agent-collusion/.env`,
which is git-ignored.
