# Handshake for the cameras: reading and steering emergent model collusion

_Apart AI Collusion Research Sprint (Oct 23–25, 2026)._
Solo entry. [PDF report](https://github.com/star2vec/apart/blob/main/handshake-for-the-cameras.pdf).

As AI agents increasingly collaborate on solving tasks and checking each other’s work, the risk of collusion rises. Shi et al. showed that this behaviour can emerge on its own, with or without an explicit agreement in the transcript.  We locate the collusion decision by reading the upcoming verdict from the internal activations at fixed points of an episode, verify that reading by repeatedly restarting generation from that point, edit the transcript to find what causes the verdict, and push along the readable direction to test whether it is causal. The decision favoring ACCEPT emerges from the previous feedback, before the message exchange, and while ACCEPT needs the discussion to happen, it does not depend on the agreement. The verdict becomes readable in the activations only once the model is asked for it, before any reasoning, and on hard calls this reading predicts the output better than the previous feedback and than the full chain of thought. Pushing along the collusion direction changes the verdict in either direction, but only when sustained throughout the reasoning, which argues for the steered verdict, and removing the direction leaves the decision as it was. The handshake is for the cameras: by the time the agents agree, the decision is already settled, and it can be read and moved before a word of reasoning is written.

Main contributions:
1. A timeline of the decision to collude. It forms from the previous episode’s feedback before the agents talk, and it depends on the model discussing, not on explicit agreement.
2. A probe at the decision prompt. It is validated against resampled behaviour and, on hard calls, it predicts the verdict better than the previous feedback and the model’s full chain of thought.
3. Causal evidence that the readable direction is steerable and can change the verdict in both directions, but only when pushed through the reasoning, and ablating it leaves the decision as it was.
4. The decision can also be read and changed by a general approval/rejection direction fitted on unrelated review tasks and never on this setup.
5. Activation cache, per-call logging, prompt reconstruction, and resampling for a public emergent collusion environment, released with the code.
   
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
