# Apart AI Collusion Research Sprint (Oct 23–25, 2026)

Solo entry. Runs the two-agent collusion environment against a self-hosted
Qwen3.6-27B served with vLLM on Modal.

## Layout

```text
.
├── agent-collusion/         # environment (copy, see Source below)
├── server/                  # Modal apps
│   ├── pins.py              # model revision, vLLM version, max-model-len, GPU
│   ├── download_weights.py  # CPU only: fills the weights Volume, verifies symlinks
│   ├── serve_qwen.py        # GPU: OpenAI-compatible vLLM endpoint
│   ├── replay_qwen.py       # GPU: transformers replay, saves residual-stream vectors
│   └── replay_worker.py     #   runs inside the image under its own torch
└── tools/
    ├── smoke_completion.py  # one LiteLLM call with the repo's tools (step 3)
    ├── run_trajectory.py    # one trajectory, hard timeout, server.json + summary.json
    ├── teacher_force.py     # paper replies' log-probs under our server vs our samples
    ├── log_prefix_check.py  # per-agent prefix check on a run's llm_calls.jsonl
    ├── replay_prepare.py    # rebuild a prompt, pick token positions for replay
    ├── resample.py          # experiment 1: resample verdicts from fixed histories (plan unless --go)
    ├── step6.sh             # step 6 in one GPU start
    ├── cache/               # activation cache job: rebuild comm prompts, positions, index
    ├── paper_prompts/       # rebuild and render the paper's verdict calls
    └── requirements.txt     # analysis venv (tools/.venv), torch-free
```

## Progress (2026-10-23, after the overnight run)

| Step | Status |
| --- | --- |
| 1. Modal vLLM server | done; boots at 262,144 context in ~2.5 min, KV cache 310,575 tokens |
| 2. Environment, `.env` | done |
| 3. LiteLLM smoke test (tools, `tool_choice=required`) | passed: tool call returned, `reasoning_content` populated |
| 4. Logging | `run.json` written after every episode; per-call wire log added |
| 5. Prompt rendering | on the paper's data: 1,000/1,000 rebuilt verdict prompts match logged `input_tokens`; earlier thinking is dropped by the template; history is a strict prefix, prompt + own reply is not |
| 6. Boot, teacher forcing, timing | 5 paper verdict replies score −0.256/token vs −0.223 for our own samples (no token below −10); 2 episodes in 6.0 + 5.3 min, 0 failures, no pruning; ~$0.40 per early episode |
| 7. Overnight: 2 × 10-episode trajectories in parallel | sequence 2 complete (37 min, mutual ACCEPT from episode 4); sequence 1 crashed in episode 1 on a 64k-token degenerate reply (1,882 repeated tool calls). See `reports/2026-10-23-overnight.md` |

GPU spend to date: about 105 H100-minutes (~$7.00), including one failed 31-minute start.
Run outputs (`agent-collusion/results/`) are git-ignored.

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

Secrets live in Modal (`huggingface`, `vllm-api-key`) and in `agent-collusion/.env`,
which is git-ignored.
