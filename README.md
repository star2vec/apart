# Apart AI Collusion Research Sprint (Oct 23–25, 2026)

Solo entry. Runs the two-agent collusion environment against a self-hosted
Qwen3.6-27B served with vLLM on Modal.

## Layout

```text
.
├── agent-collusion/   # Environment (plain copy, see Source below)
└── server/            # Modal apps: weight download and vLLM endpoint
    ├── pins.py              # Model revision, vLLM version, max-model-len, GPU
    ├── download_weights.py  # CPU only: fills the weights Volume, verifies symlinks
    └── serve_qwen.py        # GPU: OpenAI-compatible vLLM endpoint
```

## Source

`agent-collusion/` is an unmodified copy of
[SALT-NLP/agent-collusion](https://github.com/SALT-NLP/agent-collusion) at commit
`5afb2c595aa3eff6b9b829a032f4f117b36ee54d` (2026-09-23), from Shi, Zhang & Yang (2026),
*Emergent Collusion in Long-Horizon LLM Agent Interaction*. It is MIT-licensed; its
license is kept at [agent-collusion/LICENSE](agent-collusion/LICENSE). Changes made for
this project are listed below.

### Changes to agent-collusion

1. **Per-call wire log** (`experiments/llm.py`, enabled in `experiments/runner.py`,
   noted in `run_config.llm_call_log` by `experiments/output.py`). Each run directory gets
   `llm_calls.jsonl` next to `run.json`: for every LiteLLM call, a `request` record with
   the exact JSON body sent to the server (messages including replayed
   `reasoning_content`, tools, `tool_choice`, sampling), then a `response` record with the
   raw server response before LiteLLM's conversion (including vLLM's `reasoning` field),
   or an `error` record. Records are fsync'd as each call happens, so they survive a run
   that dies mid-episode. No credentials or headers are written.

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

Secrets live in Modal (`huggingface`, `vllm-api-key`) and in `agent-collusion/.env`,
which is git-ignored.
