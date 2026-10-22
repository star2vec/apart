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

None yet.

## Model and serving

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, bf16, no quantisation |
| Server | vLLM 0.31.0 (`vllm/vllm-openai:v0.31.0` pinned by digest) on one Modal H100 80 GB |
| Served name | `qwen3.6-27b` (LiteLLM route `openai/qwen3.6-27b`) |
| Parsers | reasoning `qwen3`, tool calls `qwen3_coder` with auto tool choice |
| `--max-model-len` | 131,072 (64,000 output + up to 67,072 prompt tokens) |
| Sampling | temperature 1.0, max output tokens 64,000 (paper); top_p 0.95, top_k 20 from the model's generation config |

Secrets live in Modal (`huggingface`, `vllm-api-key`) and in `agent-collusion/.env`,
which is git-ignored.
