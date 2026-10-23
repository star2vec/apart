# Paper prompt reconstruction

Checks against the paper's released Qwen 3.6 trajectories
(`SALT-NLP/agent-collusion` dataset, revision `2eba8f3771e8fbcc0f49f6cbbfd2111b939a117a`,
`traces/main/main__qwen-3-6-27b/rep001..050.json.gz`). No GPU needed.

| Script | Venv | Does |
| --- | --- | --- |
| `paper_context.py <traces>` | any Python | prompt/output token sizes per call and episode |
| `rebuild_wire.py <traces> <out.jsonl>` | `agent-collusion/.venv` | rebuilds every first-attempt verdict request from saved history plus repo tool code, passes it through LiteLLM to a local capture server, writes the wire bodies |
| `render_wire.py <wire.jsonl> <tokenizer_dir> [prompt_out]` | transformers 5.17.0 | applies vLLM 0.31's message preprocessing and the chat template, compares token counts with the logged `input_tokens` |
| `prefix_check.py <wire.jsonl> <tokenizer_dir> <this_dir>` | transformers 5.17.0 | is each agent's verdict prompt in episode k a text prefix of episode k+1's |

`<tokenizer_dir>` holds the tokenizer files and `chat_template.jinja` of `Qwen/Qwen3.6-27B`
at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

Results (2026-10-23): rendered token count equals the logged `input_tokens` for 1,000 of
1,000 verdict calls; 900 of 900 consecutive-episode verdict prompts are strict prefixes;
largest prompt 94,483 tokens, 1,422 of 13,037 calls above 67,072.
