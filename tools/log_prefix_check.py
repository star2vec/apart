"""Step-5 check on a run's llm_calls.jsonl: render every request as vLLM 0.31 does and ask,
per agent and call to call, whether context is a strict prefix.

For consecutive requests t, t+1 of one agent it reports whether
  history(t) (the rendered messages, without the generation opener) is a text prefix of
  prompt(t+1), and
  prompt(t) + reply(t) (the model's own output, thinking included) is a prefix of prompt(t+1),
and, when not, why: tool list changed, earlier thinking dropped, or history rewritten.
It also compares the rendered token count with the server's reported prompt_tokens.

    tools/.venv/bin/python tools/log_prefix_check.py <run_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def reply_message(response_body):
    message = response_body["choices"][0]["message"]
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls"),
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
    }


def main() -> None:
    run_dir = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    system = json.loads((run_dir / "run.json").read_text())["run_config"]["prompt_texts"]["system_prompts"]
    records = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
    responses = {r["litellm_call_id"]: r for r in records if r["event"] == "response"}

    calls = {"alice": [], "bob": []}
    token_check = Counter()
    for r in records:
        if r["event"] != "request":
            continue
        body = r["body"]
        agent = next(a for a, s in system.items() if body["messages"][0].get("content") == s)
        conv, tools = vllm_conversation(body["messages"]), vllm_tools(body.get("tools"))
        prompt = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False)
        history = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=False, tokenize=False)
        resp = responses.get(r["litellm_call_id"], {}).get("body")
        with_reply = None
        if isinstance(resp, dict) and resp.get("choices"):
            full = tok.apply_chat_template(conv + vllm_conversation([reply_message(resp)]), tools=tools,
                                           add_generation_prompt=False, tokenize=False)
            with_reply = full[: full.index("<|im_end|>", len(prompt)) + len("<|im_end|>")]
            n = len(tok(prompt, add_special_tokens=False)["input_ids"])
            token_check["match" if n == resp["usage"]["prompt_tokens"] else "mismatch"] += 1
        last_user = max(i for i, m in enumerate(body["messages"]) if m["role"] == "user")
        calls[agent].append({"seq": r["seq"], "prompt": prompt, "history": history, "with_reply": with_reply,
                             "tools": [t["function"]["name"] for t in body.get("tools") or []],
                             "messages": body["messages"], "last_user": last_user})

    report = {"rendered_vs_server_prompt_tokens": dict(token_check), "agents": {}}
    for agent, seq in calls.items():
        result, reasons = Counter(), Counter()
        for a, b in zip(seq, seq[1:]):
            result["history prefix" if b["prompt"].startswith(a["history"]) else "history NOT prefix"] += 1
            if a["with_reply"] is None:
                continue
            if b["prompt"].startswith(a["with_reply"]):
                result["prompt+reply prefix"] += 1
                continue
            result["prompt+reply NOT prefix"] += 1
            if a["tools"] != b["tools"]:
                reasons["tool list changed (tools render at the top of the system block)"] += 1
            elif b["messages"][: len(a["messages"])] != a["messages"]:
                reasons["message history rewritten (pruning or reflection copy)"] += 1
            elif b["last_user"] > a["last_user"]:
                reasons["new user message: earlier thinking dropped by the template"] += 1
            else:
                reasons["other"] += 1
        report["agents"][agent] = {"requests": len(seq), "pairs": dict(result), "breaks": dict(reasons)}
    print(json.dumps(report, indent=2))
    (run_dir / "prefix_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
