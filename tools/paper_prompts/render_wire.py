"""Stage B: apply vLLM 0.31's message preprocessing and the chat template, count tokens.

Mirrors vllm/entrypoints/chat_utils.py (_parse_chat_message_content, _postprocess_messages)
and the tool serialisation in chat_completion/protocol.py at v0.31.0.
Run with the render venv: python render_wire.py <wire.jsonl> <tokenizer_dir> [example_out]
"""

import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def vllm_conversation(messages):
    conv = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
        msg = {"role": m["role"], "content": content if content is not None else ""}
        if m["role"] == "assistant":
            if m.get("tool_calls") is not None:
                calls = json.loads(json.dumps(m["tool_calls"]))
                for c in calls:
                    fn = c.get("function", {})
                    args = fn.get("arguments")
                    fn["arguments"] = json.loads(args) if args else {}
                msg["tool_calls"] = calls
            if m.get("reasoning") is not None:
                msg["reasoning"] = m["reasoning"]
                msg["reasoning_content"] = m["reasoning"]
        elif m["role"] == "tool" and "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        if isinstance(m.get("name"), str):
            msg["name"] = m["name"]
        conv.append(msg)
    return conv


def vllm_tools(tools):
    out = []
    for t in tools or []:
        fn = t["function"]
        out.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description"), "parameters": fn.get("parameters")}})
    return out



def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    tok = AutoTokenizer.from_pretrained(sys.argv[2])
    stats, diffs, example = Counter(), [], None
    for r in rows:
        if "skip" in r:
            stats["skipped:" + r["skip"]] += 1
            continue
        body = r["body"]
        prompt = tok.apply_chat_template(vllm_conversation(body["messages"]), tools=vllm_tools(body.get("tools")),
                                         add_generation_prompt=True, tokenize=False)
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        d = n - r["logged_input_tokens"]
        diffs.append(d)
        stats["exact" if d == 0 else "mismatch"] += 1
        if example is None and r["ep"] == 3 and r["agent"] == "alice":
            example = (r, prompt, n)

    print("verdict calls:", dict(stats))
    print("token diff (rendered - logged) distribution:", Counter(diffs).most_common(8))
    if example:
        r, prompt, n = example
        print(f"\nexample: {r['run']} episode {r['ep'] + 1} alice verdict: rendered {n} tokens, logged {r['logged_input_tokens']}")
        print("body keys:", sorted(r["body"]), "| tool_choice:", r["body"].get("tool_choice"),
              "| tools:", [t["function"]["name"] for t in r["body"].get("tools", [])])
        print("history messages:", r["history_len"],
              "| assistant turns:", sum(m["role"] == "assistant" for m in r["body"]["messages"]),
              "| with reasoning:", sum(bool(m.get("reasoning")) for m in r["body"]["messages"]))
        print("'<think>' blocks in rendered prompt:", prompt.count("<think>"),
              "| non-empty:", prompt.count("<think>\n") - prompt.count("<think>\n\n</think>"))
        if len(sys.argv) > 3:
            Path(sys.argv[3]).write_text(prompt)
            print("rendered prompt written to", sys.argv[3])


if __name__ == "__main__":
    main()
