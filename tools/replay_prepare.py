"""Build replay jobs: rebuild a prompt exactly as vLLM saw it, pick token positions to probe.

Sources (one per invocation):
  --calls <run_dir>          requests from our llm_calls.jsonl (select with --seq)
  --wire <paper_wire.jsonl>  rebuilt paper verdict calls (select with --pick run:episode:agent)

Anchors, resolved to token indices in the rendered prompt (+ reply with --with-reply):
  last                 final prompt token (the position that predicts the reply's first token)
  last_user_end        last token of the final user message's content
  user_ends            last content token of every user turn
  assistant_opens      the "assistant" token after each <|im_start|>
  reply                every reply token (needs --with-reply)
  "substr:<text>"      last token of the last occurrence of <text>

    tools/.venv/bin/python tools/replay_prepare.py --wire paper_wire.jsonl \
        --pick rep001:4:alice --with-reply --anchors last last_user_end reply \
        --out results/replay/jobs.jsonl
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

END = "<|im_end|>"


def token_at(offsets, char_index):
    """Index of the token covering character position ``char_index``."""
    for i, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return i
    raise ValueError(f"no token covers character {char_index}")


def resolve(anchor, text, offsets, n_prompt, n_total):
    if anchor == "last":
        return [n_prompt - 1]
    if anchor == "reply":
        if n_total == n_prompt:
            raise ValueError("anchor 'reply' needs --with-reply")
        return list(range(n_prompt, n_total))
    if anchor in ("last_user_end", "user_ends"):
        ends, start = [], 0
        while (i := text.find("<|im_start|>user\n", start)) != -1:
            close = text.index(END, i)
            if close <= offsets[n_prompt - 1][1]:
                ends.append(token_at(offsets, close - 1))
            start = close
        return ends[-1:] if anchor == "last_user_end" else ends
    if anchor == "assistant_opens":
        out, start = [], 0
        while (i := text.find("<|im_start|>assistant", start)) != -1 and i < offsets[n_prompt - 1][1]:
            out.append(token_at(offsets, i + len("<|im_start|>")))
            start = i + 1
        return out
    if anchor.startswith("substr:"):
        needle = anchor[len("substr:"):]
        i = text.rfind(needle, 0, offsets[n_prompt - 1][1])
        if i == -1:
            raise ValueError(f"{needle!r} not in prompt")
        return [token_at(offsets, i + len(needle) - 1)]
    raise ValueError(f"unknown anchor {anchor}")


def load_sources(args):
    if args.wire:
        rows = {(r["run"].split(".")[0], r["ep"] + 1, r["agent"]): r
                for r in map(json.loads, open(args.wire)) if "skip" not in r}
        for pick in args.pick:
            run, ep, agent = pick.split(":")
            row = rows[(run, int(ep), agent)]
            yield f"paper:{pick}", row["body"], row.get("reply")
    else:
        run_dir = Path(args.calls)
        records = [json.loads(l) for l in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
        responses = {r["litellm_call_id"]: r["body"] for r in records if r["event"] == "response"}
        for r in records:
            if r["event"] == "request" and r["seq"] in args.seq:
                resp = responses.get(r["litellm_call_id"])
                reply = None
                if isinstance(resp, dict) and resp.get("choices"):
                    m = resp["choices"][0]["message"]
                    reply = {"role": "assistant", "content": m.get("content") or "",
                             "tool_calls": m.get("tool_calls"),
                             "reasoning": m.get("reasoning") or m.get("reasoning_content")}
                yield f"ours:{run_dir.name}:seq{r['seq']}", r["body"], reply


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wire")
    source.add_argument("--calls")
    parser.add_argument("--pick", nargs="*", default=[])
    parser.add_argument("--seq", nargs="*", type=int, default=[])
    parser.add_argument("--with-reply", action="store_true")
    parser.add_argument("--anchors", nargs="+", default=["last"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for job_id, body, reply in load_sources(args):
            conv, tools = vllm_conversation(body["messages"]), vllm_tools(body.get("tools"))
            prompt = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False)
            text = prompt
            if args.with_reply:
                if reply is None:
                    raise ValueError(f"{job_id} has no reply")
                full = tok.apply_chat_template(conv + vllm_conversation([reply]), tools=tools,
                                               add_generation_prompt=False, tokenize=False)
                if not full.startswith(prompt):
                    raise ValueError(f"{job_id}: reply does not extend prompt")
                text = full[: full.index(END, len(prompt)) + len(END)]
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if enc["input_ids"][:n_prompt] != tok(prompt, add_special_tokens=False)["input_ids"]:
                raise ValueError(f"{job_id}: prompt tokens are not a prefix")
            positions = {a: resolve(a, text, enc["offset_mapping"], n_prompt, len(enc["input_ids"]))
                         for a in args.anchors}
            job = {
                "id": job_id,
                "token_ids": enc["input_ids"],
                "n_prompt_tokens": n_prompt,
                "positions": positions,
                "sha256_token_ids": hashlib.sha256(json.dumps(enc["input_ids"]).encode()).hexdigest(),
                "model_revision": pins.MODEL_REVISION,
            }
            f.write(json.dumps(job) + "\n")
            print(job_id, "tokens", len(enc["input_ids"]), "prompt", n_prompt,
                  {a: (len(p), p[:3]) for a, p in positions.items()})


if __name__ == "__main__":
    main()
