"""Teacher-forcing check: how likely are the paper's saved verdict replies under our server?

For each selected verdict call rebuilt by tools/paper_prompts/rebuild_wire.py, render the
prompt and the paper's reply with the chat template, send prompt+reply token ids to
/v1/completions with prompt_logprobs, and read the reply tokens' log-probabilities.
As a baseline, sample --samples replies from our server on the same prompt and score
them the same way (vLLM returns raw log-probs for both, before temperature/top-k/top-p).

    tools/.venv/bin/python tools/teacher_force.py --wire <paper_wire_verdict.jsonl> \
        --pick rep001:1:alice rep002:2:bob ... --out results/teacher_force.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from openai import OpenAI  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

END = "<|im_end|>"


def reply_ids(tok, conversation, tools, reply):
    prompt = tok.apply_chat_template(conversation, tools=tools, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(conversation + vllm_conversation([reply]), tools=tools,
                                   add_generation_prompt=False, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("rendered reply does not extend the rendered prompt")
    full = full[: full.index(END, len(prompt)) + len(END)]  # generation stops at <|im_end|>
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("prompt tokens are not a prefix of prompt+reply tokens")
    return p_ids, f_ids[len(p_ids):]


def segment(tok, ids):
    """Split reply token positions into reasoning (before </think>) and the rest."""
    close = tok.convert_tokens_to_ids("</think>")
    cut = ids.index(close) + 1 if close in ids else 0
    return cut


def stats(values):
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4) if values else None,
        "min": round(min(values), 3) if values else None,
        "below_-10": sum(v < -10 for v in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wire", required=True)
    parser.add_argument("--pick", nargs="+", required=True, help="run:episode(1-based):agent")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = dotenv_values(ROOT / "agent-collusion" / ".env")
    client = OpenAI(base_url=env["OPENAI_API_BASE"], api_key=env["OPENAI_API_KEY"], timeout=3600)
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    rows = {(r["run"].split(".")[0], r["ep"] + 1, r["agent"]): r
            for r in map(json.loads, open(args.wire)) if "skip" not in r}

    results = []
    for pick in args.pick:
        run, ep, agent = pick.split(":")
        row = rows[(run, int(ep), agent)]
        body = row["body"]
        conversation, tools = vllm_conversation(body["messages"]), vllm_tools(body.get("tools"))
        p_ids, r_ids = reply_ids(tok, conversation, tools, row["reply"])
        cut = segment(tok, r_ids)

        started = time.time()
        forced = client.completions.create(
            model=pins.SERVED_MODEL_NAME, prompt=p_ids + r_ids, max_tokens=1, temperature=1.0,
            extra_body={"prompt_logprobs": 0},
        )
        # vLLM puts prompt_logprobs on the choice; accept the top level too.
        entries = (forced.choices[0].model_extra or {}).get("prompt_logprobs") or (
            forced.model_extra or {}).get("prompt_logprobs")
        if entries is None:
            raise ValueError("no prompt_logprobs in response: " + forced.model_dump_json()[:400])
        lp = [entries[len(p_ids) + i][str(t)]["logprob"] for i, t in enumerate(r_ids)]

        sampled = client.completions.create(
            model=pins.SERVED_MODEL_NAME, prompt=p_ids, max_tokens=8192, temperature=1.0,
            n=args.samples, logprobs=0,
        )
        own = [c.logprobs.token_logprobs for c in sampled.choices]
        results.append({
            "pick": pick,
            "prompt_tokens_rendered": len(p_ids),
            "prompt_tokens_logged_by_paper": row["logged_input_tokens"],
            "prompt_tokens_our_server": forced.usage.prompt_tokens - len(r_ids),
            "paper_reply": {"all": stats(lp), "reasoning": stats(lp[:cut]), "after_think": stats(lp[cut:]),
                            "lowest": sorted(((round(v, 2), tok.decode([t])) for v, t in zip(lp, r_ids)))[:8]},
            "own_samples": [stats(s) for s in own],
            "own_finish_reasons": [c.finish_reason for c in sampled.choices],
            "wall_s": round(time.time() - started, 1),
        })
        print(json.dumps(results[-1]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"server": pins.SERVED_MODEL_NAME, "revision": pins.MODEL_REVISION,
                               "results": results}, indent=2))


if __name__ == "__main__":
    main()
