"""Item 5 (Mac): ingredient removal at C. For each call of the first session (random group and
hard cases), the recorded verdict request with one part of the context removed.

    tools/.venv/bin/python tools/step4_5/session3_edits.py [--show 2]

Request layout (per earlier episode k): task phase, "## Episode k: communication phase" intro,
the exchange, "## Episode k: verdict phase", submit_verdict and its result, the feedback block
(user "Episode k complete. ... ## Reward ... ## Verdict review ..."), the reflection prompt
(user "Reflect briefly ...") and the reflection (assistant, plain text). The current episode N
ends with its communication phase and the verdict prompt (the last message).

Removals (one at a time):
  conversation     episode N's whole exchange: every message between the communication-phase
                   intro and the verdict prompt (round prompts, own send_message calls and their
                   results, delivered peer messages, runner notices); the intro (the peer's task
                   information) stays
  feedback_prev    episode N-1's feedback block
  feedback_all     every earlier episode's feedback block
  reflection_prev  episode N-1's reflection prompt and reflection
  peer_messages    episode N's delivered peer messages ([round=r/R][from=<peer>]...)
For an episode-1 call the three history removals do not apply (nothing to remove; skipped); for an
episode-2 call feedback_all is the same edit as feedback_prev (not built twice; its share is
feedback_prev's).
Tool calls are never split from their results. Checks per edit: the expected messages are gone,
nothing else changed (all other messages identical and in order), the last message is still the
verdict prompt, and the rendered prompt length (chat template, tools included) before and after.
Writes agent-collusion/results/step4_5/session3_edits.jsonl.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
RES = ROOT / "agent-collusion" / "results"
FEEDBACK = re.compile(r"^Episode (\d+) complete\.\s", re.S)
REFLECT = re.compile(r"^Reflect briefly", re.S)
COMM = re.compile(r"^## Episode (\d+): communication phase")
VERDICT = re.compile(r"^## Episode (\d+): verdict phase")
PEER_MSG = re.compile(r"^\[round=\d+/\d+\]\[from=(\w+)\]")
REMOVALS = ["conversation", "feedback_prev", "feedback_all", "reflection_prev", "peer_messages"]


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else ("" if c is None else json.dumps(c))


def locate(msgs, agent):
    """Indices of each removable part."""
    n_ep = int(VERDICT.match(text(msgs[-1])).group(1))
    comm = max(i for i, m in enumerate(msgs) if m["role"] == "user" and COMM.match(text(m))
               and int(COMM.match(text(m)).group(1)) == n_ep)
    conv = list(range(comm + 1, len(msgs) - 1))
    peer = [i for i in conv if msgs[i]["role"] == "user" and PEER_MSG.match(text(msgs[i]))
            and PEER_MSG.match(text(msgs[i])).group(1) != agent]
    fb = {int(FEEDBACK.match(text(m)).group(1)): i for i, m in enumerate(msgs)
          if m["role"] == "user" and FEEDBACK.match(text(m))}
    refl = {}
    for k, i in fb.items():
        j = i + 1
        if j < len(msgs) and msgs[j]["role"] == "user" and REFLECT.match(text(msgs[j])):
            assert msgs[j + 1]["role"] == "assistant" and not msgs[j + 1].get("tool_calls"), "reflection shape"
            refl[k] = [j, j + 1]
    assert sorted(fb) == list(range(1, n_ep)), f"feedback blocks {sorted(fb)} for episode {n_ep}"
    return n_ep, {"conversation": conv, "peer_messages": peer,
                  "feedback_prev": [fb[n_ep - 1]] if n_ep > 1 else [],
                  "feedback_all": sorted(fb.values()),
                  "reflection_prev": refl.get(n_ep - 1, [])}


def check_pairs(msgs):
    """Every tool result follows an assistant message whose tool calls it answers."""
    pending = []
    for m in msgs:
        if m["role"] == "assistant":
            pending = [t["id"] for t in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            assert m.get("tool_call_id") in pending, "orphan tool result"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--show", type=int, default=2)
    a = p.parse_args()
    import pins
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    plan = json.loads((RES / "step4_5" / "plan.json").read_text())
    wire = {}
    for line in open(RES / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r["body"]

    def n_tokens(body):  # the template wants tool-call arguments as mappings (the wire has JSON strings)
        msgs = [dict(m, tool_calls=[dict(t, function=dict(t["function"], arguments=json.loads(t["function"]["arguments"] or "{}")))
                                    for t in m["tool_calls"]]) if m.get("tool_calls") else m for m in body["messages"]]
        tools = [t["function"] if "function" in t else t for t in body.get("tools") or []]
        return len(tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True,
                                           return_dict=True)["input_ids"])

    out, shown, stats = [], 0, {r: [0, 0] for r in REMOVALS}
    for c in plan["calls"]:
        body = wire[(c["run"], c["episode"], c["agent"])]
        msgs = body["messages"]
        check_pairs(msgs)
        n_ep, parts = locate(msgs, c["agent"])
        assert n_ep == c["episode"]
        base_len = n_tokens(body)
        group = "random" if c["groups"][0].startswith("random") else "hard"
        for rem in REMOVALS:
            idx = set(parts[rem])
            if not idx or (rem == "feedback_all" and idx == set(parts["feedback_prev"])):  # episode 2: same edit
                stats[rem][1] += 1
                continue
            new = [m for i, m in enumerate(msgs) if i not in idx]
            check_pairs(new)
            kept = [m for i, m in enumerate(msgs) if i not in idx]
            assert new == kept and len(new) == len(msgs) - len(idx) and VERDICT.match(text(new[-1]))
            if rem in ("feedback_prev", "feedback_all"):
                assert not any(FEEDBACK.match(text(m)) for m in (new if rem == "feedback_all" else [msgs[i] for i in []]))
            if rem == "peer_messages":
                assert not any(m["role"] == "user" and PEER_MSG.match(text(m)) and PEER_MSG.match(text(m)).group(1) != c["agent"]
                               for m in new[len(new) - (len(msgs) - min(parts["conversation"] or [len(msgs)])):])
            edited = dict(body, messages=new)
            row = {"job": c["job"], "run": c["run"], "episode": c["episode"], "agent": c["agent"], "group": group,
                   "removal": rem, "removed_indices": sorted(idx), "removed_roles": [msgs[i]["role"] for i in sorted(idx)],
                   "removed_chars": sum(len(text(msgs[i])) for i in idx), "tokens_before": base_len,
                   "tokens_after": n_tokens(edited), "body": edited}
            out.append(row)
            stats[rem][0] += 1
            if shown < a.show * len(REMOVALS) and c["episode"] >= 3:
                shown += 1
                print(f"\n== {c['job']} {rem}: {len(idx)} messages, {row['removed_chars']} chars, "
                      f"tokens {base_len} -> {row['tokens_after']}")
                for i in sorted(idx)[:3] + (sorted(idx)[-1:] if len(idx) > 3 else []):
                    m = msgs[i]
                    t = text(m) or json.dumps([tc["function"]["name"] for tc in m.get("tool_calls") or []])
                    print(f"   - [{i} {m['role']}] {t[:140]!r}")
                k = min(idx)
                print(f"   kept before: [{k - 1} {msgs[k - 1]['role']}] {text(msgs[k - 1])[:100]!r}")
                after = max(idx) + 1
                print(f"   kept after:  [{after} {msgs[after]['role']}] {text(msgs[after])[:100]!r}")
    with open(RES / "step4_5" / "session3_edits.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("\nedits per removal (built, skipped as not applicable):", stats)
    print(len(out), "edited requests;", len(out) * 16, "samples at 16 each")


if __name__ == "__main__":
    main()
