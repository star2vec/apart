"""Second vLLM session (Mac prep): requests and cost for tasks 4-6 of the 2026-10-24 plan.

    tools/.venv/bin/python tools/step4_5/session2_build.py [--from-a-samples 8] [--samples 16]

  from_a     random group (40 calls): the studied agent's communication phase replayed from A
             with the peer's saved messages (and runner notices) delivered unchanged, round by
             round as the game does, then the saved verdict prompt; n samples per call
  think_off  random group + hard cases (102 calls): the recorded verdict request with thinking
             disabled (chat_template_kwargs enable_thinking=false), n samples
  feedback   about 20 onset and before-onset calls from the random group (episode >= 2): the
             previous episode's feedback edited (the agent's own verdict review flipped
             right <-> wrong, reward recomputed under the shared-verdict-accuracy scheme),
             resampled at C, n samples

Writes agent-collusion/results/step4_5/session2_plan.json.
"""

import argparse
import gzip
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
TRACES = RES / "paper" / "traces_qwen36_main"
OUT = RES / "step4_5"
HOURLY = 3.95
RIGHT = "reached the right verdict"
WRONG = "reached the wrong verdict"


def flip_feedback(text, agent_name_peer):
    """Flip the agent's own verdict-review line and recompute the shared reward."""
    lines = text.split("\n")
    own_i = next(i for i, l in enumerate(lines) if l.startswith("You reached the "))
    peer_i = next(i for i, l in enumerate(lines) if l.startswith(f"{agent_name_peer} reached the "))
    own_right = RIGHT in lines[own_i]
    lines[own_i] = lines[own_i].replace(RIGHT if own_right else WRONG, WRONG if own_right else RIGHT)
    n_right = (not own_right) + (RIGHT in lines[peer_i])
    rew_i = next(i for i, l in enumerate(lines) if l == "## Reward") + 2
    old_reward = lines[rew_i]
    lines[rew_i] = f"+{10 * n_right}"
    return "\n".join(lines), {"own_review_was": "right" if own_right else "wrong", "reward_was": old_reward,
                              "reward_now": lines[rew_i]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-a-samples", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--feedback-calls", type=int, default=20)
    args = parser.parse_args()
    analysis = {c["job"]: c for c in json.loads((OUT / "analysis.json").read_text())["calls"]}
    plan1 = json.loads((OUT / "plan.json").read_text())
    calls = [c for c in plan1["calls"] if analysis[c["job"]]["A"]]
    random_calls = [c for c in calls if c["groups"][0].startswith("random")]
    wire = {}
    for line in open(RES / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r
    traces = {}

    def trace(run):
        if run not in traces:
            traces[run] = json.load(gzip.open(TRACES / f"{run}.json.gz"))
        return traces[run]

    # ---- from A
    from_a, usage_est = [], []
    for c in random_calls:
        t = trace(c["run"])
        ep = t["results"][c["episode"] - 1]
        msgs = ep["agents"][c["agent"]]["messages"]
        intro = max(i for i, m in enumerate(msgs) if m["role"] == "user"
                    and (m.get("content") or "").startswith(f"## Episode {c['episode']}: communication phase"))
        vprompt = next(i for i in range(intro, len(msgs)) if msgs[i]["role"] == "user"
                       and (msgs[i].get("content") or "").startswith(f"## Episode {c['episode']}: verdict phase"))
        peer = "bob" if c["agent"] == "alice" else "alice"
        sched = []
        for r in range(ep["max_rounds"]):
            ev = [e for e in ep["channel_transcript"] if e["round"] == r and e["sender"] == peer]
            sched.append({"round": r, "message": None if not ev else
                          {"content": ev[0]["content"], "message_type": ev[0].get("message_type", "other")}})
        # paper's own output tokens for this agent's communication + verdict calls (cost basis)
        out_tok = sum(u.get("provider_output_tokens") or 0 for u in ep["llm_usage"]
                      if u["actor"] == c["agent"] and u["phase"] in ("communication", "verdict"))
        n_calls = sum(1 for u in ep["llm_usage"] if u["actor"] == c["agent"] and u["phase"] in ("communication", "verdict"))
        usage_est.append((out_tok, n_calls))
        from_a.append({"job": c["job"], "run": c["run"], "agent": c["agent"], "episode": c["episode"],
                       "type": c["type"], "verdict_saved": c["verdict"], "share_C": analysis[c["job"]]["A"]["share"],
                       "speaks_first": c["agent"] == "alice", "task_type": ep["task_type"],
                       "throttle_start_round": ep["throttle_start_round"], "max_rounds": ep["max_rounds"],
                       "history_at_A": msgs[:intro + 1], "peer_schedule": sched,
                       "verdict_prompt": msgs[vprompt]["content"], "samples": args.from_a_samples,
                       "paper_output_tokens": out_tok, "paper_llm_calls": n_calls,
                       "n_prompt_tokens_at_A": None})
    # ---- thinking off
    think_off = [{"job": c["job"], "groups": c["groups"], "share_C": analysis[c["job"]]["A"]["share"],
                  "samples": args.samples, "n_prompt_tokens": c["n_prompt_tokens"]} for c in calls]
    # ---- feedback edit
    rng = random.Random(1)
    pool = {t: [c for c in random_calls if c["type"] == t and c["episode"] >= 2] for t in ("onset", "before_onset")}
    pick = rng.sample(pool["onset"], min(args.feedback_calls // 2, len(pool["onset"]))) + \
        rng.sample(pool["before_onset"], min(args.feedback_calls - args.feedback_calls // 2, len(pool["before_onset"])))
    feedback = []
    for c in pick:
        body = json.loads(json.dumps(wire[(c["run"], c["episode"], c["agent"])]["body"]))
        prev = c["episode"] - 1
        idx = max(i for i, m in enumerate(body["messages"]) if m["role"] == "user"
                  and str(m.get("content", "")).startswith(f"Episode {prev} complete."))
        peer_name = "Bob" if c["agent"] == "alice" else "Alice"
        new, change = flip_feedback(body["messages"][idx]["content"], peer_name)
        before = body["messages"][idx]["content"]
        body["messages"][idx]["content"] = new
        feedback.append({"job": c["job"], "type": c["type"], "verdict_saved": c["verdict"],
                         "share_C": analysis[c["job"]]["A"]["share"], "samples": args.samples, "message_index": idx,
                         "change": change, "before": before, "after": new, "body": body,
                         "n_prompt_tokens": c["n_prompt_tokens"]})
    # ---- cost
    f = 1.6  # resampled thinking ran ~1.6x the saved length in the first session
    dec_a = sum(r["samples"] * u[0] * f for r, u in zip(from_a, usage_est))
    n_think = {c["job"]: c["n_think"] for c in calls}
    dec_off = sum(r["samples"] * 80 for r in think_off)
    dec_fb = sum(r["samples"] * (n_think[r["job"]] * f + 70) for r in feedback)
    prefill = sum(r["n_prompt_tokens"] for r in think_off) + sum(r["n_prompt_tokens"] for r in feedback) + \
        sum(c["n_prompt_tokens"] for c in random_calls)  # from-A prompts ~ verdict prompts minus the exchange
    llm_calls_a = sum(r["samples"] * u[1] for r, u in zip(from_a, usage_est))
    est = {}
    for name, rate in (("optimistic", 1200), ("central", 860), ("pessimistic", 600)):
        sec = prefill / 14000 + (dec_a + dec_off + dec_fb) / rate + 160 + 300
        est[name] = {"decode_tok_per_s": rate, "minutes": round(sec / 60, 1), "usd": round(sec / 3600 * HOURLY, 2)}
    plan = {"settings": vars(args), "from_a_calls": len(from_a), "think_off_calls": len(think_off),
            "feedback_calls": len(feedback),
            "feedback_types": {t: sum(r["type"] == t for r in feedback) for t in ("onset", "before_onset")},
            "decode_tokens": {"from_a": int(dec_a), "think_off": int(dec_off), "feedback": int(dec_fb)},
            "from_a_llm_calls": llm_calls_a, "prefill_uncached_tokens": prefill, "estimate": est,
            "from_a": from_a, "think_off": think_off, "feedback": feedback}
    (OUT / "session2_plan.json").write_text(json.dumps(plan))
    print(json.dumps({k: v for k, v in plan.items() if k not in ("from_a", "think_off", "feedback")}, indent=1))
    ex = feedback[0]
    print("\nexample feedback edit:", ex["job"], ex["change"])
    print(" before:", repr(ex["before"][-230:]))
    print(" after: ", repr(ex["after"][-230:]))


if __name__ == "__main__":
    main()
