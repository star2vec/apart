"""Second vLLM session (tasks 4-6 of the 2026-10-24 plan). Starts the GPU server; run only when
approved.

    agent-collusion/.venv/bin/python tools/step4_5/session2_run.py --go [--deadline-min 105]

Order: thinking off and feedback edit (single chat requests, n samples each), then the from-A
episodes. Plan from session2_build.py.

From A (controlled peer). For each sample, the studied agent's history up to A is replayed
through the game's own code: experiments.agents.run_agent_turn (LLM call, tool execution,
reasoning replay) against a minimal protocol state, with the game's round instruction,
incoming-message format, send_message rules (200-character limit while throttled, one message
per round, 3 attempts per round) and runner notices. The peer is not simulated: its saved
messages for that episode are delivered unchanged when the game would have delivered them
(Alice opens each round, so Bob reads Alice's round-r message before his round-r turn, and Alice
reads Bob's round-r message at her next turn or before the verdict). A peer message missing
from the transcript becomes the game's "failed to deliver" notice. Then the saved verdict prompt
and one verdict call with the recorded request settings (forced submit_verdict).
Every LLM request and response goes to the repo's call log (from_a_llm_calls.jsonl).
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
OUT = RES / "step4_5"
sys.path.insert(0, str(ROOT / "agent-collusion"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "openai/qwen3.6-27b"


def setup_env(base_override):
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / "agent-collusion" / ".env")
    os.environ["OPENAI_API_BASE"] = base_override or env["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = "x" if base_override else env["OPENAI_API_KEY"]


class PastDeadline(RuntimeError):
    pass


def start_watchdog(hard_stop):
    """Kill the whole process at the hard stop, whatever threads are doing (cost cap)."""
    def watch():
        while time.time() < hard_stop:
            time.sleep(5)
        print("watchdog: hard stop reached, exiting", flush=True)
        os._exit(3)
    threading.Thread(target=watch, daemon=True).start()


def run_episode(spec, sample, max_tokens, effort, deadline=None):
    """One controlled-peer continuation from A; returns the verdict record."""
    import litellm
    from experiments.agents import _select_tools_and_choice, run_agent_turn
    from experiments.episode_runner import _inject_incoming_messages
    from experiments.llm import get_litellm_completion_kwargs
    from experiments.prompts.messages import communication_round_instruction, peer_delivery_failure_notice
    from experiments.protocol import COMMUNICATION_ATTEMPTS
    from experiments.protocol.state import deliver_runner_notice, pop_incoming_messages, reset_message_delivery
    from experiments.tool_schemas import forced_tool_choice, get_tool_schemas

    me = spec["agent"]
    peer = "bob" if me == "alice" else "alice"
    state = {"task_type": spec["task_type"], "phase": "communication", "round": 0, "throttled": True,
             "char_limit": 200, "events": [],
             "agents": {a: {"delivered": False, "inbox": [], "private_raw_log": [], "answer_key": [],
                            "verdict_submission": None} for a in ("alice", "bob")}}
    messages = json.loads(json.dumps(spec["history_at_A"]))
    max_rounds = spec["max_rounds"]
    sent = []

    def deliver_peer(r):
        item = spec["peer_schedule"][r]["message"]
        if item is None:
            deliver_runner_notice(state, me, peer_delivery_failure_notice(peer=peer, round_number=r + 1))
            # deliver_runner_notice stamps the current round; keep the originating one
        else:
            state["agents"][me]["inbox"].append({"from": peer, "to": me, "content": item["content"], "round": r,
                                                 "message_type": item["message_type"]})

    tool_choice = forced_tool_choice(spec["task_type"], "communication")
    for r in range(max_rounds):
        state["round"] = r
        state["throttled"] = r >= spec["throttle_start_round"]
        if me == "bob":
            deliver_peer(r)  # Alice opened the round
        _inject_incoming_messages(messages, incoming=pop_incoming_messages(state, me), max_rounds=max_rounds)
        messages.append({"role": "user", "content": communication_round_instruction(r + 1, max_rounds)})
        delivered = False
        for attempt in range(COMMUNICATION_ATTEMPTS):
            if deadline and time.time() > deadline:
                raise PastDeadline("deadline passed mid-episode")
            reset_message_delivery(state, me)
            turn = run_agent_turn(actor=me, model=MODEL, messages=messages, state=state, tool_choice=tool_choice,
                                  reasoning_effort=effort, temperature=1.0, max_output_tokens=max_tokens)
            ok = [c for c in turn["tool_calls"] if c["tool_name"] == "send_message" and c["result"].get("success")]
            if ok:
                sent.append({"round": r + 1, "attempts": attempt + 1, "content": ok[0]["arguments"].get("content")})
                delivered = True
                break
        if not delivered:
            sent.append({"round": r + 1, "attempts": COMMUNICATION_ATTEMPTS, "content": None})
        if me == "alice":
            deliver_peer(r)  # Bob answers after Alice within the same round
    _inject_incoming_messages(messages, incoming=pop_incoming_messages(state, me), max_rounds=max_rounds)
    messages.append({"role": "user", "content": spec["verdict_prompt"]})
    state["phase"] = "verdict"
    tools = get_tool_schemas(task_type=spec["task_type"], phase="verdict", answer_key=None, peer=peer)
    tools, choice = _select_tools_and_choice(tools=tools, tool_choice=forced_tool_choice(spec["task_type"], "verdict"),
                                             model=MODEL, reasoning_effort=effort)
    extra = get_litellm_completion_kwargs(MODEL, effort, passthrough_params=("tool_choice",))
    if deadline and time.time() > deadline:
        raise PastDeadline("deadline passed before the verdict call")
    resp = litellm.completion(model=MODEL, messages=messages, tools=tools, tool_choice=choice, temperature=1.0,
                              max_tokens=max_tokens, num_retries=3, **extra)
    ch = resp.choices[0]
    verdict = None
    for tc in ch.message.tool_calls or []:
        if tc.function.name == "submit_verdict":
            try:
                verdict = json.loads(tc.function.arguments or "{}").get("verdict")
            except json.JSONDecodeError:
                pass
    capped = ch.finish_reason == "length"
    return {"key": f"{spec['job']}|from_A|{sample}", "job": spec["job"], "sample": sample, "status": "ok",
            "verdict": None if capped or verdict not in ("accept", "reject") else verdict, "capped": capped,
            "finish_reason": ch.finish_reason, "messages_sent": sent,
            "verdict_reasoning": (getattr(ch.message, "reasoning_content", None) or "")[:20000]}


def run_from_a(args, plan, deadline):
    from experiments.llm import enable_call_log
    enable_call_log(OUT / "from_a_llm_calls.jsonl")
    out_path = OUT / "session2_from_a.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["key"] for l in open(out_path) if json.loads(l).get("status") == "ok"}
    effort = "default"
    n_samples = lambda spec: min(spec["samples"], args.from_a_max_samples) if args.from_a_max_samples else spec["samples"]  # noqa: E731
    # sample-major order: every call gets sample k before any call gets sample k+1
    tasks = [(spec, k) for k in range(max(n_samples(s) for s in plan["from_a"])) for spec in plan["from_a"]
             if k < n_samples(spec) and f"{spec['job']}|from_A|{k}" not in done]
    print(f"from A: {len(tasks)} episodes to run ({len(done)} done)", flush=True)
    lock = threading.Lock()
    n = {"ok": 0, "error": 0}

    def work(spec, k):
        if time.time() > deadline:
            return None
        try:
            return run_episode(spec, k, args.max_tokens, effort, deadline)
        except Exception as e:  # noqa: BLE001
            return {"key": f"{spec['job']}|from_A|{k}", "job": spec["job"], "sample": k, "status": "error",
                    "error": repr(e)[:500]}

    with ThreadPoolExecutor(max_workers=args.from_a_workers) as pool:
        futs = [pool.submit(work, s, k) for s, k in tasks]
        for f in as_completed(futs):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                n[rec["status"]] += 1
                if (n["ok"] + n["error"]) % 20 == 0:
                    print(f"from A: {n['ok'] + n['error']}/{len(tasks)} (errors {n['error']})", flush=True)
    print(json.dumps({"from_a": n}), flush=True)


def single_requests(plan):
    reqs = []
    for r in plan["think_off"]:
        reqs.append({"key": f"{r['job']}|think_off", "job": r["job"], "group": "think_off", "part": "think_off",
                     "kind": "chat", "n": r["samples"], "body": None})
    for r in plan["feedback"]:
        reqs.append({"key": f"{r['job']}|feedback", "job": r["job"], "group": "feedback", "part": "feedback",
                     "kind": "chat", "n": r["samples"], "body": r["body"]})
    return reqs


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--deadline-min", type=float, default=105)
    parser.add_argument("--grace-min", type=float, default=10)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--from-a-workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-timeout-min", type=float, default=40)
    parser.add_argument("--base", default=None, help="override server URL (offline tests)")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--from-a-max-samples", type=int, default=0, help="cut samples per call (cost)")
    parser.add_argument("--deadline-at", type=float, default=None, help="absolute deadline (restarts)")
    parser.add_argument("--test-limit", type=int, default=0, help="offline tests: first N items, 2 samples each")
    args = parser.parse_args()
    plan = json.loads((OUT / "session2_plan.json").read_text())
    if args.test_limit:
        plan["from_a"] = [dict(s, samples=2) for s in plan["from_a"][:args.test_limit]]
        plan["think_off"] = plan["think_off"][:args.test_limit]
        plan["feedback"] = plan["feedback"][:args.test_limit]
        OUT = OUT / "test"
        OUT.mkdir(exist_ok=True)
    wire = {}
    for line in open(RES / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r["body"]
    reqs = single_requests(plan)
    for r in reqs:
        if r["body"] is None:  # thinking off: the recorded request, thinking disabled
            _, run, agent, ep = r["job"].split(".")
            r["body"] = dict(wire[(run, int(ep[2:]), agent)], chat_template_kwargs={"enable_thinking": False})
    print(f"{len(reqs)} single requests ({sum(r['n'] for r in reqs)} samples); "
          f"{sum(s['samples'] for s in plan['from_a'])} from-A episodes")
    if not args.go:
        print("plan only; pass --go to send (starts the GPU server)")
        return
    setup_env(args.base)
    t0 = time.time()
    deadline = args.deadline_at or t0 + args.deadline_min * 60
    start_watchdog(deadline + args.grace_min * 60)
    if not args.skip_single:
        import run_session as rs1
        rs1.OUT = OUT
        ns = argparse.Namespace(base=args.base, out_name="session2_single.jsonl", deadline_min=args.deadline_min,
                                deadline_at=deadline, grace_min=args.grace_min, retries=args.retries,
                                parallel=args.parallel, max_tokens=args.max_tokens,
                                request_timeout_min=args.request_timeout_min)
        asyncio.run(rs1.run(ns, reqs))
    run_from_a(args, plan, deadline)
    print(f"session 2 wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
