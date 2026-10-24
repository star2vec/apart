"""Item 5: the vLLM ingredient-removal session (edits from session3_edits.py). Starts the GPU
server; run only when approved.

    caffeinate -i agent-collusion/.venv/bin/python tools/step4_5/session3_run.py --go [--deadline-min 105]

Each edited request: chat completions with the recorded settings (named tool choice
submit_verdict), 16 samples, max_tokens 24000, verdict from the tool call (run_session.py's
request loop, deadline, grace, watchdog and resume). A call's five edits are sent together (they
share the prompt up to the first removed message); calls alternate between the random group and
the hard cases, so a deadline cut leaves both groups covered.
Writes agent-collusion/results/step4_5/session3_responses.jsonl.

Final round: --edits session4_edits.jsonl --out-name session4_responses.jsonl --priority
own_messages,feedback_reversed sends every own_messages request (calls alternating between the
groups) before any feedback_reversed one, so a deadline cut falls on the second edit only.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_session as rs  # noqa: E402


def build_requests(samples, edits="session3_edits.jsonl", priority=None):
    rows = [json.loads(l) for l in open(rs.OUT / edits)]
    if priority:  # one pass per edit, in this order
        return [r for name in priority for r in build_requests_from([x for x in rows if x["removal"] == name], samples)]
    return build_requests_from(rows, samples)


def build_requests_from(rows, samples):
    by_call = {}
    for r in rows:
        by_call.setdefault(r["job"], []).append(r)
    groups = {"random": [], "hard": []}
    for job, rs_ in by_call.items():
        groups[rs_[0]["group"]].append(job)
    order = []
    for i in range(max(len(g) for g in groups.values())):
        for g in ("random", "hard"):
            if i < len(groups[g]):
                order.append(groups[g][i])
    return [{"key": f"{r['job']}|{r['removal']}", "job": r["job"], "group": r["group"], "part": r["removal"],
             "kind": "chat", "body": r["body"], "n": samples} for job in order for r in by_call[job]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--go", action="store_true")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--deadline-min", type=float, default=105)
    p.add_argument("--grace-min", type=float, default=10)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=24000)
    p.add_argument("--request-timeout-min", type=float, default=40)
    p.add_argument("--base", default=None)
    p.add_argument("--out-name", default="session3_responses.jsonl")
    p.add_argument("--deadline-at", type=float, default=None)
    p.add_argument("--edits", default="session3_edits.jsonl")
    p.add_argument("--priority", default="", help="comma-separated edit names, sent in this order")
    a = p.parse_args()
    reqs = build_requests(a.samples, a.edits, [x for x in a.priority.split(",") if x])
    by = {}
    for r in reqs:
        by[(r["group"], r["part"])] = by.get((r["group"], r["part"]), 0) + r["n"]
    print(f"{len(reqs)} requests, {sum(r['n'] for r in reqs)} samples; by group/removal: {by}")
    if not a.go:
        print("plan only; pass --go to send (starts the GPU server)")
        return
    asyncio.run(rs.run(a, reqs))


if __name__ == "__main__":
    main()
