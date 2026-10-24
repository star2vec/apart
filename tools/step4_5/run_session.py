"""Steps 4-5: the vLLM resampling session (plan from build_plan.py). Starts the GPU server;
run only when approved.

    agent-collusion/.venv/bin/python tools/step4_5/run_session.py --go [--deadline-min 110] [--parallel 6]

Requests, in priority order (each call's requests are sent together so they share vLLM's prefix
cache): chosen calls, then the random group, then step 5's variant B edited prompts.
  A      chat completions with the recorded verdict request (named tool choice submit_verdict),
         n samples, max_tokens 24000; verdict read from the tool call's arguments
  B      completions with prompt = token ids of the rendered prompt + the first k thinking tokens,
         n samples, max_tokens 24000; verdict read from the continuation's
         <parameter=verdict> ... </parameter> (no forced tool call on this endpoint)
  C      chat completions with the edited request, as A
A sample is invalid if it hit the token cap (finish_reason "length") or has no verdict.
Every response goes to responses.jsonl (resumable: finished requests are skipped). No new
request starts after --deadline-min (100); requests still running at deadline + grace (110 min)
are cancelled; failed requests are retried up to --retries times while time remains. The server
scales down 5 minutes after the last request, so GPU time stays under ~2.5 + 110 + 5 minutes.

Without --go: prints the request list and exits.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
OUT = RES / "step4_5"
VERDICT_RE = re.compile(r"<parameter=verdict>\s*(accept|reject)\b", re.I)


def load_requests(plan):
    jobs = {}
    for line in open(RES / "step6" / "jobs.jsonl"):
        j = json.loads(line)
        jobs[j["id"]] = j
    calls6 = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl"))}
    wire = {}
    for line in open(RES / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r["body"]
    edits = {(e["run"], e["agent"]): e["body"] for e in map(json.loads, open(RES / "step5" / "edits.jsonl"))
             if e["variant"] == "B"}
    reqs = []
    ordered = sorted(plan["calls"], key=lambda c: c["groups"][0].startswith("random"))
    for c in ordered:
        group = "random" if c["groups"][0].startswith("random") else "chosen"
        body = dict(wire[(c["run"], c["episode"], c["agent"])])
        reqs.append({"key": f"{c['job']}|A", "job": c["job"], "group": group, "part": "A", "kind": "chat",
                     "body": body, "n": c["full_samples"]})
        ids = jobs[c["job"]]["token_ids"]
        n_prompt = calls6[c["job"]]["think_start"]
        for k in c["cuts"]:
            reqs.append({"key": f"{c['job']}|B|{k['name']}", "job": c["job"], "group": group, "part": "B",
                         "cut": k["name"], "think_tokens": k["think_tokens"], "kind": "completion",
                         "prompt_ids": ids[:n_prompt + k["think_tokens"]], "n": k["samples"]})
    for e in plan["edits"]:
        reqs.append({"key": f"{e['job']}|C", "job": e["job"], "group": "edit", "part": "C", "kind": "chat",
                     "body": dict(edits[(e["run"], e["agent"])]), "n": plan["settings"]["samples"]})
    return reqs


def reorder(reqs, order):
    rank = {g: i for i, g in enumerate(order)}
    return sorted(reqs, key=lambda r: rank[r["group"]])  # stable: keeps each call's requests together


def verdict_from_chat(choice):
    msg = choice.get("message") or {}
    for tc in msg.get("tool_calls") or []:
        if tc.get("function", {}).get("name") == "submit_verdict":
            try:
                v = json.loads(tc["function"]["arguments"] or "{}").get("verdict")
            except json.JSONDecodeError:
                v = None
            if v in ("accept", "reject"):
                return v
    m = VERDICT_RE.search(msg.get("content") or "")
    return m.group(1).lower() if m else None


def verdict_from_text(text):
    m = VERDICT_RE.search(text or "")
    return m.group(1).lower() if m else None


async def run(args, reqs):
    import httpx
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / "agent-collusion" / ".env")
    base = args.base or env["OPENAI_API_BASE"]
    key = env["OPENAI_API_KEY"] if not args.base else "x"
    model = "qwen3.6-27b"
    out_path = OUT / args.out_name
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add(r["key"])
    todo = [r for r in reqs if r["key"] not in done]
    print(f"{len(reqs)} requests, {len(done)} already done, {len(todo)} to send", flush=True)
    t0 = time.time()
    deadline = args.deadline_at if args.deadline_at else t0 + args.deadline_min * 60
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.parallel)
    client = httpx.AsyncClient(base_url=base, headers={"Authorization": f"Bearer {key}"},
                               timeout=httpx.Timeout(args.request_timeout_min * 60), follow_redirects=True,
                               max_redirects=200)
    stats = {"ok": 0, "error": 0, "skipped_deadline": 0}

    async def one(r, seed):
        async with sem:
            if time.time() > deadline:
                stats["skipped_deadline"] += 1
                return
            started = time.time()
            try:
                if r["kind"] == "chat":
                    payload = dict(r["body"], n=r["n"], max_tokens=args.max_tokens, temperature=1.0, seed=seed)
                    resp = await client.post("/chat/completions", json=payload)
                else:
                    payload = {"model": model, "prompt": r["prompt_ids"], "n": r["n"], "max_tokens": args.max_tokens,
                               "temperature": 1.0, "seed": seed, "skip_special_tokens": False}
                    resp = await client.post("/completions", json=payload)
                resp.raise_for_status()
                body = resp.json()
                samples = []
                for ch in body["choices"]:
                    capped = ch.get("finish_reason") == "length"
                    if r["kind"] == "chat":
                        v = verdict_from_chat(ch)
                        msg = ch.get("message") or {}
                        text = (msg.get("reasoning") or msg.get("reasoning_content") or "") + "\n</think>\n" + (msg.get("content") or "") + json.dumps(msg.get("tool_calls"))
                    else:
                        text = ch.get("text") or ""
                        v = verdict_from_text(text)
                    samples.append({"verdict": None if capped else v, "capped": capped,
                                    "finish_reason": ch.get("finish_reason"), "text": text})
                rec = {k: r[k] for k in ("key", "job", "group", "part") if k in r} | {
                    "cut": r.get("cut"), "think_tokens": r.get("think_tokens"), "n": r["n"], "seed": seed,
                    "status": "ok", "wall_s": round(time.time() - started, 1), "usage": body.get("usage"),
                    "samples": samples}
                stats["ok"] += 1
            except Exception as e:  # noqa: BLE001
                rec = {"key": r["key"], "job": r["job"], "status": "error", "error": repr(e)[:500],
                       "wall_s": round(time.time() - started, 1)}
                stats["error"] += 1
            async with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_done = stats["ok"] + stats["error"]
                if n_done % 20 == 0:
                    print(f"[{(time.time() - t0) / 60:5.1f} min] {n_done}/{len(todo)} requests "
                          f"(ok {stats['ok']}, errors {stats['error']})", flush=True)

    seeds = {r["key"]: 1000 + i for i, r in enumerate(reqs)}
    hard_stop = deadline + args.grace_min * 60
    for attempt in range(1 + args.retries):
        tasks = [asyncio.ensure_future(one(r, seeds[r["key"]])) for r in todo]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=max(1.0, hard_stop - time.time()))
        except asyncio.TimeoutError:
            print("hard stop: cancelled requests still running", flush=True)
            break
        ok_keys = {json.loads(l)["key"] for l in open(out_path) if json.loads(l).get("status") == "ok"}
        todo = [r for r in reqs if r["key"] not in ok_keys]
        if not todo or time.time() > deadline or attempt == args.retries:
            break
        print(f"retrying {len(todo)} failed requests", flush=True)
    await client.aclose()
    wall = (time.time() - t0) / 60
    print(json.dumps({"stats": stats, "wall_min": round(wall, 1)}), flush=True)
    return wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--deadline-min", type=float, default=100, help="no new request after this")
    parser.add_argument("--grace-min", type=float, default=10, help="then cancel whatever is still running")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=6, help="requests in flight (each carries n samples)")
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--request-timeout-min", type=float, default=40)
    parser.add_argument("--base", default=None, help="override server URL (offline tests)")
    parser.add_argument("--out-name", default="responses.jsonl")
    parser.add_argument("--order", default="chosen,random,edit", help="group priority")
    parser.add_argument("--deadline-at", type=float, default=None, help="absolute unix time for the deadline (restarts)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((OUT / "plan.json").read_text())
    reqs = reorder(load_requests(plan), args.order.split(","))
    by = {}
    for r in reqs:
        by[(r["group"], r["part"])] = by.get((r["group"], r["part"]), 0) + r["n"]
    print(f"{len(reqs)} requests; samples by group/part: {by}")
    if not args.go:
        print("plan only; pass --go to send (starts the GPU server)")
        return
    asyncio.run(run(args, reqs))


if __name__ == "__main__":
    main()
