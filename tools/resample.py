"""Resample verdict decisions from fixed histories (the verdict resampling of step 4).

Each source decision point is one recorded verdict request: the agent's full history up
to the verdict prompt, the submit_verdict tool, the forced tool_choice and the paper's
sampling settings. We re-send that exact request N times through the repo's LiteLLM
path and record each sampled verdict, giving P(accept) per decision point to compare
with the recorded verdict.

Sources
  --wire <paper_wire_verdict.jsonl>   paper decision points (tools/paper_prompts/rebuild_wire.py)
  --calls <run_dir> [<run_dir> ...]   decision points from our runs' llm_calls.jsonl

Without --go it only writes plan.json (prompts, tokens, estimated GPU time and cost) and
sends nothing. With --go it appends one line per sample to samples.jsonl (resumable:
finished (point, sample) pairs are skipped), logs every request/response to
llm_calls.jsonl, and writes summary.json.

    agent-collusion/.venv/bin/python tools/resample.py --wire <file> --runs rep001 rep002 \
        --n 8 --out agent-collusion/results/resample/pilot            # plan only
    ... --go --max-cost-usd 5                                          # sends requests
"""

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "agent-collusion"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from dotenv import dotenv_values, load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

GPU_USD_PER_S = 0.001097  # Modal H100, 2026-10
VERDICT_TOOL = "submit_verdict"


# --- decision points ------------------------------------------------------------------


def verdict_of(message):
    """The verdict in an assistant message's submit_verdict call, or None."""
    for call in (message or {}).get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") != VERDICT_TOOL:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            return "unparseable"
        return str(args.get("verdict", "")).lower() or "missing"
    return None


def paper_points(wire, runs, episodes, agents):
    for row in map(json.loads, open(wire)):
        if "skip" in row:
            continue
        run = row["run"].split(".")[0]
        if (runs and run not in runs) or (episodes and row["ep"] + 1 not in episodes) or row["agent"] not in agents:
            continue
        yield {
            "point_id": f"paper:{run}:ep{row['ep'] + 1}:{row['agent']}",
            "source": "paper", "run": run, "episode": row["ep"] + 1, "agent": row["agent"],
            "body": row["body"], "prompt_tokens": row["logged_input_tokens"],
            "recorded_verdict": verdict_of(row.get("reply")),
            "recorded_output_tokens": None,
        }


def run_points(run_dirs, episodes, agents):
    for run_dir in map(Path, run_dirs):
        run = json.loads((run_dir / "run.json").read_text())
        system = run["run_config"]["prompt_texts"]["system_prompts"]
        records = [json.loads(l) for l in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
        responses = {r["litellm_call_id"]: r["body"] for r in records if r["event"] == "response"}
        taken = set()
        for r in records:
            body = r.get("body") or {}
            choice = body.get("tool_choice")
            if r["event"] != "request" or not isinstance(choice, dict) or choice["function"]["name"] != VERDICT_TOOL:
                continue
            agent = next(a for a, s in system.items() if body["messages"][0].get("content") == s)
            resp = responses.get(r["litellm_call_id"])
            msg = resp["choices"][0]["message"] if isinstance(resp, dict) and resp.get("choices") else None
            episode = sum(1 for m in body["messages"] if m["role"] == "user"
                          and str(m.get("content", "")).startswith("## Episode") and "verdict phase" in m["content"])
            if (episodes and episode not in episodes) or agent not in agents or (agent, episode) in taken:
                continue  # later verdict requests in the same episode are retries
            taken.add((agent, episode))
            yield {
                "point_id": f"ours:{run_dir.name}:seq{r['seq']}:{agent}",
                "source": "ours", "run": run_dir.name, "episode": episode, "agent": agent,
                "body": body, "prompt_tokens": (resp or {}).get("usage", {}).get("prompt_tokens"),
                "recorded_verdict": verdict_of(msg),
                "recorded_output_tokens": (resp or {}).get("usage", {}).get("completion_tokens"),
            }


# --- plan -----------------------------------------------------------------------------


def plan(points, args):
    n_points = len(points)
    prompt_tokens = sum(p["prompt_tokens"] or 0 for p in points)
    out_per_sample = args.expected_output_tokens
    samples = n_points * args.n
    output_tokens = samples * out_per_sample
    decode_tps = args.decode_tps_per_stream * args.concurrency * args.batch_efficiency
    decode_s = output_tokens / decode_tps
    prefill_once_s = prompt_tokens / args.prefill_tps          # prefix cache reuses the prompt
    prefill_every_s = prompt_tokens * args.n / args.prefill_tps  # no reuse across samples
    overhead_s = args.cold_start_s + args.idle_s
    best = decode_s + prefill_once_s + overhead_s
    worst = decode_s + prefill_every_s + overhead_s
    return {
        "decision_points": n_points,
        "samples_per_point": args.n,
        "requests": samples,
        "prompt_tokens_unique": prompt_tokens,
        "prompt_tokens_sent": prompt_tokens * args.n,
        "expected_output_tokens": output_tokens,
        "assumptions": {
            "expected_output_tokens_per_sample": out_per_sample,
            "decode_tps_per_stream": args.decode_tps_per_stream,
            "concurrency": args.concurrency,
            "batch_efficiency": args.batch_efficiency,
            "prefill_tps_uncached": args.prefill_tps,
            "cold_start_s": args.cold_start_s, "idle_s": args.idle_s,
            "gpu_usd_per_s": GPU_USD_PER_S,
        },
        "gpu_minutes": {"prefix_cache_reused": round(best / 60, 1), "no_prefix_reuse": round(worst / 60, 1)},
        "cost_usd": {"prefix_cache_reused": round(best * GPU_USD_PER_S, 2), "no_prefix_reuse": round(worst * GPU_USD_PER_S, 2)},
    }


# --- sampling -------------------------------------------------------------------------


def sample_once(point, k, model, effort, seed_base):
    import litellm

    from experiments.llm import get_litellm_completion_kwargs

    body = point["body"]
    seed = int(hashlib.sha256(f"{seed_base}:{point['point_id']}:{k}".encode()).hexdigest()[:8], 16)
    started = time.time()
    record = {"point_id": point["point_id"], "sample": k, "seed": seed,
              "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = litellm.completion(
            model=model, messages=body["messages"], tools=body.get("tools"),
            tool_choice=body.get("tool_choice"), temperature=body.get("temperature", 1.0),
            max_tokens=body.get("max_tokens", 64000), seed=seed, num_retries=0,
            **get_litellm_completion_kwargs(model, effort, passthrough_params=("tool_choice",)),
        )
        message = response.choices[0].message
        raw = message.model_dump() if hasattr(message, "model_dump") else dict(message)
        record.update({
            "status": "ok",
            "verdict": verdict_of(raw),
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "reasoning_chars": len(getattr(message, "reasoning_content", None) or ""),
        })
    except Exception as exc:  # recorded and retried on the next resume, not here
        record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:500]})
    record["wall_s"] = round(time.time() - started, 2)
    return record


def summarize(points, samples_path, n):
    rows = [json.loads(l) for l in open(samples_path)] if samples_path.exists() else []
    ok = [r for r in rows if r["status"] == "ok"]
    by_point = {}
    for r in ok:
        by_point.setdefault(r["point_id"], []).append(r)
    per_point = []
    for p in points:
        got = by_point.get(p["point_id"], [])
        verdicts = [r["verdict"] for r in got]
        valid = [v for v in verdicts if v in ("accept", "reject")]
        per_point.append({
            "point_id": p["point_id"], "episode": p["episode"], "agent": p["agent"],
            "recorded_verdict": p["recorded_verdict"],
            "samples_ok": len(got), "valid": len(valid),
            "p_accept": round(sum(v == "accept" for v in valid) / len(valid), 3) if valid else None,
            "agrees_with_recorded": round(sum(v == p["recorded_verdict"] for v in valid) / len(valid), 3) if valid else None,
            "invalid_verdicts": len(verdicts) - len(valid),
        })
    p_acc = [x["p_accept"] for x in per_point if x["p_accept"] is not None]
    return {
        "samples_total": len(rows), "samples_ok": len(ok),
        "samples_error": len(rows) - len(ok),
        "points_complete": sum(x["samples_ok"] >= n for x in per_point),
        "mean_p_accept": round(statistics.fmean(p_acc), 3) if p_acc else None,
        "mean_completion_tokens": round(statistics.fmean(r["completion_tokens"] for r in ok), 1) if ok else None,
        "per_point": per_point,
    }


def fidelity(points, call_log):
    """Does the first logged request for each point equal its source body (seed aside)?"""
    if not call_log.exists():
        return None
    sources = {json.dumps(p["body"]["messages"], sort_keys=True): p["body"] for p in points}
    checked = equal = 0
    for line in open(call_log):
        r = json.loads(line)
        if r["event"] != "request":
            continue
        sent = {k: v for k, v in r["body"].items() if k != "seed"}
        src = sources.get(json.dumps(sent.get("messages"), sort_keys=True))
        if src is None:
            continue
        checked += 1
        equal += sent == src
    return {"requests_checked": checked, "identical_to_source": equal}


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wire")
    source.add_argument("--calls", nargs="+")
    parser.add_argument("--runs", nargs="*", default=[], help="paper runs, e.g. rep001 (default: all)")
    parser.add_argument("--episodes", nargs="*", type=int, default=[], help="1-based (default: all)")
    parser.add_argument("--agents", nargs="*", default=["alice", "bob"])
    parser.add_argument("--n", type=int, default=8, help="samples per decision point")
    parser.add_argument("--concurrency", type=int, default=28, help="max requests in flight")
    parser.add_argument("--lanes", type=int, default=4,
                        help="decision points in flight at once; keep lanes x prompt within the ~310k-token KV cache")
    parser.add_argument("--seed-base", default="exp1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--go", action="store_true", help="send requests (starts the GPU)")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="refuse if the no-reuse estimate exceeds this")
    # cost model: decode and prefill measured in setup S6; batch efficiency is a guess
    parser.add_argument("--expected-output-tokens", type=int, default=425, help="paper verdict mean")
    parser.add_argument("--decode-tps-per-stream", type=float, default=47.0)
    parser.add_argument("--batch-efficiency", type=float, default=0.5)
    parser.add_argument("--prefill-tps", type=float, default=14000.0,
                        help="uncached prefill, measured in setup S6 (cached prefixes run ~185k tok/s)")
    parser.add_argument("--cold-start-s", type=float, default=160.0)
    parser.add_argument("--idle-s", type=float, default=300.0)
    args = parser.parse_args()

    agents = set(args.agents)
    points = list(paper_points(args.wire, set(args.runs), set(args.episodes), agents) if args.wire
                  else run_points(args.calls, set(args.episodes), agents))
    if not points:
        sys.exit("no decision points selected")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    the_plan = plan(points, args)
    the_plan.update({"created_at": datetime.now().astimezone().isoformat(),
                     "source": args.wire or args.calls, "model": dotenv_values(REPO / ".env")["MODEL"],
                     "model_revision": pins.MODEL_REVISION, "seed_base": args.seed_base})
    (out / "plan.json").write_text(json.dumps(the_plan, indent=2, default=str))
    print(json.dumps({k: the_plan[k] for k in ("decision_points", "requests", "prompt_tokens_unique",
                                                 "expected_output_tokens", "gpu_minutes", "cost_usd")}))
    if not args.go:
        print("plan only; pass --go to send requests")
        return
    if args.max_cost_usd is not None and the_plan["cost_usd"]["no_prefix_reuse"] > args.max_cost_usd:
        sys.exit(f"estimate ${the_plan['cost_usd']['no_prefix_reuse']} exceeds --max-cost-usd {args.max_cost_usd}")

    from experiments.llm import enable_call_log

    enable_call_log(out / "llm_calls.jsonl")
    samples_path = out / "samples.jsonl"
    done = set()
    if samples_path.exists():
        done = {(r["point_id"], r["sample"]) for r in map(json.loads, open(samples_path)) if r["status"] == "ok"}
    env = dotenv_values(REPO / ".env")
    todo = [(p, k) for p in points for k in range(args.n) if (p["point_id"], k) not in done]
    print(f"{len(done)} samples already done, {len(todo)} to send", flush=True)
    lock = threading.Lock()
    progress = {"done": 0}
    by_point = {}
    for p, k in todo:
        by_point.setdefault(p["point_id"], (p, []))[1].append(k)
    per_lane = max(1, args.concurrency // max(1, args.lanes))

    def record(sink, rec):
        with lock:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            progress["done"] += 1
            if progress["done"] % 25 == 0 or progress["done"] == len(todo):
                print(f"{progress['done']}/{len(todo)} samples", flush=True)

    def run_point(point, ks, sink):
        # The first sample prefills the prompt into vLLM's prefix cache; the rest reuse it.
        call = lambda k: sample_once(point, k, env["MODEL"], env["REASONING_EFFORT"], args.seed_base)
        record(sink, call(ks[0]))
        with ThreadPoolExecutor(max_workers=per_lane) as inner:
            for future in as_completed([inner.submit(call, k) for k in ks[1:]]):
                record(sink, future.result())

    with ThreadPoolExecutor(max_workers=args.lanes) as lanes, samples_path.open("a") as sink:
        for future in as_completed([lanes.submit(run_point, p, ks, sink) for p, ks in by_point.values()]):
            future.result()
    summary = summarize(points, samples_path, args.n)
    summary["fidelity"] = fidelity(points, out / "llm_calls.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_point"}, indent=2))


if __name__ == "__main__":
    main()
