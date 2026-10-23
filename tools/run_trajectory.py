"""Run one agent-collusion trajectory against the Modal server, with a hard timeout.

Uses the repo's own launcher to build the runner command (--dry-run), swapping in a
copy of the chosen sequence's manifest cut to --episodes. Writes server.json (served
model, revision, sampling defaults) and summary.json (per-episode wall time, tokens,
verdict pair, largest prompt) into the run directory.

    agent-collusion/.venv/bin/python tools/run_trajectory.py --sequence 1 --episodes 2 \
        --output-dir results/timing --timeout-min 90
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "agent-collusion"
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from openai import OpenAI  # noqa: E402

SEQUENCES = "task/task_sequences_50x10"


def server_snapshot(env: dict) -> dict:
    client = OpenAI(base_url=env["OPENAI_API_BASE"], api_key=env["OPENAI_API_KEY"], timeout=1800)
    models = [m.model_dump() for m in client.models.list().data]
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "api_base": env["OPENAI_API_BASE"],
        "models_endpoint": models,
        "pins": {k: getattr(pins, k) for k in dir(pins) if k.isupper()},
        "sampling": {
            "temperature": 1.0,
            "max_tokens": 64000,
            "server_defaults_from_generation_config": {"top_p": 0.95, "top_k": 20},
        },
    }


def runner_command(manifest: Path, args: argparse.Namespace, env: dict) -> list[str]:
    model, effort = env["MODEL"], env["REASONING_EFFORT"]
    dry = subprocess.run(
        [sys.executable, "-m", "experiments", "--dry-run",
         "--alice-model", model, "--bob-model", model,
         "--alice-reasoning-effort", effort, "--bob-reasoning-effort", effort,
         "--task-sequence-record", SEQUENCES, "--repeats", "1",
         "--start-index", str(args.sequence), "--output-dir", args.output_dir],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    line = next(l for l in dry.stdout.splitlines() if l.startswith("$ "))
    command = shlex.split(line[2:])
    command[command.index("--manifest") + 1] = str(manifest)
    return command


def summarize(run_dir: Path, wall_s: float, log_text: str) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    usage = [json.loads(l) for l in (run_dir / "llm_usage.jsonl").read_text().splitlines()]
    episodes, previous_end = [], None
    started = datetime.fromisoformat(min(u["recorded_at"] for u in usage)) if usage else None
    for result in run["results"]:
        index = result["episode_index"] + 1
        calls = [u for u in usage if u["episode_index"] == index]
        end = datetime.fromisoformat(max(u["recorded_at"] for u in calls))
        begin = previous_end or started
        previous_end = end
        meta = result["analysis_metadata"]
        verdicts = meta["verdict_by_agent"]
        episodes.append({
            "episode": index,
            "task_type": result["task_type"],
            "wall_s_between_last_calls": round((end - begin).total_seconds(), 1),
            "llm_calls": len(calls),
            "failed_attempts": sum(u.get("response_status") != "accepted" for u in calls),
            "input_tokens": sum(u.get("input_tokens") or 0 for u in calls),
            "output_tokens": sum(u.get("provider_output_tokens") or 0 for u in calls),
            "reasoning_tokens": sum(u.get("reasoning_tokens") or 0 for u in calls),
            "largest_prompt": max((u.get("input_tokens") or 0) for u in calls),
            "largest_output": max((u.get("provider_output_tokens") or 0) for u in calls),
            "finish_reasons": sorted({str(u.get("finish_reason")) for u in calls}),
            "verdicts": verdicts,
            "mutual_accept": all(v == "accept" for v in verdicts.values()),
            "verdict_correct": meta["verdict_correct_by_agent"],
        })
    return {
        "run_dir": str(run_dir),
        "process_wall_s": round(wall_s, 1),
        "episodes_finished": len(episodes),
        "context_pruning_events": log_text.count("context limit: dropped"),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=1, help="1-based task sequence")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output-dir", default="results/timing", help="relative to agent-collusion/")
    parser.add_argument("--timeout-min", type=float, required=True)
    args = parser.parse_args()

    env = dotenv_values(REPO / ".env")
    out = REPO / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = REPO / SEQUENCES / f"rep{args.sequence:03d}_sampled_manifest.json"
    manifest = out / f"rep{args.sequence:03d}_first{args.episodes}_manifest.json"
    pairs = json.loads(source.read_text())["pairs"][: args.episodes]
    manifest.write_text(json.dumps({"pairs": pairs, "source": str(source.relative_to(REPO))}, indent=2))

    print("waking server and recording its identity ...", flush=True)
    snapshot = server_snapshot(env)
    command = runner_command(manifest, args, env)
    before = {p.name for p in out.iterdir() if p.is_dir()}
    log_path = out / f"seq{args.sequence:03d}_{datetime.now():%Y%m%d_%H%M%S}.log"
    print("$", shlex.join(command), flush=True)

    started = time.time()
    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=args.timeout_min * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    wall = time.time() - started

    new = sorted(p for p in out.iterdir() if p.is_dir() and p.name not in before)
    if not new:
        sys.exit(f"no run directory created; see {log_path}")
    run_dir = new[-1]
    (run_dir / "server.json").write_text(json.dumps(snapshot, indent=2, default=str))
    summary = summarize(run_dir, wall, log_path.read_text())
    summary.update({"exit_code": process.returncode, "timed_out": timed_out,
                    "timeout_min": args.timeout_min, "log": str(log_path), "command": command})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
