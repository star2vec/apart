"""Cache job, stage 1: rebuild communication-phase request bodies for the paper's runs.

Run with the agent-collusion venv (LiteLLM 1.94.1, the repo's tool code):
    agent-collusion/.venv/bin/python tools/cache/rebuild_comm_wire.py <traces_dir> <out.jsonl>

Per run and agent, two bodies, both with the communication-phase tool list:
  kind=pass   history of the last episode up to (not including) its verdict-phase prompt:
              the last communication call's prompt extended by the phase's remaining turns,
              so every message of the last episode is inside it
  kind=check  the last real communication call of the last episode (history before the
              agent's final send_message turn), with that call's logged input_tokens
Same capture path as tools/paper_prompts/rebuild_wire.py.
"""

import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent-collusion"))

import litellm  # noqa: E402

from experiments.agents import _select_tools_and_choice  # noqa: E402
from experiments.llm import get_litellm_completion_kwargs  # noqa: E402
from experiments.tool_schemas import forced_tool_choice, get_tool_schemas  # noqa: E402

CAPTURED = []
REPLY = json.dumps({
    "id": "x", "object": "chat.completion", "created": 1, "model": "qwen3.6-27b",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}).encode()


class Capture(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        CAPTURED.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(REPLY)))
        self.end_headers()
        self.wfile.write(REPLY)


server = HTTPServer(("127.0.0.1", 0), Capture)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_port}/v1"
MODEL = "openai/qwen3.6-27b"
PEER = {"alice": "bob", "bob": "alice"}


def wire_body(history, task_type, peer, effort):
    tools = get_tool_schemas(task_type=task_type, phase="communication", answer_key=None, peer=peer)
    tools, choice = _select_tools_and_choice(
        tools=tools, tool_choice=forced_tool_choice(task_type, "communication"),
        model=MODEL, reasoning_effort=effort)
    extra = get_litellm_completion_kwargs(MODEL, effort, passthrough_params=("tool_choice",))
    litellm.completion(model=MODEL, messages=history, tools=tools, tool_choice=choice,
                       temperature=1.0, max_tokens=64000, api_base=BASE, api_key="x", **extra)
    return CAPTURED.pop()


def main() -> None:
    traces, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    with out_path.open("w") as out:
        for f in sorted(traces.glob("rep*.json.gz")):
            run = json.load(gzip.open(f))
            effort = run["run_config"]["reasoning_efforts"]
            ep = run["results"][-1]
            for agent in ("alice", "bob"):
                msgs = ep["agents"][agent]["messages"]
                verdict_prompt = ep["agents"][PEER[agent]]["verdict_prompt"]
                cuts = [i for i, m in enumerate(msgs) if m.get("role") == "user" and m.get("content") == verdict_prompt]
                if len(cuts) != 1:
                    raise ValueError(f"{f.name} {agent}: {len(cuts)} verdict prompts in the last episode")
                comm_start = max(i for i, m in enumerate(msgs[: cuts[0]])
                                 if m.get("role") == "user" and "communication phase" in (m.get("content") or ""))
                sends = [i for i in range(comm_start, cuts[0]) if msgs[i].get("role") == "assistant"]
                usage = [u for u in ep["llm_usage"] if u["actor"] == agent and u["phase"] == "communication"
                         and u.get("response_status") == "accepted"]
                rows = [("pass", msgs[: cuts[0]], None),
                        ("check", msgs[: sends[-1]], usage[-1]["input_tokens"] if usage else None)]
                for kind, history, logged in rows:
                    out.write(json.dumps({
                        "run": f.name.split(".")[0], "ep": ep["episode_index"], "agent": agent, "kind": kind,
                        "history_len": len(history), "logged_input_tokens": logged,
                        "body": wire_body(history, ep["task_type"], PEER[agent], effort[agent]),
                    }) + "\n")
                    n += 1
    print("bodies written:", n)


if __name__ == "__main__":
    main()
