"""Stage A: rebuild the paper's verdict-call requests and capture LiteLLM's wire body.

Run with the agent-collusion venv. Writes one JSON line per verdict call.
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

traces, out_path = Path(sys.argv[1]), Path(sys.argv[2])
model = "openai/qwen3.6-27b"
peer_of = {"alice": "bob", "bob": "alice"}
rows = 0
with out_path.open("w") as out:
    for f in sorted(traces.glob("*.json.gz")):
        run = json.load(gzip.open(f))
        effort = run["run_config"]["reasoning_efforts"]
        for ep in run["results"]:
            for agent in ("alice", "bob"):
                rec = ep["agents"][agent]
                prompt = ep["agents"][peer_of[agent]]["verdict_prompt"]
                msgs = rec["messages"]
                cuts = [i for i, m in enumerate(msgs) if m.get("role") == "user" and m.get("content") == prompt]
                usage = [u for u in ep["llm_usage"] if u["actor"] == agent and u["phase"] == "verdict"
                         and u.get("agent_attempt") == 1 and u.get("api_attempt") == 1]
                if len(cuts) != 1 or len(usage) != 1:
                    out.write(json.dumps({"run": f.name, "ep": ep["episode_index"], "agent": agent,
                                          "skip": f"cuts={len(cuts)} usage={len(usage)}"}) + "\n")
                    continue
                history = msgs[: cuts[0] + 1]
                tools = get_tool_schemas(task_type=ep["task_type"], phase="verdict",
                                         answer_key=[str(c) for c in rec.get("answer_key", []) or []],
                                         peer=peer_of[agent])
                tools, choice = _select_tools_and_choice(
                    tools=tools, tool_choice=forced_tool_choice(ep["task_type"], "verdict"),
                    model=model, reasoning_effort=effort[agent])
                extra = get_litellm_completion_kwargs(model, effort[agent], passthrough_params=("tool_choice",))
                litellm.completion(model=model, messages=history, tools=tools, tool_choice=choice,
                                   temperature=1.0, max_tokens=64000, api_base=BASE, api_key="x", **extra)
                body = CAPTURED.pop()
                out.write(json.dumps({"run": f.name, "ep": ep["episode_index"], "agent": agent,
                                      "logged_input_tokens": usage[0]["input_tokens"],
                                      "history_len": len(history), "body": body}) + "\n")
                rows += 1
print("wire bodies written:", rows)
