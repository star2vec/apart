"""Steering prep (Mac): unsteered check job. The 3 pilot calls (unsteered generation) plus the
5 paper replies teacher-forced in setup S6 under vLLM, so the transformers forward path can be
compared token log-prob for token log-prob with the serving engine.

    tools/.venv/bin/python tools/steer/prepare_check.py
Writes agent-collusion/results/steer/jobs_check.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))
import pins  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from teacher_force import reply_ids, segment  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    jobs = json.loads((RES / "steer" / "jobs_pilot.json").read_text())
    s6 = {r["pick"]: r for r in json.loads((RES / "setup_s6" / "teacher_force.json").read_text())["results"]}
    rows = {(r["run"].split(".")[0], r["ep"] + 1, r["agent"]): r
            for r in map(json.loads, open(RES / "paper" / "paper_wire_verdict.jsonl")) if "skip" not in r}
    for pick, v in s6.items():
        run, ep, agent = pick.split(":")
        row = rows[(run, int(ep), agent)]
        conv, tools = vllm_conversation(row["body"]["messages"]), vllm_tools(row["body"].get("tools"))
        p_ids, r_ids = reply_ids(tok, conv, tools, row["reply"])
        jobs.append({"id": f"tf.{pick}", "token_ids": p_ids, "reply_ids": r_ids, "think_end": segment(tok, r_ids),
                     "vllm": {"mean": v["paper_reply"]["all"]["mean"], "min": v["paper_reply"]["all"]["min"],
                              "reasoning_mean": v["paper_reply"]["reasoning"]["mean"],
                              "after_think_mean": v["paper_reply"]["after_think"]["mean"], "n": v["paper_reply"]["all"]["n"]}})
    (RES / "steer" / "jobs_check.json").write_text(json.dumps(jobs))
    print([(j["id"], len(j["token_ids"]), len(j.get("reply_ids", []))) for j in jobs])


if __name__ == "__main__":
    main()
