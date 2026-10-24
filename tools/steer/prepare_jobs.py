"""Steering prep (Mac): job file for the steering worker.

    tools/.venv/bin/python tools/steer/prepare_jobs.py --set pilot|main

Token ids: the call's rendered verdict prompt with the generation prompt (from step 6's jobs),
C = the "\\n" after <|im_start|>assistant. Longest prompt first (memory check on it).
Writes agent-collusion/results/steer/jobs_<set>.json.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["pilot", "main"], required=True)
    a = p.parse_args()
    calls = json.loads((RES / "steer" / "steer_calls.json").read_text())
    chosen = {j: c for j, c in calls.items() if c["pilot"] or a.set == "main"}
    meta = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl")) if c["job"] in chosen}
    jobs = []
    for line in open(RES / "step6" / "jobs.jsonl"):
        j = json.loads(line)
        if j["id"] in chosen:
            m = meta[j["id"]]
            ids = j["token_ids"][:m["think_start"]]
            assert m["c_token"] == len(ids) - 3
            jobs.append({"id": j["id"], "token_ids": ids, "c_token": m["c_token"],
                         "target_sign": chosen[j["id"]]["target_sign"], "group": chosen[j["id"]]["group"],
                         "share_A_vllm": chosen[j["id"]]["share_A_vllm"]})
    jobs.sort(key=lambda j: -len(j["token_ids"]))
    out = RES / "steer" / f"jobs_{a.set}.json"
    out.write_text(json.dumps(jobs))
    print(out, [(j["id"], len(j["token_ids"]), j["group"], j["target_sign"], j["share_A_vllm"]) for j in jobs])


if __name__ == "__main__":
    main()
