"""Steering prep (Mac): job file for steering session 3, items 1-2 (6 samples per condition).

    tools/.venv/bin/python tools/steer/prepare_session3.py

Item 1  2 SD from C on: toward accept on 5 settled rejects (with a new unsteered condition, since
        they have no transformers samples yet) and 5 hard cases; toward reject on the 8 hard cases
        not yet run at 2 SD; a random direction at 2 SD (the vector runs A and A2 used) on the
        accept-push calls, which also serves the reject push on 5 of the 8.
Item 2  at C only, toward reject: 2 SD on 5 hard cases and the 3 settled accepts; 4 SD, and a
        random direction at 4 SD, on 3 of them.
Unsteered samples for the hard cases and settled accepts come from run A (8 per call).
One job per call with all its conditions; hard cases, then settled rejects (item 1), then settled
accepts (item 2 only); longest prompt first within a group.
Writes agent-collusion/results/steer/jobs_session3.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
SETTLED_REJECTS = ["call.rep048.alice.ep01", "call.rep035.alice.ep01", "call.rep034.bob.ep02",
                   "call.rep024.alice.ep02", "call.rep040.bob.ep01"]  # vLLM share 0/16, C probe < 0.08
ACCEPT_PUSH_HARD = ["call.rep010.alice.ep03", "call.rep011.alice.ep02", "call.rep013.bob.ep03",
                    "call.rep013.bob.ep02", "call.rep043.alice.ep02"]  # lowest unsteered shares
REJECT_DONE_2SD = ["call.rep012.alice.ep04", "call.rep010.alice.ep03"]
RANDOM_REJECT = ["call.rep049.alice.ep03", "call.rep011.alice.ep02", "call.rep013.bob.ep03",
                 "call.rep013.bob.ep02", "call.rep043.alice.ep02"]
C_ONLY_2SD = ["call.rep012.alice.ep04", "call.rep049.alice.ep03", "call.rep036.bob.ep03", "call.rep015.alice.ep02",
              "call.rep048.alice.ep03", "call.rep025.bob.ep03", "call.rep042.alice.ep03", "call.rep024.alice.ep03"]
C_ONLY_4SD = ["call.rep036.bob.ep03", "call.rep015.alice.ep02", "call.rep025.bob.ep03"]


def main() -> None:
    reduced = {j["id"]: j for j in json.loads((RES / "steer" / "jobs_reduced.json").read_text())}
    meta = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl")) if c["job"] in SETTLED_REJECTS}
    an = {c["job"]: c for c in json.loads((RES / "step4_5" / "analysis.json").read_text())["calls"]}
    base = dict(reduced)
    for line in open(RES / "step6" / "jobs.jsonl"):
        j = json.loads(line)
        if j["id"] in SETTLED_REJECTS:
            ids = j["token_ids"][:meta[j["id"]]["think_start"]]
            assert meta[j["id"]]["c_token"] == len(ids) - 3
            base[j["id"]] = {"id": j["id"], "token_ids": ids, "c_token": meta[j["id"]]["c_token"], "target_sign": -1,
                             "group": "settled_reject", "share_A_vllm": an[j["id"]]["A"]["share"]}
    conds = {k: [] for k in base}

    def add(call, name, kind, vec, mult, where="from_c"):
        conds[call].append([name, kind, vec, mult, where])

    for c in SETTLED_REJECTS:
        add(c, "unsteered", "none", None, 0)
    for c in SETTLED_REJECTS + ACCEPT_PUSH_HARD:  # target_sign -1: a negative multiple pushes toward accept
        add(c, "toward_accept_2sd", "add", "target", -2)
    hard = [k for k, j in base.items() if j["group"] == "hard"]
    for c in hard:
        if c not in REJECT_DONE_2SD:
            add(c, "toward_reject_2sd", "add", "target", 2)
    for c in sorted(set(SETTLED_REJECTS + ACCEPT_PUSH_HARD + RANDOM_REJECT)):
        add(c, "random_2sd", "add", "random", 2)
    for c in C_ONLY_2SD:
        add(c, "c_only_reject_2sd", "add", "target", 2, "c_only")
    for c in C_ONLY_4SD:
        add(c, "c_only_reject_4sd", "add", "target", 4, "c_only")
        add(c, "c_only_random_4sd", "add", "random", 4, "c_only")
    order = {"hard": 0, "settled_reject": 1, "settled_accept": 2}
    jobs = [dict(base[k], conds=v, samples=6) for k, v in conds.items() if v]
    jobs.sort(key=lambda j: (order[j["group"]], -len(j["token_ids"])))
    out = RES / "steer" / "jobs_session3.json"
    out.write_text(json.dumps(jobs))
    for j in jobs:
        print(f"{j['id']:24s} {j['group']:15s} {len(j['token_ids']):6d}  " + ", ".join(c[0] for c in j["conds"]))
    print(len(jobs), "calls,", sum(len(j["conds"]) * 6 for j in jobs), "rows")


if __name__ == "__main__":
    main()
