"""Final round (Mac): own messages removed and previous feedback reversed, against the earlier
shares of the same calls.

    tools/.venv/bin/python tools/step4_5/session4_analyse.py

Per call (16 samples each): unedited (first session's resample from C), own messages removed and
feedback reversed (session 4), peer messages removed, whole conversation removed and previous
feedback removed (session 3). Reported per group (random, hard, and hard by subgroup) over the
calls that have every share in the row: mean shares, mean shift from unedited, calls moving by at
least 0.25 up / down, majority flips, invalid samples. Writes session4_analysis.json.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results" / "step4_5"


def shares(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("status") == "ok":
            v = [s["verdict"] for s in r["samples"] if s["verdict"]]
            if v:
                out[(r["job"], r["part"])] = (sum(x == "accept" for x in v) / len(v), len(r["samples"]) - len(v))
    return out


def main() -> None:
    an = {c["job"]: c for c in json.loads((RES / "analysis.json").read_text())["calls"]}
    s3, s4 = shares(RES / "session3_responses.jsonl"), shares(RES / "session4_responses.jsonl")
    rows = []
    for job, c in an.items():
        if not c["A"] or c["A"]["share"] is None:
            continue
        r = {"job": job, "group": "random" if c["groups"][0].startswith("random") else "hard",
             "subgroup": c["groups"][0], "unedited": c["A"]["share"]}
        for name, src, part in (("own_removed", s4, "own_messages"), ("feedback_reversed", s4, "feedback_reversed"),
                                ("peer_removed", s3, "peer_messages"), ("conversation_removed", s3, "conversation"),
                                ("feedback_removed", s3, "feedback_prev")):
            if (job, part) in src:
                r[name], r[name + "_invalid"] = src[(job, part)]
        rows.append(r)
    tables = {"conversation": ["own_removed", "peer_removed", "conversation_removed"],
              "feedback": ["feedback_reversed", "feedback_removed"]}
    out = {}
    for tname, cols in tables.items():
        out[tname] = {}
        for g in ("random", "hard", "flip", "reverted_reject", "confident_wrong"):
            rs = [r for r in rows if (r["group"] == g or r["subgroup"] == g) and all(k in r for k in cols)]
            if not rs:
                continue
            u = np.array([r["unedited"] for r in rs])
            e = {"calls": len(rs), "unedited": round(float(u.mean()), 3)}
            for k in cols:
                x = np.array([r[k] for r in rs])
                d = x - u
                e[k] = {"share": round(float(x.mean()), 3), "shift": round(float(d.mean()), 3),
                        "up_25": int(np.sum(d >= 0.25)), "down_25": int(np.sum(d <= -0.25)),
                        "majority_flips": int(np.sum((u > 0.5) != (x > 0.5))),
                        "invalid": int(sum(r[k + "_invalid"] for r in rs))}
            out[tname][g] = e
            print(tname, g, json.dumps(e))
    (RES / "session4_analysis.json").write_text(json.dumps({"tables": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
