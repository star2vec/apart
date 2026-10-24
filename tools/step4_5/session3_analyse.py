"""Item 5 (Mac): ACCEPT share per ingredient removal against the unedited share.

    tools/.venv/bin/python tools/step4_5/session3_analyse.py

Unedited share: the first session's full resample from C (16 samples per call, same request
settings). Edited share: session 3's 16 samples per (call, removal); for an episode-2 call the
feedback_all row reuses feedback_prev (the same edit). Reported per removal, random group and hard
cases separately (and hard cases by subgroup): mean share unedited vs edited over the calls that
have the removal, mean shift, calls that move by at least 0.25 either way, calls whose majority
flips, invalid samples; and per call the shifts, written to session3_analysis.json.
"""

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results" / "step4_5"
REMOVALS = ["conversation", "peer_messages", "feedback_prev", "feedback_all", "reflection_prev"]


def main() -> None:
    an = {c["job"]: c for c in json.loads((RES / "analysis.json").read_text())["calls"]}
    edits = {}
    for l in open(RES / "session3_edits.jsonl"):
        r = json.loads(l)
        edits[(r["job"], r["removal"])] = r
    got = {}
    for l in open(RES / "session3_responses.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok":
            got[(r["job"], r["part"])] = r["samples"]
    for (job, rem), e in list(edits.items()):  # episode 2: feedback_all = feedback_prev
        if rem == "feedback_prev" and e["episode"] == 2 and (job, rem) in got:
            got.setdefault((job, "feedback_all"), got[(job, rem)])
    rows = []
    for (job, rem), samples in got.items():
        v = [s["verdict"] for s in samples if s["verdict"]]
        if not v:
            continue
        c = an[job]
        rows.append({"job": job, "removal": rem, "group": "random" if c["groups"][0].startswith("random") else "hard",
                     "subgroup": c["groups"][0], "base": c["A"]["share"], "edited": sum(x == "accept" for x in v) / len(v),
                     "n": len(samples), "valid": len(v)})
    out = {}
    for g in ("random", "hard"):
        out[g] = {}
        print(f"\n{g}")
        print(f"   {'removal':16s} {'calls':>5s} {'unedited':>8s} {'edited':>7s} {'shift':>6s} {'|shift|>=.25':>12s} {'majority flips':>14s} {'invalid':>7s}")
        for rem in REMOVALS:
            rs = [r for r in rows if r["group"] == g and r["removal"] == rem]
            if not rs:
                continue
            b, e = np.array([r["base"] for r in rs]), np.array([r["edited"] for r in rs])
            d = e - b
            flips = int(np.sum((b > 0.5) != (e > 0.5)))
            inv = sum(r["n"] - r["valid"] for r in rs)
            out[g][rem] = {"calls": len(rs), "unedited": round(float(b.mean()), 3), "edited": round(float(e.mean()), 3),
                           "shift": round(float(d.mean()), 3), "moved_25": int(np.sum(np.abs(d) >= 0.25)),
                           "up_25": int(np.sum(d >= 0.25)), "down_25": int(np.sum(d <= -0.25)), "majority_flips": flips,
                           "invalid": inv, "samples": sum(r["n"] for r in rs)}
            o = out[g][rem]
            print(f"   {rem:16s} {len(rs):5d} {o['unedited']:8.3f} {o['edited']:7.3f} {o['shift']:+6.3f} "
                  f"{o['moved_25']:5d} ({o['up_25']} up) {flips:9d} {inv:7d}")
        if g == "hard":
            sub = collections.defaultdict(dict)
            for rem in REMOVALS:
                for sg in sorted({r["subgroup"] for r in rows if r["group"] == "hard"}):
                    rs = [r for r in rows if r["subgroup"] == sg and r["removal"] == rem]
                    if rs:
                        sub[sg][rem] = (len(rs), round(float(np.mean([r["base"] for r in rs])), 3),
                                        round(float(np.mean([r["edited"] for r in rs])), 3))
            out["hard_by_subgroup"] = sub
            for sg, v in sub.items():
                print(f"   {sg:16s} " + "  ".join(f"{k}: {n} calls {b:.2f}->{e:.2f}" for k, (n, b, e) in v.items()))
    (RES / "session3_analysis.json").write_text(json.dumps({"by_group": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
