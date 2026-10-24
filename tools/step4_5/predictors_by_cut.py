"""Steps 4-5 follow-up (Mac): at each cut of the saved thinking, what predicts the behavioural
ACCEPT share best?

    tools/.venv/bin/python tools/step4_5/predictors_by_cut.py

Behavioural share: first session's continuations from each cut (0% = full resample from C).
Predictors (all out of fold where fitted):
  probe_at_C       step 6's C-probe P(accept) at C (one value per call)
  stated_so_far    the latest decision the thinking has stated before the cut
                   (+1 accept, -1 reject, 0 none yet; same rule as verdict_statements.py)
  feedback         previous-episode feedback baseline (baselines_vs_resampling.py)
  probe_at_cut     extra, labelled: the C probe read at the cut token (step 6 per-token logits,
                   smoothed as in step 6); not a probe fitted at the cut
Reported per cut: Spearman with the share, random group and hard cases separately.
Writes agent-collusion/results/step4_5/predictors_by_cut.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step6"))
sys.path.insert(0, str(ROOT / "server"))
sys.argv = sys.argv[:1]
import analyse_step6 as a6  # noqa: E402
import verdict_statements as vsm  # noqa: E402

RES = ROOT / "agent-collusion" / "results"
CUTS = ["A", "fixed_30", "fixed_50", "fixed_70", "fixed_90", "fixed_100"]


def main() -> None:
    from scipy.stats import spearmanr
    import pins
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    an = json.loads((RES / "step4_5" / "analysis.json").read_text())["calls"]
    plan = {c["job"]: c for c in json.loads((RES / "step4_5" / "plan.json").read_text())["calls"]}
    calls6 = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl")) if c["kind"] == "call"}
    fb = {c["job"]: c["feedback"] for c in json.loads((RES / "step4_5" / "baselines_vs_resampling.json").read_text())["calls"]}
    rows = []
    for c in an:
        if not c["A"] or c["A"]["share"] is None:
            continue
        j = c["job"]
        m = calls6[j]
        text = m["reasoning"].strip()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        stmts = []
        for s in vsm.statements(text):
            k = next((i for i, (a, b) in enumerate(offs) if a <= s["char"] < b), None)
            if k is not None:
                stmts.append((k, s["polarity"]))
        lg = a6.read_safetensors(RES / "step6" / "acts" / f"{j}.safetensors")["probe_logits"].astype(float)
        sp = a6.smooth(a6.sigmoid(lg[m["think_start"] - m["c_token"]:]))
        cutk = {k["name"]: k["think_tokens"] for k in plan[j]["cuts"]}
        for cut in CUTS:
            share = c["A"]["share"] if cut == "A" else (c["B"].get(cut) or {}).get("share")
            if share is None:
                continue
            k = 0 if cut == "A" else cutk[cut]
            said = [p for t, p in stmts if t < k]
            rows.append({"job": j, "random": c["groups"][0].startswith("random"), "cut": cut, "share": share,
                         "probe_at_C": c["p_C"], "feedback": fb[j],
                         "stated_so_far": 0 if not said else (1 if said[-1] == "accept" else -1),
                         "probe_at_cut": float(sp[min(max(k - 1, 0), len(sp) - 1)]) if cut != "A" else c["p_C"]})
    out = {}
    for g in ("random", "hard"):
        out[g] = {}
        for cut in CUTS:
            rs = [r for r in rows if r["cut"] == cut and r["random"] == (g == "random")]
            share = np.array([r["share"] for r in rs])
            e = {"n": len(rs), "share_mean": round(float(share.mean()), 3),
                 "stated_any_yet": int(sum(r["stated_so_far"] != 0 for r in rs))}
            for p in ("probe_at_C", "stated_so_far", "feedback", "probe_at_cut"):
                x = np.array([r[p] for r in rs], float)
                e[p] = None if np.std(x) == 0 else round(float(spearmanr(x, share).statistic), 3)
            out[g][cut] = e
            print(g, cut, e, flush=True)
    (RES / "step4_5" / "predictors_by_cut.json").write_text(json.dumps({"by_group": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
